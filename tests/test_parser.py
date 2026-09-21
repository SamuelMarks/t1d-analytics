"""Tests for HTML parsing."""

from unittest.mock import Mock

import pytest
from bs4 import BeautifulSoup, Tag
from requests import HTTPError

from t1d_analytics.parser import extract_url, fetch_html, parse_datasets


def test_fetch_html_success(requests_mock: Mock) -> None:
    """Test successful HTML fetching."""
    requests_mock.get("http://test.com", text="<html></html>")
    assert fetch_html("http://test.com") == "<html></html>"


def test_fetch_html_error(requests_mock: Mock) -> None:
    """Test HTTP error during fetch."""
    requests_mock.get("http://test.com", status_code=404)
    with pytest.raises(HTTPError):
        fetch_html("http://test.com")


def test_extract_url_img_alt() -> None:
    """Test extracting URL from img alt attribute."""
    soup = BeautifulSoup('<td><img alt="http://url"></td>', "html.parser")
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert extract_url(td) == "http://url"


def test_extract_url_a_data_url() -> None:
    """Test extracting URL from a data-url attribute."""
    soup = BeautifulSoup('<td><a data-url="http://url2"></a></td>', "html.parser")
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert extract_url(td) == "http://url2"


def test_extract_url_none() -> None:
    """Test extracting URL when neither is present."""
    soup = BeautifulSoup("<td><span>Text</span></td>", "html.parser")
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert extract_url(td) is None


def test_extract_url_javascript_and_invalid_schemes() -> None:
    """Test extracting URL filters out javascript: and non-http schemes."""
    soup = BeautifulSoup(
        '<td><a href="javascript:__doPostBack()">Click</a></td>', "html.parser"
    )
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert extract_url(td) is None

    soup_ftp = BeautifulSoup(
        '<td><a href="ftp://files.example.com/data.zip">FTP</a></td>', "html.parser"
    )
    td_ftp = soup_ftp.find("td")
    assert isinstance(td_ftp, Tag)
    assert extract_url(td_ftp) is None


def test_parse_datasets_empty() -> None:
    """Test parsing HTML without table."""
    assert parse_datasets("<html></html>") == []


def test_parse_datasets_skip_headers() -> None:
    """Test parsing skips header rows."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr class="headerstyle"><td>Header</td></tr>
        <tr><td class="GroupHeaderStyle">Group</td></tr>
        Text Without Tag
    </table>
    """
    assert parse_datasets(html) == []


def test_parse_datasets_short_row() -> None:
    """Test parsing skips rows with too few cells."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr><td>Cell 1</td></tr>
    </table>
    """
    assert parse_datasets(html) == []


def test_parse_datasets_empty_protocol() -> None:
    """Test parsing skips rows with empty protocol name."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr>
            <td></td>
            <td></td><td></td><td></td><td></td><td></td>
        </tr>
    </table>
    """
    assert parse_datasets(html) == []


def test_parse_datasets_valid() -> None:
    """Test parsing a valid row."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr>
            <td>Valid Protocol</td>
            <td>ID</td><td>Year1</td><td>Year2</td>
            <td><img alt="http://dataset"></td>
            <td><a data-url="http://doc"></a></td>
        </tr>
    </table>
    """
    datasets = parse_datasets(html)
    assert len(datasets) == 1
    assert datasets[0].protocol == "Valid Protocol"
    assert datasets[0].dataset_url == "http://dataset"
    assert datasets[0].document_url == "http://doc"


def test_parse_datasets_empty_url() -> None:
    """Test parsing empty URLs resolves to None."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr>
            <td>Valid Protocol</td>
            <td>ID</td><td>Year1</td><td>Year2</td>
            <td><img alt=""></td>
            <td></td>
        </tr>
    </table>
    """
    datasets = parse_datasets(html)
    assert len(datasets) == 1
    assert datasets[0].protocol == "Valid Protocol"
    assert datasets[0].dataset_url is None
    assert datasets[0].document_url is None


def test_extract_url_standard_a_href() -> None:
    """Test extracting URL from standard a href attribute with relative path."""
    soup = BeautifulSoup(
        '<td><a href="/downloads/data.zip">Download</a></td>', "html.parser"
    )
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert (
        extract_url(td, base_url="https://public.t1d.org/base/")
        == "https://public.t1d.org/downloads/data.zip"
    )


def test_parse_datasets_multi_class_header() -> None:
    """Test parsing skips rows with multiple classes including headerstyle."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr class="headerstyle custom-header-theme"><td>Header</td></tr>
        <tr>
            <td>Valid Protocol</td>
            <td>ID</td><td>Year1</td><td>Year2</td>
            <td><a href="/relative/dataset.zip">Dataset</a></td>
            <td><a href="/relative/doc.pdf">Doc</a></td>
        </tr>
    </table>
    """
    datasets = parse_datasets(html, base_url="https://public.t1d.org/datasets/")
    assert len(datasets) == 1
    assert datasets[0].protocol == "Valid Protocol"
    assert datasets[0].dataset_url == "https://public.t1d.org/relative/dataset.zip"
    assert datasets[0].document_url == "https://public.t1d.org/relative/doc.pdf"


def test_extract_url_descriptive_alt_fallback() -> None:
    """Test extract_url ignores descriptive image alt text and extracts anchor href."""
    soup = BeautifulSoup(
        '<td><img src="icon.png" alt="Download File Icon"><a href="/files/data.csv">Download</a></td>',
        "html.parser",
    )
    td = soup.find("td")
    assert isinstance(td, Tag)
    assert (
        extract_url(td, base_url="https://public.t1d.org/datasets/")
        == "https://public.t1d.org/files/data.csv"
    )


def test_parse_datasets_generic_html5_table() -> None:
    """Test parsing standard HTML5 tables matching protocol and dataset keywords."""
    html = """
    <div>
        <table class="clinical-studies">
            <tr>
                <th>Protocol</th>
                <th>Dataset</th>
                <th>Documentation</th>
            </tr>
            <tr>
                <td>JAEB-T1D-01</td>
                <td><a href="/data/jaeb01.zip">Download Data</a></td>
                <td><a href="/docs/jaeb01.pdf">Manual</a></td>
            </tr>
            <tr>
                <td></td>
                <td><a href="/data/empty.zip">No Protocol</a></td>
            </tr>
            <tr>
                <td>Protocol</td>
                <td><a href="/data/dup_header.zip">Header Text</a></td>
            </tr>
            <tr>
                <td>SingleCellRow</td>
            </tr>
            <tr>
            </tr>
        </table>
        <table class="irrelevant">
            <tr><td>ignore this completely</td></tr>
        </table>
    </div>
    """
    datasets = parse_datasets(html, base_url="https://trials.org")
    assert len(datasets) == 1
    assert datasets[0].protocol == "JAEB-T1D-01"
    assert datasets[0].dataset_url == "https://trials.org/data/jaeb01.zip"
    assert datasets[0].document_url == "https://trials.org/docs/jaeb01.pdf"


def test_parse_datasets_tables_without_matching_keywords() -> None:
    """Test parsing when document has tables but none match protocol/dataset/study keywords."""
    html = "<html><body><table><tr><td>unrelated info</td></tr></table></body></html>"
    assert parse_datasets(html) == []


def test_extract_aspnet_form_data() -> None:
    """Test extracting ASP.NET hidden form state fields by name and by id."""
    from t1d_analytics.parser import extract_aspnet_form_data

    html = """
    <form method="post" action="./datasets.aspx" id="form1">
        <input type="hidden" name="__VIEWSTATE" id="__VIEWSTATE" value="/wEPDwULLTEx=" />
        <input type="hidden" name="__EVENTVALIDATION" value="/wEdAAM098=" />
        <input type="hidden" id="__VIEWSTATEGENERATOR" value="CA0B0334" />
        <input type="hidden" name="__EVENTTARGET" value="" />
        <input type="hidden" id="__EVENTARGUMENT" value="" />
        <input type="hidden" name="__PREVIOUSPAGE" value="prev_page_token" />
        <input type="text" name="other" value="ignored" />
    </form>
    """
    data = extract_aspnet_form_data(html)
    assert data["__VIEWSTATE"] == "/wEPDwULLTEx="
    assert data["__EVENTVALIDATION"] == "/wEdAAM098="
    assert data["__VIEWSTATEGENERATOR"] == "CA0B0334"
    assert data["__PREVIOUSPAGE"] == "prev_page_token"
    assert "__VIEWSTATE" in data
    assert "__EVENTTARGET" in data


def test_extract_postback_event() -> None:
    """Test extract_postback_event from various HTML element attributes."""
    from t1d_analytics.parser import extract_postback_event

    # 1. From href attribute
    soup1 = BeautifulSoup(
        "<a href=\"javascript:__doPostBack('ctl00$Grid','Select$0')\">Export</a>",
        "html.parser",
    )
    res1 = extract_postback_event(soup1.find("a"))  # type: ignore[arg-type]
    assert res1 == ("ctl00$Grid", "Select$0")

    # 2. From onclick attribute
    soup2 = BeautifulSoup(
        '<button onclick="__doPostBack(&quot;btnExport&quot;, &quot;arg1&quot;)">Click</button>',
        "html.parser",
    )
    res2 = extract_postback_event(soup2.find("button"))  # type: ignore[arg-type]
    assert res2 == ("btnExport", "arg1")

    # 3. From data-postback attribute
    soup3 = BeautifulSoup(
        "<span data-postback=\"__doPostBack('ctlTarget','cmdArg')\">Row</span>",
        "html.parser",
    )
    res3 = extract_postback_event(soup3.find("span"))  # type: ignore[arg-type]
    assert res3 == ("ctlTarget", "cmdArg")

    # 4. From nested child anchor
    soup4 = BeautifulSoup(
        "<td><a href=\"javascript:__doPostBack('nested$ctl','arg$2')\">Nested</a></td>",
        "html.parser",
    )
    res4 = extract_postback_event(soup4.find("td"))  # type: ignore[arg-type]
    assert res4 == ("nested$ctl", "arg$2")

    # 5. No postback present and child with no target attributes
    soup5 = BeautifulSoup(
        '<div><input type="text" name="txt" /><a href="https://example.com">Normal</a></div>',
        "html.parser",
    )
    res5 = extract_postback_event(soup5.find("div"))  # type: ignore[arg-type]
    assert res5 is None

    # 6. Child with data-postback
    soup6 = BeautifulSoup(
        '<td><button data-postback="__doPostBack(&#39;btn1&#39;,&#39;arg1&#39;)">Btn</button></td>',
        "html.parser",
    )
    res6 = extract_postback_event(soup6.find("td"))  # type: ignore[arg-type]
    assert res6 == ("btn1", "arg1")


def test_parse_datasets_with_postback() -> None:
    """Test parse_datasets extracts postback targets when direct URLs are absent."""
    html = """
    <table id="ctl00_CphMain_GridViewPublicDataSets">
        <tr>
            <td>PostBack Protocol</td>
            <td>ID</td><td>Year1</td><td>Year2</td>
            <td><a href="javascript:__doPostBack('ctl00$CphMain$GridViewPublicDataSets','Export$1')">Export Data</a></td>
            <td><a href="/docs/guide.pdf">Guide</a></td>
        </tr>
    </table>
    """
    datasets = parse_datasets(html, base_url="https://public.t1d.org/")
    assert len(datasets) == 1
    assert datasets[0].protocol == "PostBack Protocol"
    assert datasets[0].dataset_url is None
    assert datasets[0].postback_target == "ctl00$CphMain$GridViewPublicDataSets"
    assert datasets[0].postback_argument == "Export$1"
    assert datasets[0].document_url == "https://public.t1d.org/docs/guide.pdf"

    # Also test generic table with postback, and row with neither URL nor postback
    generic_html = """
    <table>
        <tr><th>Study</th><th>Export</th></tr>
        <tr>
            <td>Study A</td>
            <td><a onclick="__doPostBack('ctlGeneric','Export$0')">Export</a></td>
        </tr>
        <tr>
            <td>Study B</td>
            <td><span>Plain text without any link or postback</span></td>
        </tr>
    </table>
    """
    gen_datasets = parse_datasets(generic_html)
    assert len(gen_datasets) == 2
    assert gen_datasets[0].protocol == "Study A"
    assert gen_datasets[0].postback_target == "ctlGeneric"
    assert gen_datasets[0].postback_argument == "Export$0"
    assert gen_datasets[1].protocol == "Study B"
    assert gen_datasets[1].dataset_url is None
    assert gen_datasets[1].postback_target is None
