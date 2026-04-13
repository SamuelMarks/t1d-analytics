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
