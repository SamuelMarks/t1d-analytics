"""HTML parser for T1D public datasets."""

from typing import List, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from t1d_analytics.models import DatasetInfo


def fetch_html(url: str) -> str:
    """
    Fetch HTML content from a given URL.

    Args:
    ----
        url: The URL to fetch.

    Returns:
    -------
        The HTML content as a string.

    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def extract_url(td: Tag, base_url: Optional[str] = None) -> Optional[str]:
    """
    Extract dataset/document URL from a table cell and optionally resolve relative URLs.

    Args:
    ----
        td: A BeautifulSoup Tag representing a table cell.
        base_url: Optional base URL to resolve relative URLs against.

    Returns:
    -------
        The extracted URL if found, else None.

    """
    url: Optional[str] = None
    img = td.find("img")
    if isinstance(img, Tag) and img.has_attr("alt"):
        candidate = str(img["alt"]).strip()
        if candidate.lower().startswith(("http://", "https://", "/")):
            url = candidate
    if not url:
        a_tag = td.find("a", attrs={"data-url": True})
        if isinstance(a_tag, Tag):
            url = str(a_tag["data-url"]).strip()
    if not url:
        a_href = td.find("a", href=True)
        if isinstance(a_href, Tag) and a_href.get("href"):
            url = str(a_href["href"]).strip()

    if url:
        url_lower = url.lower()
        if url_lower.startswith(("javascript:", "data:", "mailto:", "vbscript:")):
            return None
        if base_url:
            url = urljoin(base_url, url)
        parsed_scheme = urlparse(url).scheme.lower()
        if parsed_scheme not in ("http", "https"):
            return None
    return url


def extract_aspnet_form_data(html: str) -> dict[str, str]:
    """
    Extract ASP.NET hidden form state fields from HTML content.

    Captures standard ASP.NET fields (__VIEWSTATE, __EVENTVALIDATION,
    __VIEWSTATEGENERATOR, __EVENTTARGET, __EVENTARGUMENT, __PREVIOUSPAGE)
    required for simulated form postback submissions.

    Args:
    ----
        html: Raw HTML string of the ASP.NET page.

    Returns:
    -------
        dict[str, str]: Dictionary of extracted form field names and their values.

    """
    soup = BeautifulSoup(html, "html.parser")
    form_data: dict[str, str] = {}
    target_fields = [
        "__VIEWSTATE",
        "__EVENTVALIDATION",
        "__VIEWSTATEGENERATOR",
        "__EVENTTARGET",
        "__EVENTARGUMENT",
        "__PREVIOUSPAGE",
    ]
    for field_name in target_fields:
        input_tag = soup.find("input", attrs={"name": field_name})
        if isinstance(input_tag, Tag) and input_tag.has_attr("value"):
            form_data[field_name] = str(input_tag["value"])
        else:
            input_tag_id = soup.find("input", attrs={"id": field_name})
            if isinstance(input_tag_id, Tag) and input_tag_id.has_attr("value"):
                form_data[field_name] = str(input_tag_id["value"])

    return form_data


def extract_postback_event(tag: Tag) -> Optional[tuple[str, str]]:
    """
    Extract __doPostBack(eventTarget, eventArgument) parameters from an element or its children.

    Inspects href, onclick, or data-postback attributes on the tag and nested elements.

    Args:
    ----
        tag: BeautifulSoup Tag to inspect.

    Returns:
    -------
        Optional[tuple[str, str]]: Tuple of (event_target, event_argument) if found, else None.

    """
    import re

    candidates: list[str] = []
    elements: list[Tag] = [tag]
    for el in tag.find_all(["a", "button", "input"]):
        elements.append(el)

    for el in elements:
        for attr in ("href", "onclick", "data-postback"):
            val = el.get(attr)
            if val:
                candidates.append(str(val))

    pattern = re.compile(
        r"__doPostBack\(\s*['\"]([^'\"]*)['\"]\s*,\s*['\"]([^'\"]*)['\"]\s*\)"
    )
    for text in candidates:
        match = pattern.search(text)
        if match:
            return match.group(1), match.group(2)
    return None


def parse_datasets(html: str, base_url: Optional[str] = None) -> List[DatasetInfo]:
    """
    Parse HTML to extract dataset information.

    Supports ASP.NET GridView tables and generic HTML5 clinical tables.

    Args:
    ----
        html: The HTML string to parse.
        base_url: Optional base URL to resolve relative links.

    Returns:
    -------
        A list of DatasetInfo objects.

    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="ctl00_CphMain_GridViewPublicDataSets")
    is_standard_grid = isinstance(table, Tag)
    if not is_standard_grid:
        for cand in soup.find_all("table"):
            text = cand.text.lower()
            if "protocol" in text or "dataset" in text or "study" in text:
                table = cand
                break

    if not isinstance(table, Tag):
        return []

    datasets = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):  # pragma: no cover
            continue

        row_classes = row.get("class") or []
        if "headerstyle" in row_classes or row.find("td", class_="GroupHeaderStyle"):
            continue

        cells = row.find_all("td")
        if not cells:
            continue

        postback_target: Optional[str] = None
        postback_argument: Optional[str] = None

        if is_standard_grid:
            if len(cells) < 6:
                continue
            protocol = cells[0].text.strip()
            if not protocol:
                continue
            dataset_url = extract_url(cells[4], base_url=base_url)
            document_url = extract_url(cells[5], base_url=base_url)
            if not dataset_url:
                postback = extract_postback_event(cells[4])
                if postback:
                    postback_target, postback_argument = postback
        else:
            if len(cells) < 2:
                continue
            protocol = cells[0].text.strip()
            if not protocol or protocol.lower() in ("protocol", "study", "dataset"):
                continue
            dataset_url = extract_url(cells[1], base_url=base_url)
            document_url = (
                extract_url(cells[2], base_url=base_url) if len(cells) > 2 else None
            )
            if not dataset_url:
                postback = extract_postback_event(cells[1])
                if postback:
                    postback_target, postback_argument = postback

        datasets.append(
            DatasetInfo(
                protocol=protocol,
                dataset_url=dataset_url if dataset_url else None,
                document_url=document_url if document_url else None,
                postback_target=postback_target,
                postback_argument=postback_argument,
            )
        )

    return datasets
