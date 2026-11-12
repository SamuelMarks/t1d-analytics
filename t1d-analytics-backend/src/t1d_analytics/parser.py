"""HTML parser for T1D public datasets."""

from collections.abc import Sequence
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag

from t1d_analytics.errors import DownloadError
from t1d_analytics.models import DatasetInfo

DEFAULT_MAX_HTML_BYTES = 10 * 1024 * 1024
_ALLOWED_HTML_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
)


def fetch_html(url: str, max_bytes: int = DEFAULT_MAX_HTML_BYTES) -> str:
    """
    Fetch HTML content from `url` with strict `Content-Type` validation and streaming byte-size caps.

    Args:
        url: The HTTP/HTTPS URL to fetch.
        max_bytes: Maximum allowed response payload size in bytes (default: 10 MiB).

    Returns:
        str: Decoded HTML document text.

    Raises:
        DownloadError: If `Content-Type` is non-HTML or the response exceeds `max_bytes`.
        requests.HTTPError: If the HTTP status code indicates an error.

    """
    response = requests.get(url, timeout=30, stream=True)
    try:
        response.raise_for_status()
        raw_ct = (response.headers.get("Content-Type") or "").strip()
        if raw_ct:
            media_type = raw_ct.split(";", 1)[0].strip().lower()
            if media_type and media_type not in _ALLOWED_HTML_CONTENT_TYPES:
                raise DownloadError(
                    f"Unexpected non-HTML Content-Type '{media_type}' from '{url}'.",
                    params={"url": url, "content_type": media_type},
                )

        raw_cl = (response.headers.get("Content-Length") or "").strip()
        if raw_cl.isdigit() and int(raw_cl) > max_bytes:
            raise DownloadError(
                f"HTML response Content-Length ({raw_cl} bytes) exceeds limit ({max_bytes} bytes).",
                params={"url": url, "max_bytes": max_bytes},
            )

        chunks: list[bytes] = []
        total = 0
        for chunk in response.iter_content(chunk_size=65536):
            if not chunk:
                continue
            total += len(chunk)
            if total > max_bytes:
                raise DownloadError(
                    f"HTML response body from '{url}' exceeded maximum size ({max_bytes} bytes).",
                    params={"url": url, "max_bytes": max_bytes},
                )
            chunks.append(chunk)

        encoding = response.encoding or "utf-8"
        return b"".join(chunks).decode(encoding, errors="replace")
    finally:
        response.close()


def extract_url(td: Tag, base_url: str | None = None) -> str | None:
    """
    Extract dataset/document URL from a table cell and optionally resolve relative URLs.

    Args:
        td: A BeautifulSoup Tag representing a table cell.
        base_url: Optional base URL to resolve relative URLs against.

    Returns:
        The extracted URL if found, else None.

    """
    url: str | None = None
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
        html: Raw HTML string of the ASP.NET page.

    Returns:
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


def extract_postback_event(tag: Tag) -> tuple[str, str] | None:
    """
    Extract __doPostBack(eventTarget, eventArgument) parameters from an element or its children.

    Inspects href, onclick, or data-postback attributes on the tag and nested elements.

    Args:
        tag: BeautifulSoup Tag to inspect.

    Returns:
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


def _parse_dataset_row(
    cells: Sequence[Tag],
    is_standard_grid: bool,
    base_url: str | None,
) -> DatasetInfo | None:
    """
    Parse a single HTML table row's `<td>` cells into a `DatasetInfo` instance.

    Args:
        cells: Sequence of `<td>` tags in the row.
        is_standard_grid: Whether the table is the standard 6-column ASP.NET GridView.
        base_url: Optional base URL to resolve relative links.

    Returns:
        Optional[DatasetInfo]: Parsed dataset record, or None if the row is not a data row.

    """
    min_cells = 6 if is_standard_grid else 2
    if len(cells) < min_cells:
        return None

    protocol = cells[0].text.strip()
    if not protocol or (
        not is_standard_grid and protocol.lower() in ("protocol", "study", "dataset")
    ):
        return None

    ds_cell = cells[4] if is_standard_grid else cells[1]
    doc_cell: Tag | None = (
        cells[5] if is_standard_grid else (cells[2] if len(cells) > 2 else None)
    )
    dataset_url = extract_url(ds_cell, base_url=base_url)
    document_url = (
        extract_url(doc_cell, base_url=base_url) if doc_cell is not None else None
    )

    postback_target: str | None = None
    postback_argument: str | None = None
    if not dataset_url:
        postback = extract_postback_event(ds_cell)
        if postback:
            postback_target, postback_argument = postback

    return DatasetInfo(
        protocol=protocol,
        dataset_url=dataset_url if dataset_url else None,
        document_url=document_url if document_url else None,
        postback_target=postback_target,
        postback_argument=postback_argument,
    )


def parse_datasets(html: str, base_url: str | None = None) -> list[DatasetInfo]:
    """
    Parse HTML to extract dataset information.

    Supports ASP.NET GridView tables and generic HTML5 clinical tables.

    Args:
        html: The HTML string to parse.
        base_url: Optional base URL to resolve relative links.

    Returns:
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

    datasets: list[DatasetInfo] = []
    for row in table.find_all("tr"):
        row_classes = row.get("class") or []
        if "headerstyle" in row_classes or row.find("td", class_="GroupHeaderStyle"):
            continue
        parsed = _parse_dataset_row(row.find_all("td"), is_standard_grid, base_url)
        if parsed is not None:
            datasets.append(parsed)

    return datasets
