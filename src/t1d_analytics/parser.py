"""HTML parser for T1D public datasets."""

from typing import List, Optional

import requests
from bs4 import BeautifulSoup, Tag

from t1d_analytics.models import DatasetInfo


def fetch_html(url: str) -> str:
    """
    Fetch HTML content from a given URL.

    Args:
        url: The URL to fetch.

    Returns:
        The HTML content as a string.

    """
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.text


def extract_url(td: Tag) -> Optional[str]:
    """
    Extract dataset/document URL from a table cell.

    Args:
        td: A BeautifulSoup Tag representing a table cell.

    Returns:
        The extracted URL if found, else None.

    """
    img = td.find("img")
    if isinstance(img, Tag) and img.has_attr("alt"):
        return str(img["alt"])
    a_tag = td.find("a", attrs={"data-url": True})
    if isinstance(a_tag, Tag):
        return str(a_tag["data-url"])
    return None


def parse_datasets(html: str) -> List[DatasetInfo]:
    """
    Parse HTML to extract dataset information.

    Args:
        html: The HTML string to parse.

    Returns:
        A list of DatasetInfo objects.

    """
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="ctl00_CphMain_GridViewPublicDataSets")
    if not isinstance(table, Tag):
        return []

    datasets = []
    for row in table.find_all("tr"):
        if not isinstance(row, Tag):  # pragma: no cover
            continue

        if row.get("class") == ["headerstyle"] or row.find(
            "td", class_="GroupHeaderStyle"
        ):
            continue

        cells = row.find_all("td")
        if len(cells) < 6:
            continue

        protocol = cells[0].text.strip()
        if not protocol:
            continue

        dataset_url = extract_url(cells[4])
        document_url = extract_url(cells[5])

        datasets.append(
            DatasetInfo(
                protocol=protocol,
                dataset_url=dataset_url if dataset_url else None,
                document_url=document_url if document_url else None,
            )
        )

    return datasets
