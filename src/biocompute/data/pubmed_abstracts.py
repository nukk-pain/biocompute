from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from typing import TypedDict

import httpx

from biocompute.data.pubmed import EUTILS_BASE, _throttled_get


DEFAULT_NCBI_TOOL = "biocompute"
DEFAULT_NCBI_EMAIL = "biocompute@example.invalid"


class PubMedAbstractRecord(TypedDict):
    pmid: str
    title: str
    abstract: str
    year: str | None
    authors: list[str]


def _text_content(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join(
        text.strip() for text in element.itertext() if text and text.strip()
    )


def _extract_abstract(article: ET.Element) -> str:
    abstract = article.find(".//Abstract")
    if abstract is None:
        return ""

    sections: list[str] = []
    for abstract_text in abstract.findall("AbstractText"):
        text = _text_content(abstract_text)
        if not text:
            continue
        label = (
            abstract_text.attrib.get("Label")
            or abstract_text.attrib.get("NlmCategory")
            or ""
        ).strip()
        if label:
            sections.append(f"{label}: {text}")
        else:
            sections.append(text)

    if sections:
        return "\n".join(sections)
    return _text_content(abstract)


def _extract_year(article: ET.Element) -> str | None:
    year_paths = [
        ".//PubDate/Year",
        ".//ArticleDate/Year",
        ".//DateCompleted/Year",
        ".//DateRevised/Year",
    ]
    for path in year_paths:
        year = _text_content(article.find(path))
        if year:
            return year
    medline_date = _text_content(article.find(".//PubDate/MedlineDate"))
    if medline_date:
        for token in medline_date.split():
            if len(token) == 4 and token.isdigit():
                return token
    return None


def _extract_authors(article: ET.Element) -> list[str]:
    authors: list[str] = []
    for author in article.findall(".//AuthorList/Author"):
        collective_name = _text_content(author.find("CollectiveName"))
        if collective_name:
            authors.append(collective_name)
            continue

        last_name = _text_content(author.find("LastName"))
        fore_name = _text_content(author.find("ForeName"))
        initials = _text_content(author.find("Initials"))

        if last_name and fore_name:
            authors.append(f"{fore_name} {last_name}")
        elif last_name and initials:
            authors.append(f"{initials} {last_name}")
        elif last_name:
            authors.append(last_name)
    return authors


def _parse_article(article: ET.Element) -> PubMedAbstractRecord | None:
    pmid = _text_content(article.find(".//PMID"))
    if not pmid:
        return None

    title = _text_content(article.find(".//ArticleTitle"))
    return {
        "pmid": pmid,
        "title": title,
        "abstract": _extract_abstract(article),
        "year": _extract_year(article),
        "authors": _extract_authors(article),
    }


def _build_params(pmids: list[str]) -> dict[str, str]:
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "rettype": "abstract",
        "retmode": "xml",
        "tool": os.environ.get("NCBI_TOOL", DEFAULT_NCBI_TOOL),
        "email": os.environ.get("NCBI_EMAIL", DEFAULT_NCBI_EMAIL),
    }
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


async def fetch_abstracts(
    client: httpx.AsyncClient,
    pmids: list[str],
    max_abstracts: int = 10,
) -> list[PubMedAbstractRecord]:
    """Fetch batched PubMed abstracts with fail-soft XML parsing."""
    limited_pmids = [
        pmid.strip() for pmid in pmids[:max_abstracts] if pmid and pmid.strip()
    ]
    if not limited_pmids:
        return []

    try:
        response = await _throttled_get(
            client,
            f"{EUTILS_BASE}/efetch.fcgi",
            params=_build_params(limited_pmids),
        )
    except (httpx.HTTPError, RuntimeError):
        return []

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError:
        return []

    records: list[PubMedAbstractRecord] = []
    for article in root.findall(".//PubmedArticle"):
        record = _parse_article(article)
        if record is not None:
            records.append(record)
    return records
