# pyright: reportMissingImports=false

from unittest.mock import AsyncMock, patch

import httpx
import pytest
from _pytest.monkeypatch import MonkeyPatch

from biocompute.data.pubmed import search_pubmed
from biocompute.data.pubmed_abstracts import fetch_abstracts
from biocompute.data.semantic_scholar import search_papers, _rate_limited_get


@pytest.mark.asyncio
async def test_search_pubmed_parses_id_list():
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        ),
        json={"esearchresult": {"idlist": ["12345", "67890"]}},
    )
    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            ids = await search_pubmed(client, "CXCL12 scar pain", max_results=5)
    assert ids == ["12345", "67890"]


@pytest.mark.asyncio
async def test_search_pubmed_empty_result():
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        ),
        json={"esearchresult": {"idlist": []}},
    )
    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            ids = await search_pubmed(client, "nonexistent query xyz", max_results=5)
    assert ids == []


@pytest.mark.asyncio
async def test_fetch_abstracts_batches_pmids_and_parses_xml(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("NCBI_API_KEY", "test-key")
    monkeypatch.setenv("NCBI_EMAIL", "tester@example.com")
    monkeypatch.setenv("NCBI_TOOL", "biocompute-tests")

    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>12345</PMID>
          <Article>
            <ArticleTitle>First study</ArticleTitle>
            <Abstract>
              <AbstractText>Primary abstract text.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <LastName>Smith</LastName>
                <ForeName>Jane</ForeName>
              </Author>
            </AuthorList>
            <Journal>
              <JournalIssue>
                <PubDate><Year>2024</Year></PubDate>
              </JournalIssue>
            </Journal>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>67890</PMID>
          <Article>
            <ArticleTitle>Second study</ArticleTitle>
            <Abstract>
              <AbstractText Label="BACKGROUND">Background text.</AbstractText>
              <AbstractText Label="METHODS">Methods text.</AbstractText>
            </Abstract>
            <AuthorList>
              <Author>
                <CollectiveName>Trial Consortium</CollectiveName>
              </Author>
            </AuthorList>
            <Journal>
              <JournalIssue>
                <PubDate><MedlineDate>2021 Jan-Feb</MedlineDate></PubDate>
              </JournalIssue>
            </Journal>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        ),
        text=xml_payload,
    )
    mock_get = AsyncMock(return_value=mock_response)

    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        mock_get,
    ):
        async with httpx.AsyncClient() as client:
            records = await fetch_abstracts(
                client, ["12345", "67890", "99999"], max_abstracts=2
            )

    assert mock_get.await_count == 1
    awaited_call = mock_get.await_args
    assert awaited_call is not None
    params = awaited_call.kwargs["params"]
    assert params["db"] == "pubmed"
    assert params["id"] == "12345,67890"
    assert params["rettype"] == "abstract"
    assert params["retmode"] == "xml"
    assert params["api_key"] == "test-key"
    assert params["email"] == "tester@example.com"
    assert params["tool"] == "biocompute-tests"

    assert records == [
        {
            "pmid": "12345",
            "title": "First study",
            "abstract": "Primary abstract text.",
            "year": "2024",
            "authors": ["Jane Smith"],
        },
        {
            "pmid": "67890",
            "title": "Second study",
            "abstract": "BACKGROUND: Background text.\nMETHODS: Methods text.",
            "year": "2021",
            "authors": ["Trial Consortium"],
        },
    ]


@pytest.mark.asyncio
async def test_fetch_abstracts_handles_missing_abstract_without_crashing():
    xml_payload = """<?xml version="1.0" encoding="UTF-8"?>
    <PubmedArticleSet>
      <PubmedArticle>
        <MedlineCitation>
          <PMID>11111</PMID>
          <Article>
            <ArticleTitle>No abstract paper</ArticleTitle>
            <AuthorList>
              <Author>
                <LastName>Doe</LastName>
                <Initials>J</Initials>
              </Author>
            </AuthorList>
          </Article>
        </MedlineCitation>
      </PubmedArticle>
    </PubmedArticleSet>
    """
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        ),
        text=xml_payload,
    )
    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            records = await fetch_abstracts(client, ["11111"])

    assert records == [
        {
            "pmid": "11111",
            "title": "No abstract paper",
            "abstract": "",
            "year": None,
            "authors": ["J Doe"],
        }
    ]


@pytest.mark.asyncio
async def test_fetch_abstracts_returns_empty_list_on_malformed_xml():
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        ),
        text="<PubmedArticleSet><PubmedArticle>",
    )
    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            records = await fetch_abstracts(client, ["12345"])

    assert records == []


@pytest.mark.asyncio
async def test_fetch_abstracts_returns_empty_list_on_http_failure():
    failing_response = httpx.Response(
        429,
        request=httpx.Request(
            "GET", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        ),
    )
    mock_get = AsyncMock(return_value=failing_response)
    with patch(
        "biocompute.data.pubmed.httpx.AsyncClient.get",
        mock_get,
    ):
        async with httpx.AsyncClient() as client:
            records = await fetch_abstracts(client, ["12345"])

    assert records == []
    assert mock_get.await_count == 3


@pytest.mark.asyncio
async def test_search_papers_semantic_scholar():
    mock_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
        ),
        json={
            "data": [
                {
                    "paperId": "abc123",
                    "title": "CXCL12 in pain",
                    "citationCount": 42,
                    "influentialCitationCount": 5,
                }
            ]
        },
    )
    with patch(
        "biocompute.data.semantic_scholar.httpx.AsyncClient.get",
        new_callable=AsyncMock,
        return_value=mock_response,
    ):
        async with httpx.AsyncClient() as client:
            papers = await search_papers(client, "CXCL12 scar", limit=5)
    assert len(papers) == 1
    assert papers[0]["title"] == "CXCL12 in pain"
    assert papers[0]["citationCount"] == 42


@pytest.mark.asyncio
async def test_rate_limited_get_retries_on_429():
    """429 responses trigger exponential backoff retries."""
    rate_limited_response = httpx.Response(
        429,
        request=httpx.Request(
            "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
        ),
    )
    ok_response = httpx.Response(
        200,
        request=httpx.Request(
            "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
        ),
        json={"data": []},
    )
    mock_get = AsyncMock(side_effect=[rate_limited_response, ok_response])
    with patch(
        "biocompute.data.semantic_scholar.httpx.AsyncClient.get",
        mock_get,
    ):
        async with httpx.AsyncClient() as client:
            response = await _rate_limited_get(
                client, "https://api.semanticscholar.org/graph/v1/paper/search"
            )
    assert response.status_code == 200
    assert mock_get.call_count == 2


@pytest.mark.asyncio
async def test_rate_limited_get_exhausts_retries_on_persistent_429():
    """If all retries get 429, the last 429 response is returned."""
    rate_limited_response = httpx.Response(
        429,
        request=httpx.Request(
            "GET", "https://api.semanticscholar.org/graph/v1/paper/search"
        ),
    )
    mock_get = AsyncMock(return_value=rate_limited_response)
    with patch(
        "biocompute.data.semantic_scholar.httpx.AsyncClient.get",
        mock_get,
    ):
        async with httpx.AsyncClient() as client:
            response = await _rate_limited_get(
                client, "https://api.semanticscholar.org/graph/v1/paper/search"
            )
    assert response.status_code == 429
    assert mock_get.call_count == 3
