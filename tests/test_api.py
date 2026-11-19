import pandas as pd
import pytest
from fastapi import HTTPException

from api import main
from shopsmart.search import LocalLLMError


def test_search_endpoint_uses_ollama_terms_and_returns_catalog_matches(monkeypatch):
    products = pd.DataFrame(
        {
            "Product ID": [1, 2],
            "Product Name": ["Wireless Earbuds", "Table Lamp"],
            "Category": ["Electronics", "Home"],
        }
    )
    main.app.state.products = products
    monkeypatch.setattr(main, "expand_search_query", lambda *args: ["wireless earbuds"])

    response = main.search_products("something for music", limit=6, use_llm=True)

    assert response["expanded_terms"] == ["wireless earbuds"]
    assert response["results"][0]["Product ID"] == 1


def test_search_endpoint_reports_ollama_unavailable(monkeypatch):
    def fail_expansion(*args):
        raise LocalLLMError("Ollama is unavailable.")

    monkeypatch.setattr(main, "expand_search_query", fail_expansion)

    with pytest.raises(HTTPException) as exc_info:
        main.search_products("wireless earbuds", limit=6, use_llm=True)

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Ollama is unavailable."


def test_segment_endpoint_returns_not_found_for_unknown_customer():
    main.app.state.segments = {}

    with pytest.raises(HTTPException) as exc_info:
        main.get_customer_segment(123)

    assert exc_info.value.status_code == 404


def test_root_redirects_to_docs():
    response = main.root()

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"
