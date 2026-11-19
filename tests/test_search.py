import json
from urllib.error import URLError

import pandas as pd
import pytest

from shopsmart.search import LocalLLMError, expand_search_query, search_catalog


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = json.dumps(payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.payload


def test_query_expansion_uses_local_ollama(monkeypatch):
    generated = json.dumps({"search_terms": ["weighted blanket", "sleep mask"]})
    captured = {}

    def fake_urlopen(http_request, timeout):
        captured["request"] = http_request
        captured["timeout"] = timeout
        return FakeResponse({"response": generated})

    monkeypatch.setattr("shopsmart.search.request.urlopen", fake_urlopen)

    terms = expand_search_query("something to help me sleep", "http://localhost:11434/", "qwen2.5:1.5b")

    assert terms == ["weighted blanket", "sleep mask"]
    assert captured["request"].full_url == "http://localhost:11434/api/generate"
    assert captured["timeout"] == 45.0
    request_payload = json.loads(captured["request"].data)
    assert request_payload["model"] == "qwen2.5:1.5b"
    assert request_payload["stream"] is False


def test_query_expansion_surfaces_ollama_errors(monkeypatch):
    def fail_to_connect(http_request, timeout):
        raise URLError("connection refused")

    monkeypatch.setattr("shopsmart.search.request.urlopen", fail_to_connect)

    with pytest.raises(LocalLLMError, match="Could not connect to Ollama"):
        expand_search_query("wireless earbuds", "http://localhost:11434", "qwen2.5:1.5b")


def test_catalog_search_finds_products_from_expanded_terms():
    products = pd.DataFrame(
        {
            "Product ID": [1, 2, 3],
            "Product Name": ["Weighted Blanket", "Smartwatch", "Travel Pillow"],
            "Category": ["Home", "Electronics", "Travel"],
            "Description": [
                "A soft blanket designed to support a restful night's sleep.",
                "Tracks sleep and daily activity.",
                "Comfortable support for long journeys.",
            ],
        }
    )

    results = search_catalog(products, ["something to help me sleep", "weighted blanket"])

    assert results["Product ID"].tolist()[0] == 1
    assert 2 in results["Product ID"].tolist()


def test_catalog_search_rejects_non_positive_result_limit():
    products = pd.DataFrame({"Product ID": [1], "Product Name": ["Lamp"]})

    with pytest.raises(ValueError, match="limit must be greater than zero"):
        search_catalog(products, ["lamp"], limit=0)
