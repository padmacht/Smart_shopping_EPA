"""Tests for the rule-based chat logic - run with: pytest"""
import pytest

from shopsmart.assistant import detect_intent, extract_max_price, faq_answer, strip_price
from shopsmart.data import load


@pytest.mark.parametrize("text, intent", [
    ("hi", "greeting"),
    ("thanks!", "thanks"),
    ("where is my order?", "order"),
    ("I want to cancel my order", "order"),
    ("how long does delivery take?", "faq"),
    ("can I pay with klarna", "faq"),
    ("what should I buy next?", "recommend"),
    ("recommend something for me", "recommend"),
    ("something to help me sleep under £60", "search"),
    ("hi, I need a yoga mat for my sister", "search"),
])
def test_detect_intent(text, intent):
    assert detect_intent(text) == intent


def test_faq_answer_finds_returns_policy():
    row = faq_answer("what is your returns policy?", load("faq"))
    assert row is not None and row["Topic"] == "returns"


def test_faq_answer_returns_none_for_unrelated_question():
    assert faq_answer("do you sell cars?", load("faq")) is None


@pytest.mark.parametrize("text, price", [("under £60", 60.0), ("below 30 please", 30.0),
                                          ("less than £25.50", 25.5), ("a yoga mat", None)])
def test_extract_max_price(text, price):
    assert extract_max_price(text) == price


def test_strip_price_keeps_the_product_words():
    assert strip_price("weighted blanket under £60") == "weighted blanket"
