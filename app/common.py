"""Shared data loaders, styles and helpers for the ShopSmart AI Streamlit app."""
from __future__ import annotations

import json
import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from shopsmart.config import CONFIG, path
from shopsmart.data import load
from shopsmart.recommender import RecommenderSuite

load_dotenv()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")

GREEN = "#1F6B55"
MUTED_GREEN = "#A9BDB2"

APP_CSS = """
<style>
.block-container {padding-top: 2.2rem; max-width: 1280px;}
.ss-kicker {color: #1F6B55; font-size: 0.8rem; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; margin-bottom: -0.6rem;}
.ss-subtitle {color: #4F6159; font-size: 1.05rem; margin-top: -0.6rem;}
.ss-card {border: 1px solid #E1EAE4; border-radius: 14px; padding: 1rem 1.1rem; background: #FFFFFF; height: 100%;}
.ss-card-rank {color: #1F6B55; font-size: 0.78rem; font-weight: 700;}
.ss-card-name {color: #1B3229; font-size: 1.02rem; font-weight: 700; line-height: 1.3; margin: 0.35rem 0 0.2rem;}
.ss-card-meta {color: #4F6159; font-size: 0.88rem;}
.ss-card-price {color: #1B3229; font-size: 1.25rem; font-weight: 700; margin-top: 0.6rem;}
.ss-card-stock {color: #1F6B55; font-size: 0.82rem; font-weight: 600;}
.ss-problem {font-size: 2rem; font-weight: 700; color: #B4471F; line-height: 1.1;}
.ss-brand {font-weight: 700; font-size: 1.15rem; color: #1B3229;}
.ss-brand-sub {color: #4F6159; font-size: 0.85rem; margin-top: -0.2rem;}
.ss-footer {color: #4F6159; font-size: 0.78rem; line-height: 1.6;}
.ss-dot {display: inline-block; width: 8px; height: 8px; border-radius: 4px; margin-right: 6px;}
</style>
"""


def page_header(kicker: str, title: str, subtitle: str | None = None) -> None:
    st.markdown(APP_CSS, unsafe_allow_html=True)
    st.markdown(f'<div class="ss-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    if subtitle:
        st.markdown(f'<div class="ss-subtitle">{subtitle}</div>', unsafe_allow_html=True)
        st.write("")


# --------------------------------------------------------------------------- data
@st.cache_data(show_spinner=False)
def load_core_data() -> dict[str, pd.DataFrame]:
    return {name: load(name) for name in ("products", "users", "interactions", "search_queries", "faq")}


@st.cache_data(show_spinner=False)
def load_segments(modified_at: float) -> pd.DataFrame:
    file = path("processed_data") / CONFIG["segmentation"]["output_file"]
    return pd.read_csv(file) if file.exists() else pd.DataFrame(columns=["User ID", "segment"])


def segments() -> pd.DataFrame:
    file = path("processed_data") / CONFIG["segmentation"]["output_file"]
    return load_segments(file.stat().st_mtime if file.exists() else 0.0)


@st.cache_data(show_spinner=False)
def load_profile(modified_at: float) -> pd.DataFrame:
    file = path("reports") / "segmentation_profile.csv"
    return pd.read_csv(file).set_index("segment").sort_index() if file.exists() else pd.DataFrame()


def profile() -> pd.DataFrame:
    file = path("reports") / "segmentation_profile.csv"
    return load_profile(file.stat().st_mtime if file.exists() else 0.0)


@st.cache_data(show_spinner=False)
def load_json(name: str, modified_at: float) -> dict:
    with open(path("reports") / name, encoding="utf-8") as f:
        return json.load(f)


def report_json(name: str) -> dict | None:
    file = path("reports") / name
    return load_json(name, file.stat().st_mtime) if file.exists() else None


@st.cache_resource(show_spinner=False)
def load_suite(modified_at: float) -> RecommenderSuite:
    return RecommenderSuite.load(path("models") / CONFIG["recommender"]["model_file"])


def recommender_suite() -> RecommenderSuite | None:
    file = path("models") / CONFIG["recommender"]["model_file"]
    return load_suite(file.stat().st_mtime) if file.exists() else None


@st.cache_resource(show_spinner=False)
def load_two_tower(modified_at: float, suite_modified_at: float):
    from shopsmart.recommender.two_tower import TwoTowerRecommender

    file = path("models") / CONFIG.get("two_tower", {}).get("model_file", "two_tower.npz")
    return TwoTowerRecommender.load(file, recommender_suite().popularity)


def two_tower_model():
    """The trained two-tower (deep learning) recommender, or None if it has not been trained yet."""
    file = path("models") / CONFIG.get("two_tower", {}).get("model_file", "two_tower.npz")
    suite_file = path("models") / CONFIG["recommender"]["model_file"]
    if not file.exists() or not suite_file.exists():
        return None
    try:
        return load_two_tower(file.stat().st_mtime, suite_file.stat().st_mtime)
    except ValueError:  # trained on a different catalogue: retrain it
        return None


# --------------------------------------------------------------------------- segments
def segment_name(segment: int | None) -> str:
    if segment is None or pd.isna(segment):
        return "Not segmented yet"
    names = CONFIG.get("app", {}).get("segment_names", {})
    return names.get(int(segment), f"Segment {int(segment)}")


def segment_campaign(segment: int) -> str:
    return CONFIG.get("app", {}).get("segment_campaigns", {}).get(int(segment), "No campaign defined yet.")


def customer_segment(user_id: int) -> int | None:
    seg = segments()
    match = seg.loc[seg["User ID"] == user_id, "segment"]
    return int(match.iloc[0]) if len(match) else None


# --------------------------------------------------------------------------- business KPIs
@st.cache_data(show_spinner=False)
def business_kpis() -> dict:
    data = load_core_data()
    ev, products = data["interactions"], data["products"]
    buys = ev[ev["Interaction Type"] == "purchase"].copy()
    buys["revenue"] = buys["Price"] * buys["Quantity"].astype("Float64").fillna(1).astype(float)
    revenue = float(buys["revenue"].sum())
    sessions = ev.groupby("Session ID")["Interaction Type"].apply(lambda s: (s == "purchase").any())
    carts = int((ev["Interaction Type"] == "add_to_cart").sum())
    by_product = buys.groupby("Product ID")["revenue"].sum().sort_values(ascending=False)
    top_n = max(1, len(products) // 10)
    return {
        "customers": int(data["users"]["User ID"].nunique()),
        "products": int(len(products)),
        "interactions": int(len(ev)),
        "revenue": revenue,
        "no_purchase_sessions": 1 - float(sessions.mean()),
        "cart_abandonment": 1 - len(buys) / carts if carts else 0.0,
        "top10_revenue_share": float(by_product.head(top_n).sum() / revenue) if revenue else 0.0,
    }


# --------------------------------------------------------------------------- product cards
def product_cards(rows: pd.DataFrame, label: str | None = None, columns: int = 3) -> None:
    """Render products as cards in a grid. `label` may contain {rank}."""
    from html import escape

    rows = rows.reset_index(drop=True)
    for start in range(0, len(rows), columns):
        cols = st.columns(columns)
        for col, (i, row) in zip(cols, rows.iloc[start:start + columns].iterrows()):
            rating = row.get("Avg Rating")
            rating_text = f"★ {float(rating):.1f}" if pd.notna(rating) else "No reviews yet"
            stock = row.get("In Stock")
            days = row.get("Delivery Days")
            stock_text = (f"In stock · {int(days)}-day delivery" if pd.notna(days) else "In stock") if stock else "Out of stock"
            top = f'<div class="ss-card-rank">{escape(label.format(rank=i + 1))}</div>' if label else ""
            col.markdown(
                f"""<div class="ss-card">{top}
                <div class="ss-card-name">{escape(str(row["Product Name"]))}</div>
                <div class="ss-card-meta">{escape(str(row["Category"]))} · {rating_text}</div>
                <div class="ss-card-price">£{float(row["Price"]):.2f}</div>
                <div class="ss-card-stock">{stock_text}</div></div>""",
                unsafe_allow_html=True,
            )
