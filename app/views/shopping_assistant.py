"""Shopping assistant page: search or chat, plus 'What should I buy next?'."""
from __future__ import annotations

from collections import Counter

import pandas as pd
import streamlit as st

from common import (OLLAMA_BASE_URL, OLLAMA_MODEL, customer_segment, load_core_data, page_header,
                    product_cards, recommender_suite, two_tower_model)
from shopsmart.assistant import (HELP_TEXT, ORDER_TEXT, detect_intent, extract_max_price, faq_answer,
                                 strip_price)
from shopsmart.search import LocalLLMError, expand_search_query, search_catalog

MODELS = {
    "Hybrid (ML)": "hybrid",
    "Item-KNN (ML)": "item_knn",
    "Popularity (baseline)": "popularity",
    "Two-tower (deep learning)": "two_tower",  # trained with python -m shopsmart.recommender.two_tower
}

# ---------------------------------------------------------------- sidebar controls for this page
with st.sidebar:
    mode = st.radio("Assistant mode", ["Search", "Chat"], key="assistant_mode", horizontal=True)
    model_label = st.radio("Recommendation model", list(MODELS), key="assistant_model")

customer = st.session_state.get("customer_id", 0)
segment = customer_segment(customer)
data = load_core_data()
products, search_queries, faq = data["products"], data["search_queries"], data["faq"]
details = products[["Product ID", "Product Name", "Category", "Price", "Avg Rating", "In Stock", "Delivery Days"]]

page_header("Your personal shopping space", "Shopping assistant",
            "Describe what you need in your own words, or let us suggest what to buy next.")


# ---------------------------------------------------------------- search helpers
def smart_search(query: str, limit: int = 6) -> tuple[pd.DataFrame, str, list[str], str | None]:
    """Ollama expands the request into product terms (when running); keyword search does the matching."""
    max_price = extract_max_price(query)
    core = strip_price(query) or query
    terms, error, mode_label = [], None, "Keyword search"
    try:
        with st.spinner(f"Understanding your request with {OLLAMA_MODEL}…"):
            terms = expand_search_query(core, OLLAMA_BASE_URL, OLLAMA_MODEL)
        mode_label = "Local AI search · Ollama"
    except LocalLLMError as exc:
        error = str(exc)
    found = search_catalog(products, [core, *terms], limit=60)
    found = details.merge(found[["Product ID"]], on="Product ID").set_index("Product ID").loc[found["Product ID"]]
    found = found.reset_index()
    if max_price is not None:
        found = found[found["Price"] <= max_price]
    found = found.sort_values("In Stock", ascending=False, kind="stable").head(limit)
    return found, mode_label, terms, error


def remember(query: str) -> None:
    st.session_state.setdefault("search_history", {}).setdefault(customer, []).append(query)


def recent_searches() -> list[str]:
    history = st.session_state.get("search_history", {}).get(customer, [])
    if not history:
        return search_queries.loc[search_queries["Query Type"].eq("semantic"), "Query"].head(6).tolist()
    counts, last = Counter(history), {q: i for i, q in enumerate(history)}
    return sorted(counts, key=lambda q: (-counts[q], -last[q]))[:6]


def top_picks(k: int = 3) -> tuple[pd.DataFrame, str]:
    suite = recommender_suite()
    if suite is None:
        return details.iloc[0:0], "Recommendations are not trained yet: run `python -m shopsmart.recommender.train`."
    model = MODELS[model_label]
    note = ""
    if model == "two_tower":
        two_tower = two_tower_model()
        if two_tower is None:
            model, note = "hybrid", ("The two-tower deep learning model is not trained yet, so Hybrid (ML) is shown. "
                                     "Run `python -m shopsmart.recommender.two_tower`.")
        else:
            recs = two_tower.recommend(user_id=customer, k=40, segment=segment)
    if model != "two_tower":
        recs = suite.recommend(user_id=customer, model=model, k=40, segment=segment)
    recs = recs[["Product ID"]].merge(details, on="Product ID", how="left")
    return recs[recs["In Stock"].fillna(False)].head(k), note


# ---------------------------------------------------------------- SEARCH mode
if mode == "Search":
    with st.form("search_form", clear_on_submit=False, border=False):
        c = st.columns([6, 1])
        query = c[0].text_input("Describe what you need", key="search_text", label_visibility="collapsed",
                                placeholder="e.g. something to help me sleep under £60")
        with c[1]:
            submitted = st.form_submit_button("Submit", type="primary")
    example_query = "Something to help me sleep under £60"
    example_clicked = st.button(f"Try: {example_query}", key="example_search")
    picked = st.pills("Recent searches", recent_searches(), key=f"pills_{customer}")

    run_query = example_query if example_clicked else (query.strip() if submitted and query.strip() else None)
    if not example_clicked and picked and st.session_state.get("last_pill") != (customer, picked):
        run_query = picked
    st.session_state["last_pill"] = (customer, picked)
    if run_query:
        remember(run_query)
        st.session_state["last_search"] = (customer, run_query, *smart_search(run_query))

    last = st.session_state.get("last_search")
    if last and last[0] == customer:
        _, q, results, mode_label, terms, error = last
        st.subheader("Results")
        caption = f"“{q}” · {mode_label}"
        if terms:
            caption += " · also searched for: " + ", ".join(terms)
        st.caption(caption)
        if error:
            st.info(f"Local AI search is unavailable ({error}). Showing keyword results.")
        if results.empty:
            st.warning("No matching products yet. Try a product type, such as *weighted blanket* or *backpack*.")
        else:
            product_cards(results, columns=3)

# ---------------------------------------------------------------- CHAT mode
else:
    history = st.session_state.setdefault("chat_history", {}).setdefault(customer, [])
    if not history:
        history.append({"role": "assistant", "text": "Hi! " + HELP_TEXT, "ids": []})

    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["text"])
            if msg.get("ids"):
                shown = details.set_index("Product ID").loc[msg["ids"]].reset_index()
                st.dataframe(
                    shown.assign(Price=shown["Price"].map(lambda p: f"£{p:.2f}"),
                                 Rating=shown["Avg Rating"].map(lambda r: "–" if pd.isna(r) else f"★ {r:.1f}"),
                                 Stock=shown["In Stock"].map({True: "In stock", False: "Out of stock"}))
                    [["Product Name", "Category", "Price", "Rating", "Stock"]],
                    hide_index=True, width="stretch",
                )

    starters = ["What should I buy next?", "Something to help me sleep under £60", "How long is delivery?"]
    clicked = st.pills("Try asking", starters, key=f"starters_{customer}_{len(history)}")
    prompt = st.chat_input("Ask about products, gifts, delivery or returns…") or clicked

    if prompt:
        history.append({"role": "user", "text": prompt, "ids": []})
        intent = detect_intent(prompt)
        ids: list = []
        if intent == "greeting":
            reply = "Hello! " + HELP_TEXT
        elif intent == "thanks":
            reply = "You're welcome! Anything else I can find for you?"
        elif intent == "order":
            reply = ORDER_TEXT
        elif intent == "faq":
            row = faq_answer(prompt, faq)
            reply = (f"{row['Answer']}\n\n*Source: shop policy – {row['Topic']}*" if row is not None
                     else "I couldn't find that in our shop policies. Our customer service team can help.")
        elif intent == "recommend":
            picks, note = top_picks(3)
            ids = picks["Product ID"].tolist()
            reply = (f"Here are your top {len(ids)} picks right now, ranked by **{model_label}**." if ids
                     else "I don't have personal picks for you yet.") + (f" {note}" if note else "")
        else:
            results, mode_label, terms, error = smart_search(prompt, limit=5)
            ids = results["Product ID"].tolist()
            reply = (f"I found {len(ids)} products for “{prompt}” ({mode_label})." if ids else
                     "I couldn't find a close match. Could you describe it differently, or name a product type?")
        history.append({"role": "assistant", "text": reply, "ids": ids})
        st.rerun()

# ---------------------------------------------------------------- What should I buy next?
st.write("")
with st.container(border=True):
    head = st.columns([3, 2])
    head[0].subheader("What should I buy next?")
    head[1].caption(f"Ranked by **{model_label}** for Customer {customer}")
    picks, note = top_picks(3)
    if note:
        st.info(note)
    if not picks.empty:
        product_cards(picks, label="#{rank} pick for you", columns=3)
