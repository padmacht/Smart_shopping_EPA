"""
ShopSmart AI - Streamlit app (single entry point with sidebar navigation).

Run from the project folder (C:\\Smart_shopping_EPA):
    streamlit run app/app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from common import APP_CSS, OLLAMA_MODEL, customer_segment, load_core_data, segment_name

st.set_page_config(page_title="ShopSmart AI", page_icon=":material/shopping_bag:", layout="wide")
st.markdown(APP_CSS, unsafe_allow_html=True)

pages = [
    st.Page("views/overview.py", title="Overview", icon=":material/home:", default=True),
    st.Page("views/shopping_assistant.py", title="Shopping assistant", icon=":material/chat:"),
    st.Page("views/segments.py", title="Customer segments", icon=":material/groups:"),
    st.Page("views/model_performance.py", title="Model performance", icon=":material/bar_chart:"),
]
st.logo(str(Path(__file__).parent / "assets" / "logo.svg"), size="large")
navigation = st.navigation(pages)

# ---------------------------------------------------------------- shared sidebar (every page)
with st.sidebar:
    users = load_core_data()["users"]
    customer_ids = sorted(int(u) for u in users["User ID"].dropna().unique())
    st.selectbox("Demo: signed in as", customer_ids, key="customer_id",
                 format_func=lambda u: f"Customer {u}")
    st.caption(f"Segment: **{segment_name(customer_segment(st.session_state.customer_id))}**")

navigation.run()

# Footer goes last so each page's own radio buttons sit above it.
with st.sidebar:
    st.divider()
    st.markdown(
        f'<div class="ss-footer">Local AI search: Ollama ({OLLAMA_MODEL}) when running<br>'
        f'Data: {len(users):,} customers · {len(load_core_data()["products"]):,} products</div>',
        unsafe_allow_html=True,
    )
