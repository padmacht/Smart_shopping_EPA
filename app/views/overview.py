"""Overview page: the project at a glance (all numbers computed from the data and model reports)."""
import streamlit as st

from common import business_kpis, page_header, profile, report_json

page_header("Project overview", "ShopSmart AI: Customer Intelligence & Shopping Assistant",
            "Understanding customers, predicting what each one wants and helping them shop in their own words.")

k = business_kpis()
cols = st.columns(4)
cols[0].metric("Customers", f"{k['customers']:,}")
cols[1].metric("Products", f"{k['products']:,}")
cols[2].metric("Interactions", f"{k['interactions']:,}")
cols[3].metric("Revenue (2 years)", f"£{k['revenue'] / 1000:,.0f}k")

with st.container(border=True):
    st.subheader("The business problem")
    p = st.columns(3)
    for col, value, text in [
        (p[0], k["no_purchase_sessions"], "of shopping sessions end without a purchase"),
        (p[1], k["cart_abandonment"], "of items added to the basket are never bought"),
        (p[2], k["top10_revenue_share"], "of revenue comes from just 10% of products"),
    ]:
        col.markdown(f'<div class="ss-problem">{value:.1%}</div>', unsafe_allow_html=True)
        col.write(text)

st.subheader("Three modules")
rec = report_json("recommender_metrics.json")
seg = report_json("segmentation_metrics.json")
prof = profile()
m = st.columns(3)
with m[0].container(border=True):
    st.caption("SCENARIO 1 · RECOMMENDATIONS")
    st.markdown("**A personal top 10 for every customer**")
    if rec:
        h, p0 = rec["models"]["hybrid"]["HitRate@10"], rec["models"]["popularity"]["HitRate@10"]
        st.write(f"The hybrid model finds a product the customer later buys for **{h:.1%}** of customers, "
                 f"against {p0:.1%} with best-sellers.")
    st.page_link("views/shopping_assistant.py", label="Open the shopping assistant", icon=":material/arrow_forward:")
with m[1].container(border=True):
    st.caption("SCENARIO 2 · SEGMENTATION")
    st.markdown(f"**{seg['k'] if seg else 8} customer segments for marketing**")
    if seg:
        st.write(f"K-Means on behaviour features, plus **{seg['n_non_buyers']:,} non-buyers** "
                 "who need a first-order offer.")
    st.page_link("views/segments.py", label="Explore segments", icon=":material/arrow_forward:")
with m[2].container(border=True):
    st.caption("SCENARIO 5 · AI ASSISTANT")
    st.markdown("**Search and chat in your own words**")
    st.write("A local LLM (Ollama) turns requests into product searches. Chat mode answers delivery, "
             "returns and 'what should I buy?' questions.")
    st.page_link("views/shopping_assistant.py", label="Try a search", icon=":material/arrow_forward:")

st.subheader("How it works")
f = st.columns(4)
for col, title, text in [
    (f[0], "1. Data", "8 validated datasets, loaded by name"),
    (f[1], "2. Models", "ML baselines, then deep learning where it wins"),
    (f[2], "3. App", "This Streamlit app for customers and marketing"),
    (f[3], "4. MLOps", "Tests, tracking, monitoring, retraining"),
]:
    with col.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(text)
