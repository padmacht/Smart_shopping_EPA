"""Customer segments page (marketing): all segments, or one segment in detail."""
import altair as alt
import pandas as pd
import streamlit as st

from common import GREEN, load_core_data, page_header, profile, report_json, segment_campaign, segment_name, segments

with st.sidebar:
    view = st.radio("View", ["All segments", "One segment in detail"], key="segments_view")
    seg_model = st.radio("Segmentation model", ["K-Means (ML)", "Autoencoder + K-Means (deep learning)"],
                         key="segments_model")

page_header("Marketing", "Customer segments", "Customers grouped by what, how often and how much they buy.")

prof, metrics = profile(), report_json("segmentation_metrics.json")
if prof.empty or metrics is None:
    st.error("Segmentation outputs are missing. Run the task **Train segmentation (K-Means)** first.")
    st.code("python -m shopsmart.segmentation.train")
    st.stop()
if seg_model.startswith("Autoencoder"):
    st.info("The autoencoder (deep learning) segmentation is not trained yet, so K-Means results are shown.")

cols = st.columns(4)
cols[0].metric("Customers segmented", f"{metrics['n_clustered']:,}")
cols[1].metric("Buyer segments", metrics["k"])
cols[2].metric("Never bought", f"{metrics['n_non_buyers']:,}")
cols[3].metric("Match with personas (ARI)", f"{metrics['ari_vs_personas']:.3f}",
               help="Agreement with the hidden customer types (0 = random, 1 = perfect). "
                    "Only possible because the data is synthetic.")

table = prof.reset_index().assign(Segment=lambda d: d["segment"].map(segment_name))

if view == "All segments":
    st.subheader("Segments by size")
    chart = (
        alt.Chart(table)
        .mark_bar(cornerRadiusEnd=4, color=GREEN)
        .encode(
            x=alt.X("customers:Q", title="Customers"),
            y=alt.Y("Segment:N", sort="-x", title=None, axis=alt.Axis(labelLimit=260)),
            tooltip=["Segment", alt.Tooltip("customers:Q", title="Customers"),
                     alt.Tooltip("spend:Q", title="Median spend (£)"), alt.Tooltip("top_category:N", title="Top category")],
        )
        .properties(height=34 * len(table))
    )
    st.altair_chart(chart, use_container_width=True)
    shown = table.sort_values("customers", ascending=False).assign(
        spend=lambda d: d["spend"].map("£{:,.0f}".format), avg_price=lambda d: d["avg_price"].map("£{:,.2f}".format),
        q4_share=lambda d: d["q4_share"].map("{:.0%}".format), premium_share=lambda d: d["premium_share"].map("{:.0%}".format),
        purchases=lambda d: d["purchases"].map("{:.0f}".format))
    st.dataframe(
        shown[["Segment", "customers", "purchases", "spend", "avg_price", "top_category", "q4_share", "premium_share"]]
        .rename(columns={"customers": "Customers", "purchases": "Purchases", "spend": "Median spend",
                         "avg_price": "Item price", "top_category": "Top category", "q4_share": "Nov–Dec",
                         "premium_share": "Premium brands"}),
        hide_index=True, use_container_width=True,
    )
else:
    seg_id = st.selectbox("Segment", prof.index.tolist(), format_func=lambda s: f"{segment_name(s)} (Seg {s})")
    p = prof.loc[seg_id]
    left, right = st.columns([2, 1])
    with left.container(border=True):
        st.subheader(segment_name(seg_id))
        m = st.columns(4)
        m[0].metric("Customers", f"{int(p['customers']):,}")
        m[1].metric("Median spend", f"£{p['spend']:.0f}")
        m[2].metric("Top category", str(p["top_category"]))
        m[3].metric("Nov–Dec activity", f"{p['q4_share']:.0%}")
        st.success(f"**Suggested campaign:** {segment_campaign(seg_id)}")
    with right.container(border=True):
        st.markdown("**Behaviour**")
        st.write(f"Median recency: **{p['recency_days']:.0f} days**")
        st.write(f"Median purchases: **{p['purchases']:.0f}**")
        st.write(f"Premium-brand share: **{p['premium_share']:.0%}**")
        st.write(f"Mobile share: **{p['mobile_share']:.0%}**")

    users = load_core_data()["users"]
    members = segments().merge(users, on="User ID", how="left")
    members = members[members["segment"] == seg_id]
    cols_show = [c for c in ("User ID", "Age", "Gender", "Location", "Preferred Device", "Newsletter") if c in members]
    st.subheader(f"Customers in this segment ({len(members):,})")
    st.dataframe(members[cols_show].sort_values("User ID"), hide_index=True,
                 use_container_width=True, height=280)
    st.download_button("Export customer list (CSV)", members[cols_show].to_csv(index=False).encode("utf-8"),
                       file_name=f"shopsmart_segment_{seg_id}_customers.csv", mime="text/csv", type="primary")
