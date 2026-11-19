"""Model performance page: every model against its baseline, on data it has not seen."""
import altair as alt
import pandas as pd
import streamlit as st

from common import GREEN, MUTED_GREEN, page_header, report_json

with st.sidebar:
    module = st.radio("Module", ["Recommendations", "Segmentation", "Search"], key="perf_module")

page_header("Evaluation", "Model performance",
            "Every model is compared with a simple baseline on data it has not seen. "
            "Deep learning replaces ML only if it wins.")


def bar_chart(df: pd.DataFrame, value: str, label: str, highlight: str, fmt: str = ".1%") -> alt.Chart:
    base = alt.Chart(df).encode(
        y=alt.Y("Model:N", sort=None, title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X(f"{value}:Q", title=label, axis=alt.Axis(format=fmt, tickCount=5)),
    )
    bars = base.mark_bar(cornerRadiusEnd=4).encode(
        color=alt.condition(alt.datum.Model == highlight, alt.value(GREEN), alt.value(MUTED_GREEN)),
        tooltip=["Model", alt.Tooltip(f"{value}:Q", format=fmt)],
    )
    text = base.mark_text(align="left", dx=6, fontWeight="bold").encode(text=alt.Text(f"{value}:Q", format=fmt))
    return (bars + text).properties(height=60 * len(df))


if module == "Recommendations":
    rec = report_json("recommender_metrics.json")
    if rec is None:
        st.error("No recommender metrics yet. Run `python -m shopsmart.recommender.train`.")
        st.stop()
    names = {"popularity": "Popularity (baseline)", "item_knn": "Item-KNN", "hybrid": "Hybrid",
             "two_tower": "Two-tower (deep learning)"}
    df = pd.DataFrame([{"Model": names.get(m, m), "HitRate@10": s["HitRate@10"], "NDCG@10": s["NDCG@10"],
                        "Users evaluated": s["users_evaluated"]} for m, s in rec["models"].items()])
    best = df.loc[df["HitRate@10"].idxmax()]
    base = df.loc[df["Model"].str.startswith("Popularity"), "HitRate@10"].iloc[0]
    st.subheader(f"{best['Model']} finds a product customers later buy for {best['HitRate@10']:.1%} of them, "
                 f"up from {base:.1%} with best-sellers")
    with st.container(border=True):
        st.altair_chart(bar_chart(df, "HitRate@10", "HitRate@10 (share of customers who bought one of their top 10)",
                                  best["Model"]), use_container_width=True)
    table = df.assign(**{"HitRate@10": df["HitRate@10"].map("{:.1%}".format), "NDCG@10": df["NDCG@10"].map("{:.3f}".format),
                         "Users evaluated": df["Users evaluated"].map("{:,}".format)})
    if "two_tower" not in rec["models"]:
        table = pd.concat([table, pd.DataFrame([{"Model": "Two-tower (deep learning)", "HitRate@10": "next",
                                                 "NDCG@10": "next", "Users evaluated": "–"}])], ignore_index=True)
    st.dataframe(table, hide_index=True, use_container_width=True)
    if "two_tower" in rec["models"] and "hybrid" in rec["models"]:
        dl, ml = rec["models"]["two_tower"], rec["models"]["hybrid"]
        if dl["HitRate@10"] > ml["HitRate@10"]:
            st.success(f"Deep learning wins: the two-tower model reaches {dl['HitRate@10']:.1%} HitRate@10 against "
                       f"{ml['HitRate@10']:.1%} for Hybrid, so it can replace Hybrid as the default.")
        else:
            st.info(f"Hybrid (ML) stays the default: the two-tower deep learning model reaches "
                    f"{dl['HitRate@10']:.1%} HitRate@10 against {ml['HitRate@10']:.1%} for Hybrid. "
                    f"It beats best-sellers but not Hybrid on this data.")
    st.caption(f"Time-based test: trained on events before {rec['cutoff'][:10]}, tested on the "
               f"{rec['test_events']:,} events after it.")

elif module == "Segmentation":
    seg = report_json("segmentation_metrics.json")
    if seg is None:
        st.error("No segmentation metrics yet. Run `python -m shopsmart.segmentation.train`.")
        st.stop()
    st.subheader(f"K-Means with {seg['feature_set']} features matches the hidden personas with ARI {seg['ari_vs_personas']:.3f}")
    c = st.columns(3)
    with c[0].container(border=True):
        st.metric("ARI vs hidden personas", f"{seg['ari_vs_personas']:.3f}")
        st.caption("0 = random, 1 = perfect")
    with c[1].container(border=True):
        st.metric("Silhouette", f"{seg['silhouette']:.3f}")
        st.caption("How well separated the segments are (higher is better)")
    with c[2].container(border=True):
        st.metric("Autoencoder + K-Means (DL)", "next")
        st.caption(f"Must beat ARI {seg['ari_vs_personas']:.3f}")
    st.caption(f"{seg['n_clustered']:,} buyers in {seg['k']} segments · {seg['n_non_buyers']:,} non-buyers kept "
               "as their own group. ARI is only possible because the data is synthetic.")

else:
    search = report_json("search_metrics.json")
    if search is None:
        st.warning("Search has not been evaluated yet. Run the task **Evaluate search**, or:")
        st.code("python -m shopsmart.search_eval")
        st.stop()
    labels = {"keyword": "Exact product names", "semantic": "Natural-language requests",
              "filtered_brand": "Brand + product", "filtered_colour": "Colour + product",
              "filtered_price": "Product + price", "gift": "Gift requests"}
    scores = pd.DataFrame([{"Model": labels.get(t, t), "Hit@10": v["Hit@10"], "Queries": v["queries"]}
                           for t, v in search["by_query_type"].items()]).sort_values("Hit@10")
    key = search["by_query_type"].get("keyword", {}).get("Hit@10")
    sem = search["by_query_type"].get("semantic", {}).get("Hit@10")
    if key is not None and sem is not None:
        st.subheader(f"Keyword search finds exact product names {key:.0%} of the time, "
                     f"but natural requests only {sem:.0%}")
    with st.container(border=True):
        st.altair_chart(bar_chart(scores, "Hit@10", "Hit@10 (a right product in the top 10), keyword search without Ollama",
                                  "Natural-language requests", fmt=".0%"), use_container_width=True)
    st.caption(f"{search['queries']} labelled queries from search_queries.csv. Next: Ollama query expansion and "
               "sentence embeddings (deep learning), which must beat these scores, especially on natural requests.")
