from __future__ import annotations

import io
import json
from typing import Dict, List
import os

import pandas as pd
import requests
import streamlit as st


BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
try:
    # st.secrets access can raise if no secrets file; keep it optional
    BACKEND_URL = st.secrets["BACKEND_URL"]  # type: ignore[index]
except Exception:
    pass


st.set_page_config(page_title="Precision Marketing Prototype", layout="wide")
st.title("Precision Marketing Prototype")


def call_ingest(df: pd.DataFrame) -> Dict:
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    files = {"csv_file": ("customers.csv", io.BytesIO(csv_bytes), "text/csv")}
    resp = requests.post(f"{BACKEND_URL}/ingest", files=files, timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_train(features: List[str], n_clusters: int) -> Dict:
    payload = {"features": features, "n_clusters": n_clusters}
    resp = requests.post(
        f"{BACKEND_URL}/train/segment", json=payload, timeout=120
    )
    resp.raise_for_status()
    return resp.json()


def call_summary() -> Dict:
    resp = requests.get(f"{BACKEND_URL}/segments/summary", timeout=60)
    resp.raise_for_status()
    return resp.json()


def call_predict(customer: Dict[str, float]) -> Dict:
    payload = {"customer": customer}
    resp = requests.post(f"{BACKEND_URL}/predict/segment", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


with st.sidebar:
    st.header("Settings")
    backend_url_input = st.text_input(
        "Backend URL", value=BACKEND_URL, help="FastAPI base URL"
    )
    if backend_url_input:
        BACKEND_URL = backend_url_input


tab1, tab2, tab3, tab4, tab5 = st.tabs(["Ingest", "Train & Visualize", "Predict One", "Propensity", "Experiments"])


with tab1:
    st.subheader("Upload Customers CSV")
    st.caption("Columns required: recency, frequency, monetary; optional: customer_id, avg_order")
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head())
        if st.button("Ingest CSV", type="primary"):
            try:
                result = call_ingest(df)
                st.success(f"Ingested {result['rows']} rows")
            except Exception as e:
                st.error(f"Ingest failed: {e}")


with tab2:
    st.subheader("Train Segments and Visualize")
    c1, c2, c3 = st.columns([2, 1, 1])
    with c1:
        features = st.multiselect(
            "Features",
            ["recency", "frequency", "monetary", "avg_order"],
            default=["recency", "frequency", "monetary"],
        )
    with c2:
        k = st.number_input("Clusters (k)", min_value=2, max_value=10, value=4, step=1)
    with c3:
        if st.button("Train segments", type="primary"):
            try:
                res = call_train(features, int(k))
                st.session_state["trained_meta"] = res
                st.success("Training complete")
            except Exception as e:
                st.error(f"Training failed: {e}")

    if st.button("Refresh summary"):
        pass

    try:
        summary = call_summary()
        st.markdown("**Segment counts**")
        counts_df = pd.DataFrame([{"segment": c["segment"], "count": c["count"]} for c in summary.get("counts", [])])
        if not counts_df.empty:
            st.dataframe(counts_df)
        else:
            st.info("No customers yet.")

        if summary.get("trained") and summary.get("pca"):
            plot_df = pd.DataFrame(summary["pca"])  # x, y, segment
            import altair as alt

            chart = (
                alt.Chart(plot_df)
                .mark_circle(size=60)
                .encode(x="x", y="y", color="segment:N", tooltip=["x", "y", "segment"])
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.caption("Train a model to see PCA visualization.")
    except Exception as e:
        st.error(f"Could not fetch summary: {e}")


with tab3:
    st.subheader("Predict Single Customer Segment")
    c1, c2, c3 = st.columns(3)
    with c1:
        recency = st.number_input("recency (days)", min_value=0.0, value=10.0)
    with c2:
        frequency = st.number_input("frequency (#orders)", min_value=1.0, value=5.0)
    with c3:
        monetary = st.number_input("monetary (total spent)", min_value=0.0, value=250.0)

    if st.button("Predict segment", type="primary"):
        try:
            res = call_predict({"recency": recency, "frequency": frequency, "monetary": monetary})
            st.success(f"Predicted segment: {res['segment']}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def call_propensity_train(features: List[str], strategy: str, threshold: float | None) -> Dict:
    payload = {"features": features, "label_strategy": strategy}
    if strategy == "custom_threshold" and threshold is not None:
        payload["custom_threshold"] = threshold
    resp = requests.post(f"{BACKEND_URL}/train/propensity", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def call_propensity_predict(customer: Dict[str, float]) -> Dict:
    payload = {"customer": customer}
    resp = requests.post(f"{BACKEND_URL}/predict/propensity", json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


with tab4:
    st.subheader("Propensity Model")
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        p_features = st.multiselect(
            "Features",
            ["recency", "frequency", "monetary", "avg_order"],
            default=["recency", "frequency", "monetary", "avg_order"],
        )
    with c2:
        strategy = st.selectbox("Label strategy", ["median_monetary", "median_frequency", "custom_threshold"], index=0)
    with c3:
        thr = st.number_input("Threshold (when custom)", min_value=0.0, value=500.0)

    if st.button("Train propensity", type="primary"):
        try:
            res = call_propensity_train(p_features, strategy, thr if strategy == "custom_threshold" else None)
            st.success("Propensity model trained")
            st.json(res)
        except Exception as e:
            st.error(f"Training failed: {e}")

    st.markdown("---")
    st.caption("Quick predict")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        pr = st.number_input("recency", min_value=0.0, value=10.0, key="pr")
    with c2:
        pf = st.number_input("frequency", min_value=1.0, value=5.0, key="pf")
    with c3:
        pm = st.number_input("monetary", min_value=0.0, value=250.0, key="pm")
    with c4:
        pa = st.number_input("avg_order", min_value=0.0, value=50.0, key="pa")
    if st.button("Predict propensity"):
        try:
            res = call_propensity_predict({"recency": pr, "frequency": pf, "monetary": pm, "avg_order": pa})
            st.success(f"Probability: {res['probability']:.3f} | Class: {res['klass']}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


def api_create_campaign(name: str) -> Dict:
    payload = {"name": name}
    r = requests.post(f"{BACKEND_URL}/campaigns", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_assign(campaign_id: int, customer_id: int, variants: List[str]) -> Dict:
    payload = {"campaign_id": campaign_id, "customer_id": customer_id, "variants": variants}
    r = requests.post(f"{BACKEND_URL}/assign", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_outcome(campaign_id: int, customer_id: int, converted: bool, revenue: float) -> Dict:
    payload = {"campaign_id": campaign_id, "customer_id": customer_id, "converted": converted, "revenue": revenue}
    r = requests.post(f"{BACKEND_URL}/outcomes", json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def api_report(campaign_id: int) -> Dict:
    r = requests.get(f"{BACKEND_URL}/campaigns/{campaign_id}/report", timeout=30)
    r.raise_for_status()
    return r.json()


with tab5:
    st.subheader("Experiments (A/B)")
    name = st.text_input("Campaign name", value="Welcome Email Test")
    if st.button("Create campaign"):
        try:
            st.session_state["campaign"] = api_create_campaign(name)
            st.success(f"Campaign created: ID {st.session_state['campaign']['id']}")
        except Exception as e:
            st.error(f"Create failed: {e}")

    if "campaign" in st.session_state:
        c = st.session_state["campaign"]
        st.markdown(f"**Active Campaign ID**: {c['id']} - {c['name']}")
        c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
        with c1:
            cid = st.number_input("customer_id", min_value=1, value=1)
        with c2:
            variants = st.text_input("variants (comma)", value="A,B").split(",")
        with c3:
            if st.button("Assign variant"):
                try:
                    res = api_assign(int(c["id"]), int(cid), [v.strip() for v in variants if v.strip()])
                    st.success(f"Assigned: {res['variant']}")
                except Exception as e:
                    st.error(f"Assign failed: {e}")
        with c4:
            st.write("")

        st.markdown("---")
        st.caption("Record outcome")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            conv = st.checkbox("converted", value=True)
        with c2:
            rev = st.number_input("revenue", min_value=0.0, value=100.0)
        with c3:
            if st.button("Submit outcome"):
                try:
                    api_outcome(int(c["id"]), int(cid), bool(conv), float(rev))
                    st.success("Outcome recorded")
                except Exception as e:
                    st.error(f"Outcome failed: {e}")

        if st.button("View report"):
            try:
                rep = api_report(int(c["id"]))
                st.json(rep)
                if rep.get("variants"):
                    import altair as alt
                    rep_df = pd.DataFrame(rep["variants"])  # variant, assigned, conversions, conversion_rate, avg_revenue
                    chart = alt.Chart(rep_df).mark_bar().encode(x="variant", y="conversion_rate").properties(title="Conversion Rate by Variant")
                    st.altair_chart(chart, use_container_width=True)
            except Exception as e:
                st.error(f"Report failed: {e}")


