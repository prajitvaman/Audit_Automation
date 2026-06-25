# app.py
# Streamlit UI for Audit Risk Analytics Engine

import pandas as pd
import streamlit as st

from core_logic import (
    STANDARD_SCHEMA,
    load_data,
    infer_schema,
    mapping_result_to_manual_mapping,
    run_full_audit_pipeline,
    create_high_risk_report,
    dataframe_to_csv_bytes,
    export_results_to_excel_bytes,
    apply_manual_labels,
    analyze_feedback_patterns,
    apply_semi_supervised_adjustment,
)

# ✅ CACHED WRAPPERS (ADD THIS HERE)
@st.cache_data
def cached_load_data(file):
    return load_data(file)

@st.cache_data
def cached_infer_schema(df):
    return infer_schema(df)

@st.cache_data
def cached_pipeline(df, manual_mapping, threshold, contamination, rule_weight, ml_weight):
    return run_full_audit_pipeline(
        df=df,
        manual_mapping=manual_mapping,
        threshold=threshold,
        contamination=contamination,
        rule_weight=rule_weight,
        ml_weight=ml_weight
    )


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Audit Risk Analytics Engine",
    page_icon="📊",
    layout="wide"
)

st.title("Audit Risk Analytics Engine")
st.caption("Schema mapping, audit checks, Isolation Forest anomaly scoring and HITL feedback")


# ============================================================
# SIDEBAR SETTINGS
# ============================================================

with st.sidebar:
    st.header("Settings")

    threshold = st.slider(
        "High-risk threshold",
        min_value=0,
        max_value=100,
        value=40,
        step=1
    )

    rule_weight_percent = st.slider(
        "Rule-based weight (%)",
        min_value=0,
        max_value=100,
        value=60,
        step=5
    )

    ml_weight_percent = 100 - rule_weight_percent
    st.write(f"ML weight: **{ml_weight_percent}%**")

    contamination = st.slider(
        "Isolation Forest contamination",
        min_value=0.01,
        max_value=0.20,
        value=0.05,
        step=0.01
    )

    st.divider()
    st.caption("Recommended starting point: 60% rule-based, 40% ML, threshold 40.")


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

if "labels_log" not in st.session_state:
    st.session_state.labels_log = {}

if "current_label_pos" not in st.session_state:
    st.session_state.current_label_pos = 0


# ============================================================
# FILE UPLOAD
# ============================================================



uploaded_file = st.file_uploader("Upload CSV / Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is None:
    st.info("Upload a sales register file to start.")
    st.stop()

try:
    df = cached_load_data(uploaded_file)
except Exception as e:
    st.error(f"Could not load file: {e}")
    st.stop()

st.success(f"Uploaded: {uploaded_file.name}")

col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{df.shape[0]:,}")
col2.metric("Columns", f"{df.shape[1]:,}")
col3.metric("File type", uploaded_file.name.split(".")[-1].upper())

with st.expander("Preview uploaded data", expanded=True):
    st.dataframe(df.head(20), use_container_width=True)

# ============================================================
# SCHEMA INFERENCE + MANUAL OVERRIDE
# ============================================================

st.header("1. Schema Mapping")

mapping_result = cached_infer_schema(df)
default_mapping = mapping_result_to_manual_mapping(mapping_result)

mapping_rows = []
for field, info in mapping_result.items():
    mapping_rows.append({
        "standard_field": field,
        "suggested_column": info["mapped_column"],
        "confidence": info["confidence"],
        "matched_alias": info["matched_alias"]
    })

mapping_df = pd.DataFrame(mapping_rows)
st.dataframe(mapping_df, use_container_width=True)


# ✅ FORM STARTS HERE
with st.form("run_form"):

    st.subheader("Manual Mapping Override")
    st.caption("Keep the suggested mapping if correct, or override any field below.")

    manual_mapping = {}
    raw_options = [None] + list(df.columns)

    with st.expander("Edit mapping", expanded=False):
        for field in STANDARD_SCHEMA.keys():
            suggested = default_mapping.get(field)
            default_index = raw_options.index(suggested) if suggested in raw_options else 0

            manual_mapping[field] = st.selectbox(
                label=field,
                options=raw_options,
                index=default_index,
                key=f"map_{field}"
            )

    # ✅ BUTTON MOVES INSIDE FORM
    run_clicked = st.form_submit_button("Run Audit Engine", type="primary")

if run_clicked:
    try:
        with st.spinner("Running audit checks and anomaly scoring..."):
            st.session_state.pipeline_result = cached_pipeline(
                df=df,
                manual_mapping=manual_mapping,
                threshold=threshold,
                contamination=contamination,
                rule_weight=rule_weight_percent / 100,
                ml_weight=ml_weight_percent / 100,
            )
            st.session_state.labels_log = {}
            st.session_state.current_label_pos = 0
        st.success("Audit pipeline completed.")
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()


if st.session_state.pipeline_result is None:
    st.info("Click **Run Audit Engine** after checking the mapping.")
    st.stop()

result = st.session_state.pipeline_result
std_df = result["std_df"]
audit_results = result["audit_results"]
high_risk_report = result["high_risk_report"]


# ============================================================
# RESULTS SUMMARY
# ============================================================

st.header("3. Results Summary")

summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
summary_col1.metric("Total transactions", f"{len(std_df):,}")
summary_col2.metric("High-risk transactions", f"{len(high_risk_report):,}")
summary_col3.metric("Max risk score", f"{std_df['total_risk_score'].max():.2f}")
summary_col4.metric("ML anomalies", f"{(std_df['ml_anomaly_label'] == -1).sum():,}")


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "High Risk",
    "All Transactions",
    "Audit Checks",
    "Charts",
    "Manual Feedback"
])


# ============================================================
# TAB 1: HIGH RISK
# ============================================================

with tab1:
    st.subheader("Prioritised High-Risk Transactions")
    st.dataframe(high_risk_report, use_container_width=True)

    st.download_button(
        label="Download high_risk_audit_report.csv",
        data=dataframe_to_csv_bytes(high_risk_report),
        file_name="high_risk_audit_report.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 2: ALL TRANSACTIONS
# ============================================================

with tab2:
    st.subheader("All Transactions with Risk Scores")
    st.dataframe(std_df, use_container_width=True)

    st.download_button(
        label="Download all_transactions_with_scores.csv",
        data=dataframe_to_csv_bytes(std_df),
        file_name="all_transactions_with_scores.csv",
        mime="text/csv"
    )


# ============================================================
# TAB 3: AUDIT CHECKS
# ============================================================

with tab3:
    st.subheader("Rule-Based Audit Check Outputs")

    for check_name, check_df in audit_results.items():
        with st.expander(f"{check_name} ({0 if check_df is None else len(check_df):,})", expanded=False):
            if check_df is None or check_df.empty:
                st.write("No records found.")
            else:
                st.dataframe(check_df, use_container_width=True)

    excel_bytes = export_results_to_excel_bytes(std_df, audit_results, high_risk_report)

    st.download_button(
        label="Download full Excel audit pack",
        data=excel_bytes,
        file_name="audit_risk_report.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )


# ============================================================
# TAB 4: CHARTS
# ============================================================

with tab4:
    st.subheader("Risk Score Distribution")
    st.bar_chart(std_df["total_risk_score"].value_counts().sort_index())

    st.subheader("Top High-Risk Transactions")
    chart_df = high_risk_report.head(50).copy()
    if not chart_df.empty:
        st.scatter_chart(
            chart_df,
            x="total_risk_score",
            y="gross_amount"
        )
    else:
        st.info("No high-risk transactions above the selected threshold.")


# ============================================================
# TAB 5: MANUAL FEEDBACK / HITL
# ============================================================

with tab5:
    st.subheader("Human-in-the-Loop Manual Feedback")
    st.caption("This replaces the original Colab IPyWidgets flow with Streamlit buttons and session state.")

    review_df = std_df[std_df["total_risk_score"] >= threshold].sort_values(
        by="total_risk_score",
        ascending=False
    ).copy()

    review_indices = review_df.index.tolist()

    if len(review_indices) == 0:
        st.info("No transactions available for manual review at the current threshold.")
    else:
        current_pos = min(st.session_state.current_label_pos, len(review_indices) - 1)
        current_idx = review_indices[current_pos]
        current_row = std_df.loc[[current_idx]].T.reset_index()
        current_row.columns = ["field", "value"]

        st.write(f"Reviewing transaction {current_pos + 1} of {len(review_indices)}")
        st.dataframe(current_row, use_container_width=True)

        b1, b2, b3 = st.columns(3)

        if b1.button("Valid Anomaly", type="primary"):
            st.session_state.labels_log[current_idx] = "Valid Anomaly"
            st.session_state.current_label_pos = min(current_pos + 1, len(review_indices) - 1)
            st.rerun()

        if b2.button("False Positive"):
            st.session_state.labels_log[current_idx] = "False Positive"
            st.session_state.current_label_pos = min(current_pos + 1, len(review_indices) - 1)
            st.rerun()

        if b3.button("Skip / Next"):
            st.session_state.current_label_pos = min(current_pos + 1, len(review_indices) - 1)
            st.rerun()

        st.divider()
        st.subheader("Feedback Summary")

        labelled_df = apply_manual_labels(std_df, st.session_state.labels_log)
        feedback_summary = analyze_feedback_patterns(labelled_df)

        c1, c2, c3 = st.columns(3)
        c1.metric("Reviewed", feedback_summary["reviewed_count"])
        c2.metric("Valid anomalies", feedback_summary["valid_count"])
        precision_value = feedback_summary["precision"]
        c3.metric("Precision", "N/A" if precision_value is None else f"{precision_value * 100:.2f}%")

        if feedback_summary["reviewed_count"] > 0:
            adjusted_df = apply_semi_supervised_adjustment(labelled_df)
            adjusted_high_risk = create_high_risk_report(adjusted_df, threshold=threshold)

            st.write("Post-feedback adjusted high-risk count:", len(adjusted_high_risk))

            st.download_button(
                label="Download feedback_adjusted_results.csv",
                data=dataframe_to_csv_bytes(adjusted_df),
                file_name="feedback_adjusted_results.csv",
                mime="text/csv"
            )

            with st.expander("False positive customer/material patterns"):
                st.write("False positive customers")
                st.dataframe(feedback_summary["false_positive_customers"])
                st.write("False positive materials")
                st.dataframe(feedback_summary["false_positive_materials"])
