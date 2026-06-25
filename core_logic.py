# core_logic.py
# Backend logic for Audit Risk Analytics Engine
# Streamlit-ready version generated from the original Colab/Python workflow.

import io
import re
from difflib import SequenceMatcher
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


# ============================================================
# STANDARD SCHEMA
# ============================================================

STANDARD_SCHEMA = {
    "invoice_number": ["invoice", "invoice no", "invoice number", "inv no", "billing document"],
    "invoice_date": ["invoice date", "inv date", "billing date", "date"],
    "customer_id": ["customer", "customer id", "customer code", "cust id", "cust code"],
    "customer_name": ["customer name", "cust name", "party name"],
    "material_code": ["material", "material code", "material no", "sku"],
    "material_description": ["material description", "material desc", "description"],
    "quantity": ["quantity", "qty", "billing quantity"],
    "unit_price": ["per unit price", "unit price", "price per unit"],
    "gross_amount": ["gross amount", "gross value", "total amount", "501001 sales"],
    "discount_amount": ["discount amount", "discount", "disc amount"],
    "tax_percent": ["tax percent", "tax %", "gst %", "tax rate"],
    "currency": ["currency", "curr", "document currency"],
    "country": ["country", "market", "region"],
    "plant": ["plant", "plant code", "location"],
    "foc_indicator": ["foc", "free of cost", "foc indicator"],
    "invoice_type": ["invoice type", "doc type", "billing type"],
    "invoice_type_details": ["invoice type details", "inv type details", "billing details", "description"],
    "customer_po_date": ["customer po date", "po date", "cust po date"]
}


# ============================================================
# FILE LOADING
# ============================================================

def load_data(uploaded_file):
    """
    Streamlit-compatible file loading.
    uploaded_file may be a Streamlit UploadedFile, a file-like object, or a path.
    """
    file_name = getattr(uploaded_file, "name", str(uploaded_file))

    if file_name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif file_name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError("Please upload a CSV, XLSX, or XLS file.")


# ============================================================
# SCHEMA INFERENCE FUNCTIONS
# Kept with the same function names and matching logic.
# ============================================================

def clean_col_name(x):
    x = str(x).strip().lower()
    x = re.sub(r"[\-_./\\]+", " ", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def similarity(a, b):
    return SequenceMatcher(None, clean_col_name(a), clean_col_name(b)).ratio()


def token_score(col, alias):
    col_tokens = set(clean_col_name(col).split())
    alias_tokens = set(clean_col_name(alias).split())
    if not col_tokens or not alias_tokens:
        return 0
    return len(col_tokens.intersection(alias_tokens)) / len(alias_tokens)


def profile_column(series):
    s = series.dropna()
    if len(s) == 0:
        return {"numeric_ratio": 0, "date_ratio": 0, "unique_ratio": 0}
    as_str = s.astype(str).str.strip()
    num = pd.to_numeric(as_str.str.replace(",", "", regex=False), errors="coerce")

    # format="mixed" is not supported by older pandas versions, so fallback safely.
    try:
        dt = pd.to_datetime(as_str, errors="coerce", dayfirst=True, format="mixed")
    except TypeError:
        dt = pd.to_datetime(as_str, errors="coerce", dayfirst=True)

    return {
        "numeric_ratio": num.notna().mean(),
        "date_ratio": dt.notna().mean(),
        "unique_ratio": s.nunique() / len(s)
    }


def profile_boost(field, profile):
    boost = 0
    if field in ["quantity", "unit_price", "gross_amount", "discount_amount", "tax_percent"]:
        boost += profile["numeric_ratio"] * 0.20
    if field in ["invoice_date", "customer_po_date"]:
        boost += profile["date_ratio"] * 0.30
    return boost


def infer_schema(df, threshold=0.50):
    results = {}
    for field, aliases in STANDARD_SCHEMA.items():
        best_col, best_score, best_alias = None, 0, None
        for col in df.columns:
            prof = profile_column(df[col])
            scores = [(max(similarity(col, alias), token_score(col, alias)), alias) for alias in aliases]
            name_score, alias = max(scores, key=lambda x: x[0])
            final_score = min(name_score + profile_boost(field, prof), 1)
            if final_score > best_score:
                best_col, best_score, best_alias = col, final_score, alias
        results[field] = {
            "mapped_column": best_col if best_score >= threshold else None,
            "confidence": round(best_score, 3),
            "matched_alias": best_alias
        }
    return results


# ============================================================
# CLEANING / STANDARDISATION FUNCTIONS
# ============================================================

def clean_numeric(series):
    s = series.astype(str).str.strip()
    s = s.replace(["", "nan", "None", "NULL", "null", "-"], np.nan)
    s = s.astype(str)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace(r"\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(r"[^\d.\-]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def clean_date(series):
    return pd.to_datetime(series, errors="coerce", dayfirst=True)


def standardise_dataframe(df, mapping):
    out = pd.DataFrame(index=df.index)

    numeric_fields = ["quantity", "unit_price", "gross_amount", "discount_amount", "tax_percent"]
    date_fields = ["invoice_date", "customer_po_date"]

    for standard_field in STANDARD_SCHEMA.keys():
        raw_col = mapping.get(standard_field)
        if raw_col is not None and raw_col in df.columns:
            if standard_field in numeric_fields:
                out[standard_field] = clean_numeric(df[raw_col])
            elif standard_field in date_fields:
                out[standard_field] = clean_date(df[raw_col])
            else:
                out[standard_field] = df[raw_col]
        else:
            out[standard_field] = np.nan

    if "invoice_date" in out.columns:
        out["month"] = out["invoice_date"].dt.to_period("M").astype(str)
        out.loc[out["invoice_date"].isna(), "month"] = np.nan
    else:
        out["month"] = np.nan

    return out


def mapping_result_to_manual_mapping(mapping_result):
    return {field: info["mapped_column"] for field, info in mapping_result.items()}


# ============================================================
# AUDIT CHECKS
# ============================================================

def run_duplicate_invoice_detection(std_df):
    dup_subset = ["invoice_number", "customer_id", "material_code"]

    if any(col not in std_df.columns for col in dup_subset):
        return pd.DataFrame()

    if any(std_df[col].isna().all() for col in dup_subset):
        return pd.DataFrame()

    duplicates = std_df[std_df.duplicated(subset=dup_subset, keep=False)].copy()

    if duplicates.empty:
        return pd.DataFrame()

    duplicate_invoice_result = (
        duplicates.groupby(dup_subset, dropna=False)
        .agg(
            duplicate_row_count=("invoice_number", "size"),
            total_quantity=("quantity", "sum"),
            total_gross_amount=("gross_amount", "sum")
        )
        .reset_index()
    )

    duplicate_invoice_result["audit_check"] = "Duplicate Invoice Detection"
    duplicate_invoice_result["audit_reason"] = "Same invoice number, customer and material appears multiple times"
    return duplicate_invoice_result


def identify_foc(series):
    x = series.astype(str).str.lower()
    return (
        x.str.contains("foc", na=False) |
        x.str.contains("free", na=False) |
        x.str.contains("sample", na=False) |
        x.isin(["y", "yes", "true", "1"])
    )


def run_foc_validation(std_df):
    temp = std_df.copy()

    is_foc_indicator = identify_foc(temp["foc_indicator"]) if "foc_indicator" in temp.columns else False
    is_foc_type = identify_foc(temp["invoice_type"]) if "invoice_type" in temp.columns else False

    combined_foc_mask = is_foc_indicator | is_foc_type

    foc_validation_result = temp[
        combined_foc_mask &
        (
            (temp["unit_price"].fillna(0) != 0) |
            (temp["gross_amount"].fillna(0) != 0)
        )
    ].copy()

    if len(foc_validation_result) > 0:
        foc_validation_result["audit_check"] = "FOC Validation"
        foc_validation_result["audit_reason"] = "Item identified as FOC/Free has non-zero price or amount"

    return foc_validation_result


def run_anti_foc_validation(std_df):
    temp = std_df.copy()

    is_foc_indicator = identify_foc(temp["foc_indicator"]) if "foc_indicator" in temp.columns else False
    is_foc_type = identify_foc(temp["invoice_type"]) if "invoice_type" in temp.columns else False
    combined_foc_mask = is_foc_indicator | is_foc_type

    anti_foc_result = temp[
        (~combined_foc_mask) &
        (temp["quantity"].fillna(0) > 0) &
        (
            (temp["unit_price"].fillna(0) == 0) |
            (temp["gross_amount"].fillna(0) == 0)
        )
    ].copy()

    if len(anti_foc_result) > 0:
        anti_foc_result["audit_check"] = "Anti-FOC Validation"
        anti_foc_result["audit_reason"] = "Standard item (non-FOC) has non-zero quantity but zero price/amount"

    return anti_foc_result


def run_price_consistency_nonzero(std_df):
    required = ["customer_id", "material_code", "month", "unit_price"]

    if any(c not in std_df.columns for c in required):
        return pd.DataFrame()

    if any(std_df[c].isna().all() for c in required):
        return pd.DataFrame()

    available_cols = required + [c for c in ["customer_name", "material_description", "invoice_type_details"] if c in std_df.columns]
    temp = std_df[available_cols].dropna(subset=required).copy()

    temp = temp[temp["unit_price"].fillna(0) != 0]

    if temp.empty:
        return pd.DataFrame()

    grouped = (
        temp.groupby(["customer_id", "material_code", "month"], dropna=False)
        .agg(
            min_price=("unit_price", "min"),
            max_price=("unit_price", "max"),
            distinct_price_count=("unit_price", "nunique"),
            transaction_count=("unit_price", "size")
        )
        .reset_index()
    )

    grouped["price_difference"] = grouped["max_price"] - grouped["min_price"]

    price_consistency_nonzero_result = grouped[
        (grouped["distinct_price_count"] > 1) &
        (grouped["price_difference"].abs() >= 0.01)
    ].copy()

    if len(price_consistency_nonzero_result) > 0:
        price_consistency_nonzero_result["audit_check"] = "Price Consistency - Non Zero"
        price_consistency_nonzero_result["audit_reason"] = "Different non-zero prices found for same customer, material and month"

    return price_consistency_nonzero_result


def run_price_consistency_zero_mix(std_df):
    required_cols = ["customer_id", "material_code", "month", "unit_price", "quantity"]

    if any(c not in std_df.columns for c in required_cols):
        return pd.DataFrame()

    if any(std_df[c].isna().all() for c in required_cols):
        return pd.DataFrame()

    available_cols = required_cols + [c for c in ["customer_name", "material_description", "invoice_type_details"] if c in std_df.columns]
    temp_data = std_df[available_cols].dropna(subset=required_cols).copy()

    if temp_data.empty:
        return pd.DataFrame()

    temp_data["is_zero_price"] = temp_data["unit_price"].fillna(0) == 0
    temp_data["zero_price_quantity"] = np.where(temp_data["is_zero_price"], temp_data["quantity"].fillna(0), 0)
    temp_data["total_quantity_component"] = temp_data["quantity"].fillna(0)

    grouped = (
        temp_data.groupby(["customer_id", "material_code", "month"], dropna=False)
        .agg(
            zero_price_count=("is_zero_price", "sum"),
            total_count=("is_zero_price", "size"),
            zero_price_quantity=("zero_price_quantity", "sum"),
            total_quantity=("total_quantity_component", "sum"),
            min_price=("unit_price", "min"),
            max_price=("unit_price", "max")
        )
        .reset_index()
    )

    grouped["free_ratio"] = np.where(
        grouped["total_quantity"].abs() > 0,
        grouped["zero_price_quantity"] / grouped["total_quantity"],
        0
    )

    price_consistency_zero_mix_result = grouped[
        (grouped["zero_price_count"] > 0) &
        (grouped["zero_price_count"] < grouped["total_count"])
    ].copy()

    if len(price_consistency_zero_mix_result) > 0:
        price_consistency_zero_mix_result["audit_check"] = "Price Consistency - Zero vs Non-Zero Mix"
        price_consistency_zero_mix_result["audit_reason"] = "Same customer/material/month contains both zero and non-zero priced quantities"

    return price_consistency_zero_mix_result


def run_quantity_anomaly(std_df):
    if any(c not in std_df.columns for c in ["material_code", "quantity", "customer_id"]):
        return pd.DataFrame()

    if std_df["material_code"].isna().all() or std_df["quantity"].isna().all() or std_df["customer_id"].isna().all():
        return pd.DataFrame()

    group_cols = ["customer_id", "material_code"]
    temp = std_df.dropna(subset=group_cols + ["quantity"]).copy()

    if temp.empty:
        return pd.DataFrame()

    stats = (
        temp.groupby(group_cols)["quantity"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "quantity_mean", "std": "quantity_std", "count": "quantity_count"})
    )

    temp = temp.merge(stats, on=group_cols, how="left")
    temp["quantity_zscore"] = np.where(
        temp["quantity_std"].fillna(0) != 0,
        (temp["quantity"] - temp["quantity_mean"]) / temp["quantity_std"],
        0
    )

    quantity_anomaly_result = temp[
        (temp["quantity_count"] >= 3) &
        (temp["quantity_zscore"].abs() >= 3)
    ].copy()

    if len(quantity_anomaly_result) > 0:
        quantity_anomaly_result["audit_check"] = "Quantity Anomaly"
        quantity_anomaly_result["audit_reason"] = "Customer-material quantity is a z-score outlier"

    return quantity_anomaly_result


def run_invoice_date_validation(std_df):
    invoice_date_results = []

    if "invoice_date" in std_df.columns and not std_df["invoice_date"].isna().all():
        future = std_df[std_df["invoice_date"] > pd.Timestamp.today()].copy()
        if not future.empty:
            future["audit_check"] = "Future Dated Invoice"
            future["audit_reason"] = "Invoice date is in the future"
            invoice_date_results.append(future)

    if "invoice_date" in std_df.columns and "customer_po_date" in std_df.columns:
        date_mismatch = std_df[std_df["invoice_date"] < std_df["customer_po_date"]].copy()
        if not date_mismatch.empty:
            date_mismatch["audit_check"] = "Date Sequence Anomaly"
            date_mismatch["audit_reason"] = "Invoice date is earlier than Customer PO date"
            invoice_date_results.append(date_mismatch)

    if "invoice_number" in std_df.columns and "invoice_date" in std_df.columns:
        if not std_df["invoice_number"].isna().all() and not std_df["invoice_date"].isna().all():
            temp = std_df.dropna(subset=["invoice_number", "invoice_date"]).copy()
            grouped_dates = temp.groupby("invoice_number")["invoice_date"].nunique().reset_index()
            bad_invoices = grouped_dates[grouped_dates["invoice_date"] > 1]["invoice_number"]
            mismatch_invoices = temp[temp["invoice_number"].isin(bad_invoices)].copy()
            if not mismatch_invoices.empty:
                mismatch_invoices["audit_check"] = "Multiple Invoice Dates"
                mismatch_invoices["audit_reason"] = "Same invoice number has different invoice dates"
                invoice_date_results.append(mismatch_invoices)

    if invoice_date_results:
        all_date_anomalies = pd.concat(invoice_date_results, ignore_index=False)
        invoice_date_validation_result = all_date_anomalies.copy()
    else:
        invoice_date_validation_result = pd.DataFrame()

    return invoice_date_validation_result


def run_all_audit_checks(std_df):
    duplicate_invoice_result = run_duplicate_invoice_detection(std_df)
    foc_validation_result = run_foc_validation(std_df)
    anti_foc_result = run_anti_foc_validation(std_df)
    price_consistency_nonzero_result = run_price_consistency_nonzero(std_df)
    price_consistency_zero_mix_result = run_price_consistency_zero_mix(std_df)
    quantity_anomaly_result = run_quantity_anomaly(std_df)
    invoice_date_validation_result = run_invoice_date_validation(std_df)

    return {
        "duplicate_invoice_result": duplicate_invoice_result,
        "foc_validation_result": foc_validation_result,
        "anti_foc_result": anti_foc_result,
        "price_consistency_nonzero_result": price_consistency_nonzero_result,
        "price_consistency_zero_mix_result": price_consistency_zero_mix_result,
        "quantity_anomaly_result": quantity_anomaly_result,
        "invoice_date_validation_result": invoice_date_validation_result,
    }


# ============================================================
# RISK SCORING + ML
# ============================================================

audit_weights = {
    "is_duplicate": 0,
    "is_foc_anomaly": 15,
    "is_anti_foc_anomaly": 15,
    "is_price_inconsistent_nonzero": 10,
    "is_price_inconsistent_zero_mix": 20,
    "is_quantity_anomaly": 15,
    "is_date_anomaly": 25
}


def compute_rule_based_risk(std_df, audit_results):
    std_df = std_df.copy()

    for flag_name in audit_weights.keys():
        std_df[flag_name] = 0

    duplicate_invoice_result = audit_results.get("duplicate_invoice_result", pd.DataFrame())
    foc_validation_result = audit_results.get("foc_validation_result", pd.DataFrame())
    anti_foc_result = audit_results.get("anti_foc_result", pd.DataFrame())
    price_consistency_nonzero_result = audit_results.get("price_consistency_nonzero_result", pd.DataFrame())
    price_consistency_zero_mix_result = audit_results.get("price_consistency_zero_mix_result", pd.DataFrame())
    quantity_anomaly_result = audit_results.get("quantity_anomaly_result", pd.DataFrame())
    invoice_date_validation_result = audit_results.get("invoice_date_validation_result", pd.DataFrame())

    if not duplicate_invoice_result.empty:
        idx_keys = ["invoice_number", "customer_id", "material_code"]
        dup_indices = duplicate_invoice_result[idx_keys]
        std_df.loc[
            std_df.set_index(idx_keys).index.isin(dup_indices.set_index(idx_keys).index),
            "is_duplicate"
        ] = 1

    if not price_consistency_nonzero_result.empty:
        idx_keys = ["customer_id", "material_code", "month"]
        nonzero_indices = price_consistency_nonzero_result[idx_keys]
        std_df.loc[
            std_df.set_index(idx_keys).index.isin(nonzero_indices.set_index(idx_keys).index),
            "is_price_inconsistent_nonzero"
        ] = 1

    if not price_consistency_zero_mix_result.empty:
        idx_keys = ["customer_id", "material_code", "month"]
        mix_indices = price_consistency_zero_mix_result[idx_keys]
        std_df.loc[
            std_df.set_index(idx_keys).index.isin(mix_indices.set_index(idx_keys).index),
            "is_price_inconsistent_zero_mix"
        ] = 1

    if not foc_validation_result.empty:
        std_df.loc[std_df.index.isin(foc_validation_result.index), "is_foc_anomaly"] = 1

    if not anti_foc_result.empty:
        std_df.loc[std_df.index.isin(anti_foc_result.index), "is_anti_foc_anomaly"] = 1

    if not quantity_anomaly_result.empty:
        std_df.loc[std_df.index.isin(quantity_anomaly_result.index), "is_quantity_anomaly"] = 1

    if not invoice_date_validation_result.empty:
        std_df.loc[std_df.index.isin(invoice_date_validation_result.index), "is_date_anomaly"] = 1

    std_df["rule_based_risk_score"] = sum(std_df[k] * audit_weights[k] for k in audit_weights.keys())
    return std_df


def run_isolation_forest(std_df, contamination=0.05, random_state=42):
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn is required for Isolation Forest. Install using: pip install scikit-learn")

    std_df = std_df.copy()
    features = ["quantity", "unit_price", "gross_amount"]

    for f in features:
        if f not in std_df.columns:
            std_df[f] = 0

    X = std_df[features].fillna(0).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    iso_forest = IsolationForest(contamination=contamination, random_state=random_state)
    iso_forest.fit(X_scaled)

    std_df["ml_anomaly_label"] = iso_forest.predict(X_scaled)
    std_df["ml_raw_score"] = iso_forest.decision_function(X_scaled)

    mms = MinMaxScaler(feature_range=(0, 100))
    std_df["ml_risk_score"] = mms.fit_transform(-std_df[["ml_raw_score"]])

    return std_df, iso_forest, scaler


def compute_total_risk_score(std_df, rule_weight=0.60, ml_weight=0.40):
    std_df = std_df.copy()
    std_df["total_risk_score"] = (
        std_df["rule_based_risk_score"] * rule_weight +
        std_df["ml_risk_score"] * ml_weight
    )
    return std_df


def create_high_risk_report(std_df, threshold=40):
    high_risk_report = std_df[std_df["total_risk_score"] >= threshold].copy()
    high_risk_report = high_risk_report.sort_values(by="total_risk_score", ascending=False)
    return high_risk_report


# ============================================================
# HUMAN-IN-THE-LOOP FEEDBACK LOGIC
# Streamlit replaces IPyWidgets in app.py, but this backend keeps
# the same scoring adjustment idea.
# ============================================================

def apply_manual_labels(std_df, labels_log):
    std_df = std_df.copy()
    std_df["manual_label"] = "Not Reviewed"

    if labels_log:
        for idx, label in labels_log.items():
            if idx in std_df.index:
                std_df.at[idx, "manual_label"] = label

    return std_df


def analyze_feedback_patterns(std_df):
    labeled_df = std_df[std_df["manual_label"] != "Not Reviewed"].copy()

    if len(labeled_df) == 0:
        return {
            "reviewed_count": 0,
            "valid_count": 0,
            "precision": None,
            "false_positive_customers": pd.Series(dtype=int),
            "false_positive_materials": pd.Series(dtype=int),
        }

    valid_count = len(labeled_df[labeled_df["manual_label"] == "Valid Anomaly"])
    total_reviewed = len(labeled_df)
    precision = valid_count / total_reviewed

    fp_data = labeled_df[labeled_df["manual_label"] == "False Positive"]

    return {
        "reviewed_count": total_reviewed,
        "valid_count": valid_count,
        "precision": precision,
        "false_positive_customers": fp_data.get("customer_name", pd.Series(dtype=object)).value_counts().head(5),
        "false_positive_materials": fp_data.get("material_description", pd.Series(dtype=object)).value_counts().head(5),
    }


def apply_semi_supervised_adjustment(std_df):
    std_df = std_df.copy()

    if "manual_label" not in std_df.columns:
        std_df["manual_label"] = "Not Reviewed"

    std_df["total_risk_score_pre_adjustment"] = std_df["total_risk_score"]

    fp_data = std_df[std_df["manual_label"] == "False Positive"]

    fp_customers = fp_data["customer_name"].dropna().unique().tolist() if "customer_name" in fp_data.columns else []
    fp_materials = fp_data["material_description"].dropna().unique().tolist() if "material_description" in fp_data.columns else []

    def calculate_adjustment(row):
        if "customer_name" in row.index and row["customer_name"] in fp_customers:
            return 0.80
        if "material_description" in row.index and row["material_description"] in fp_materials:
            return 0.80
        return 1.0

    std_df["feedback_adjustment"] = std_df.apply(calculate_adjustment, axis=1)
    std_df["total_risk_score"] = std_df["total_risk_score_pre_adjustment"] * std_df["feedback_adjustment"]

    return std_df


# ============================================================
# FULL PIPELINE
# ============================================================

def run_full_audit_pipeline(df, manual_mapping=None, threshold=40, contamination=0.05, rule_weight=0.60, ml_weight=0.40):
    mapping_result = infer_schema(df)

    if manual_mapping is None:
        manual_mapping = mapping_result_to_manual_mapping(mapping_result)

    std_df = standardise_dataframe(df, manual_mapping)
    audit_results = run_all_audit_checks(std_df)
    std_df = compute_rule_based_risk(std_df, audit_results)
    std_df, iso_model, scaler = run_isolation_forest(std_df, contamination=contamination)
    std_df = compute_total_risk_score(std_df, rule_weight=rule_weight, ml_weight=ml_weight)
    high_risk_report = create_high_risk_report(std_df, threshold=threshold)

    return {
        "mapping_result": mapping_result,
        "manual_mapping": manual_mapping,
        "std_df": std_df,
        "audit_results": audit_results,
        "high_risk_report": high_risk_report,
        "iso_model": iso_model,
        "scaler": scaler,
    }


# ============================================================
# EXPORT HELPERS
# ============================================================

def dataframe_to_csv_bytes(df):
    return df.to_csv(index=False).encode("utf-8")


def export_results_to_excel_bytes(std_df, audit_results, high_risk_report):
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        std_df.to_excel(writer, sheet_name="All_Transactions", index=False)
        high_risk_report.to_excel(writer, sheet_name="High_Risk_Transactions", index=False)

        for name, result_df in audit_results.items():
            sheet_name = name[:31]
            if result_df is not None and not result_df.empty:
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)
            else:
                pd.DataFrame({"message": ["No records found"]}).to_excel(writer, sheet_name=sheet_name, index=False)

    output.seek(0)
    return output.getvalue()
