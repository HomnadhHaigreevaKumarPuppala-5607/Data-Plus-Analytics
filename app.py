from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

ALLOWED_EXTENSIONS = {"csv", "xlsx", "xls", "json", "tsv"}
uploaded_files = []


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def detect_dataset_theme(df, filename):
    """Intelligently guess the domain/theme of the dataset."""
    cols_lower = [c.lower() for c in df.columns]
    fname_lower = filename.lower()

    themes = {
        "sales": ["sale", "revenue", "order", "purchase", "price", "amount", "invoice", "transaction", "discount"],
        "hr": ["employee", "salary", "department", "hire", "tenure", "leave", "payroll", "staff", "position", "job"],
        "healthcare": ["patient", "diagnosis", "hospital", "doctor", "treatment", "medicine", "health", "disease", "drug"],
        "finance": ["stock", "investment", "portfolio", "profit", "loss", "balance", "asset", "liability", "interest", "budget"],
        "marketing": ["campaign", "click", "impression", "ctr", "conversion", "lead", "funnel", "ad", "channel", "roi"],
        "ecommerce": ["product", "sku", "inventory", "cart", "shipping", "customer", "return", "review", "category", "rating"],
        "logistics": ["shipment", "delivery", "warehouse", "route", "freight", "carrier", "tracking", "dispatch"],
        "education": ["student", "grade", "score", "course", "enrollment", "teacher", "attendance", "exam"],
        "real_estate": ["property", "rent", "mortgage", "listing", "bedroom", "sqft", "location", "agent"],
    }

    scores = {theme: 0 for theme in themes}
    for theme, keywords in themes.items():
        for kw in keywords:
            if any(kw in c for c in cols_lower):
                scores[theme] += 2
            if kw in fname_lower:
                scores[theme] += 3

    best_theme = max(scores, key=scores.get)
    if scores[best_theme] == 0:
        return "general", "Data Analysis"

    theme_labels = {
        "sales": "Sales & Revenue",
        "hr": "HR & Workforce",
        "healthcare": "Healthcare & Medical",
        "finance": "Finance & Investment",
        "marketing": "Marketing & Campaigns",
        "ecommerce": "E-Commerce & Products",
        "logistics": "Logistics & Delivery",
        "education": "Education & Academic",
        "real_estate": "Real Estate",
        "general": "Data Analysis",
    }
    return best_theme, theme_labels[best_theme]


def smart_column_label(col_name):
    """Return a clean human-readable label for a column."""
    return str(col_name).replace("_", " ").replace("-", " ").title()


def generate_insights(df, numeric_cols, cat_cols, theme, filename):
    """Generate contextual AI insights based on actual data."""
    insights = []

    if numeric_cols:
        # Robust sum to prevent NaN crashes
        col_sums = {c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum() for c in numeric_cols}
        top_col = max(col_sums, key=col_sums.get)
        s = pd.to_numeric(df[top_col], errors="coerce").dropna()
        if len(s) > 0:
            mean_val = s.mean()
            std_val = s.std()

            if std_val / (mean_val + 1e-9) > 0.8:
                insights.append({
                    "type": "warning",
                    "title": f"High variability in \"{smart_column_label(top_col)}\"",
                    "body": f"Standard deviation is {round(std_val, 2):,} against a mean of {round(mean_val, 2):,}. "
                            f"This suggests strong outliers. Consider segmenting high/low performers for targeted action."
                })

        # Detect top category
        if cat_cols:
            cat_col = cat_cols[0]
            freq = df[cat_col].value_counts()
            if not freq.empty:
                top_cat = freq.index[0]
                top_pct = round(freq.iloc[0] / len(df) * 100, 1)
                insights.append({
                    "type": "info",
                    "title": f"\"{top_cat}\" dominates {smart_column_label(cat_col)}",
                    "body": f"The value \"{top_cat}\" appears in {top_pct}% of all records in the "
                            f"\"{smart_column_label(cat_col)}\" column. "
                            f"This segment may deserve dedicated analysis or targeted strategy."
                })

        # Check for potential date trends (using improved keywords)
        date_keywords = ["date", "time", "year", "month", "day", "created", "updated", "timestamp"]
        date_cols = [c for c in df.columns if any(kw in c.lower() for kw in date_keywords)]
        if date_cols:
            insights.append({
                "type": "growth",
                "title": f"Time dimension detected: \"{smart_column_label(date_cols[0])}\"",
                "body": f"Your dataset contains a time column. "
                        f"Use the Revenue chart to visualise trends over time. "
                        f"The date range spans your full dataset of {len(df):,} records."
            })

        # Fix: Safely handle correlation with NaN columns
        valid_numeric = []
        for col in numeric_cols[:6]:
            if df[col].dropna().shape[0] > 1:
                valid_numeric.append(col)

        if len(valid_numeric) >= 2:
            corr = df[valid_numeric].corr()
            pairs = []
            for i, c1 in enumerate(valid_numeric):
                for c2 in valid_numeric[i+1:]:
                    val = abs(corr.loc[c1, c2])
                    if not np.isnan(val) and val > 0.6:
                        pairs.append((c1, c2, round(val, 2)))
            if pairs:
                c1, c2, v = pairs[0]
                insights.append({
                    "type": "opportunity",
                    "title": f"Strong correlation: {smart_column_label(c1)} ↔ {smart_column_label(c2)}",
                    "body": f"These two numeric columns have a correlation of {v} — a strong linear relationship. "
                            f"This could indicate a causal link worth investigating or leveraging in your strategy."
                })

    if not insights:
        insights.append({
            "type": "info",
            "title": f"Dataset loaded: {filename}",
            "body": f"Your file contains {len(df):,} rows and {len(df.columns)} columns. "
                    f"All charts and KPI cards have been updated to reflect your actual data."
        })

    return insights[:4]


def build_chart_data(df, numeric_cols, cat_cols, date_cols):
    """Build chart datasets from actual uploaded data."""
    charts = {}

    # --- Trend / Revenue line chart ---
    if date_cols and numeric_cols:
        date_col = date_cols[0]
        val_col = sorted(numeric_cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum(), reverse=True)[0]
        try:
            df["_parsed_date"] = pd.to_datetime(df[date_col], errors="coerce")
            monthly = df.dropna(subset=["_parsed_date"]).copy()
            monthly["_ym"] = monthly["_parsed_date"].dt.to_period("M").astype(str)
            grouped = monthly.groupby("_ym")[val_col].sum().sort_index()
            if len(grouped) > 1:
                charts["trend"] = {
                    "labels": list(grouped.index),
                    "values": [round(float(v), 2) for v in grouped.values],
                    "col_name": smart_column_label(val_col),
                }
            df.drop(columns=["_parsed_date", "_ym"], errors="ignore", inplace=True)
        except Exception:
            pass

    # --- Donut chart from best categorical column ---
    donut_col = None
    if cat_cols:
        best_cat = None
        for c in cat_cols:
            uniq = df[c].nunique()
            if 2 <= uniq <= 12:
                best_cat = c
                break
        if not best_cat:
            best_cat = cat_cols[0]
        donut_col = best_cat
        freq = df[best_cat].value_counts().head(8)
        charts["donut"] = {
            "labels": list(freq.index.astype(str)),
            "values": [int(v) for v in freq.values],
            "col_name": smart_column_label(best_cat),
        }

    # --- Bar chart from second categorical column or another ---
    bar_candidates = [c for c in cat_cols if c != donut_col]
    if not bar_candidates and cat_cols:
        bar_candidates = cat_cols
    if bar_candidates:
        bar_col = bar_candidates[0]
        freq2 = df[bar_col].value_counts().head(10)
        top_val_col = sorted(numeric_cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum(), reverse=True)[0] if numeric_cols else None
        if top_val_col:
            agg = df.groupby(bar_col)[top_val_col].sum().sort_values(ascending=False).head(10)
            charts["bar"] = {
                "labels": list(agg.index.astype(str)),
                "values": [round(float(v), 2) for v in agg.values],
                "col_name": smart_column_label(bar_col),
                "val_col": smart_column_label(top_val_col),
            }
        else:
            charts["bar"] = {
                "labels": list(freq2.index.astype(str)),
                "values": [int(v) for v in freq2.values],
                "col_name": smart_column_label(bar_col),
                "val_col": "Count",
            }

    # --- Scatter / bubble data ---
    if len(numeric_cols) >= 2:
        c1, c2 = numeric_cols[0], numeric_cols[1]
        scatter_df = df[[c1, c2]].dropna()
        if len(scatter_df) > 0:
            sample = scatter_df.sample(min(80, len(scatter_df)), random_state=42)
            charts["scatter"] = {
                "x_label": smart_column_label(c1),
                "y_label": smart_column_label(c2),
                "points": [{"x": round(float(r[c1]), 2), "y": round(float(r[c2]), 2)} for _, r in sample.iterrows()],
            }

    # --- Histogram of top numeric column ---
    if numeric_cols:
        hcol = sorted(numeric_cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum(), reverse=True)[0]
        vals = pd.to_numeric(df[hcol], errors="coerce").dropna()
        if len(vals) > 0:
            counts, bin_edges = np.histogram(vals, bins=12)
            charts["histogram"] = {
                "labels": [f"{round(bin_edges[i], 1)}–{round(bin_edges[i+1], 1)}" for i in range(len(counts))],
                "values": [int(c) for c in counts],
                "col_name": smart_column_label(hcol),
            }

    return charts


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload_file():
    try:
        if "file" not in request.files:
            return jsonify({"success": False, "message": "No file uploaded"})

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"success": False, "message": "Empty filename"})
        if not allowed_file(file.filename):
            return jsonify({"success": False, "message": "Unsupported file type. Use CSV, XLSX, XLS, JSON or TSV."})

        # Fix: Prevent duplicate upload overwrites
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)

        try:
            ext = filename.rsplit(".", 1)[1].lower()

            if ext == "csv":
                df = pd.read_csv(filepath, low_memory=False)
            elif ext in ["xlsx", "xls"]:
                df = pd.read_excel(filepath)
            elif ext == "json":
                # Fix: Better JSON handling
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    first_key = next(iter(data))
                    df = pd.json_normalize(data[first_key])
                else:
                    df = pd.json_normalize(data)
            elif ext == "tsv":
                df = pd.read_csv(filepath, sep="\t", low_memory=False)
            else:
                return jsonify({"success": False, "message": "Invalid format"})

            # Fix: Empty Dataset Crash
            if df.empty:
                return jsonify({"success": False, "message": "Dataset is empty"})

            # Clean
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.where(pd.notnull(df), None)

            # Column classification
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            # Fix: Improved date detection
            date_keywords = ["date", "time", "year", "month", "day", "created", "updated", "timestamp"]
            date_cols = [c for c in df.columns if any(kw in c.lower() for kw in date_keywords)]
            cat_cols = [c for c in df.columns if c not in numeric_cols and c not in date_cols and df[c].nunique() <= max(50, len(df) * 0.3)]

            # Theme detection
            theme_key, theme_label = detect_dataset_theme(df, filename)

            # KPIs
            kpis = {}
            if numeric_cols:
                # Fix: Robust numeric sum check
                sorted_by_sum = sorted(numeric_cols, key=lambda c: pd.to_numeric(df[c], errors="coerce").fillna(0).sum(), reverse=True)
                primary = sorted_by_sum[0]
                kpis["primary_value"] = round(float(pd.to_numeric(df[primary], errors="coerce").sum()), 2)
                kpis["primary_label"] = smart_column_label(primary)
                kpis["primary_avg"] = round(float(pd.to_numeric(df[primary], errors="coerce").mean()), 2)
                kpis["primary_max"] = round(float(pd.to_numeric(df[primary], errors="coerce").max()), 2)
                kpis["primary_min"] = round(float(pd.to_numeric(df[primary], errors="coerce").min()), 2)
                kpis["primary_median"] = round(float(pd.to_numeric(df[primary], errors="coerce").median()), 2)
                kpis["primary_std"] = round(float(pd.to_numeric(df[primary], errors="coerce").std()), 2)
                if len(sorted_by_sum) > 1:
                    sec = sorted_by_sum[1]
                    kpis["secondary_value"] = round(float(pd.to_numeric(df[sec], errors="coerce").sum()), 2)
                    kpis["secondary_label"] = smart_column_label(sec)
                    kpis["secondary_avg"] = round(float(pd.to_numeric(df[sec], errors="coerce").mean()), 2)

            kpis["total_records"] = int(len(df))
            kpis["total_columns"] = int(len(df.columns))
            kpis["numeric_count"] = len(numeric_cols)
            kpis["category_count"] = len(cat_cols)

            # Column stats
            col_stats = {}
            for col in numeric_cols:
                vals = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(vals) == 0:
                    continue
                col_stats[col] = {
                    "label": smart_column_label(col),
                    "type": "numeric",
                    "sum": round(float(vals.sum()), 2),
                    "mean": round(float(vals.mean()), 2),
                    "median": round(float(vals.median()), 2),
                    "std": round(float(vals.std()), 2),
                    "min": round(float(vals.min()), 2),
                    "max": round(float(vals.max()), 2),
                    "count": int(vals.count()),
                    "nulls": int(df[col].isna().sum()),
                }
            for col in cat_cols:
                freq = df[col].value_counts().head(5)
                col_stats[col] = {
                    "label": smart_column_label(col),
                    "type": "category",
                    "unique": int(df[col].nunique()),
                    "top": str(freq.index[0]) if len(freq) > 0 else "—",
                    "top_pct": round(freq.iloc[0] / len(df) * 100, 1) if len(freq) > 0 else 0,
                    "nulls": int(df[col].isna().sum()),
                    "freq": {str(k): int(v) for k, v in freq.items()},
                }

            # Charts
            chart_data = build_chart_data(df, numeric_cols, cat_cols, date_cols)

            # Insights
            insights = generate_insights(df, numeric_cols, cat_cols, theme_key, filename)

            # Preview
            # Fix: Don't convert everything to strings, keep structure for frontend
            preview = df.head(100).replace({np.nan: None}).to_dict(orient="records")

            # File history
            uploaded_files.append({
                "filename": filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "theme": theme_label,
                "uploaded_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            })
            
            # Fix: Prevent Memory Leak
            if len(uploaded_files) > 100:
                uploaded_files.pop(0)

            return jsonify({
                "success": True,
                "filename": filename,
                "rows": int(df.shape[0]),
                "columns": int(df.shape[1]),
                "theme_key": theme_key,
                "theme_label": theme_label,
                "column_names": list(df.columns),
                "numeric_columns": numeric_cols,
                "category_columns": cat_cols,
                "date_columns": date_cols,
                "kpi": kpis,
                "col_stats": col_stats,
                "chart_data": chart_data,
                "insights": insights,
                "preview": preview,
            })

        finally:
            # Fix: Security improvement, clean up uploaded files
            if os.path.exists(filepath):
                os.remove(filepath)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "message": f"Error reading file: {str(e)}"})


@app.route("/api/files")
def get_files():
    return jsonify({"success": True, "files": uploaded_files[::-1]})


@app.route("/api/status")
def status():
    return jsonify({"success": True, "message": "Backend running", "version": "2.1"})


@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "message": "File too large. Max 32 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
