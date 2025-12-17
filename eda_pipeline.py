import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from flask import Flask, request, jsonify, session
from flask_cors import CORS
import logging
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PowerTransformer,
)
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import scipy.stats as stats

app = Flask(__name__)
CORS(app)
app.secret_key = "your-secret-key-change-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
sns.set(style="whitegrid")

# Global storage
processing_state = {}


# ==================== HELPER FUNCTIONS ====================
def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64


def create_missing_heatmap(df):
    """Create missing values heatmap"""
    if df.isnull().sum().sum() == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="coolwarm", ax=ax)
    ax.set_title("Missing Values Heatmap")
    return fig_to_base64(fig)


def create_pairplot(df, max_cols=4):
    """Create pairplot for numeric columns"""
    try:
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] > 1:
            cols_to_plot = min(max_cols, num_df.shape[1])
            fig = sns.pairplot(num_df.iloc[:, :cols_to_plot]).fig
            fig.suptitle(f"Pairplot (first {cols_to_plot} numeric cols)", y=1.02)
            return fig_to_base64(fig)
    except Exception as e:
        logging.warning(f"Pairplot error: {e}")
    return None


def generate_statistical_insights(df):
    """Generate rule-based statistical insights"""
    insights = []

    # 1. Missing Values Analysis
    total_missing = df.isnull().sum().sum()
    if total_missing == 0:
        insights.append("✅ No missing values detected.")
    else:
        ratio = (total_missing / (df.shape[0] * df.shape[1])) * 100
        if ratio > 30:
            insights.append(
                f"⚠️ Around {ratio:.1f}% cells are missing — consider strong imputation or dropping columns."
            )
        elif ratio > 10:
            insights.append(
                f"🩹 Moderate missingness (~{ratio:.1f}%) — impute numeric with median, categorical with mode."
            )
        else:
            insights.append(
                f"🟢 Low missingness ({ratio:.1f}%) — minor cleaning sufficient."
            )

    # 2. Skewness Analysis
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) > 0:
        skew_vals = df[numeric_cols].skew()
        highly_skewed = skew_vals[abs(skew_vals) > 1]
        if not highly_skewed.empty:
            insights.append(
                f"📈 {len(highly_skewed)} highly skewed features: {list(highly_skewed.index)[:5]} — consider log/Box-Cox transform."
            )
        else:
            insights.append("✅ Numeric features appear well-balanced.")

    # 3. Correlation Analysis
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        high_corr = []
        for a in corr.columns:
            for b in corr.columns:
                if a != b and corr.loc[a, b] > 0.85:
                    pair = tuple(sorted((a, b)))
                    if pair not in high_corr:
                        high_corr.append(pair)
        if high_corr:
            pairs = [f"{a}↔{b}" for a, b in high_corr[:3]]
            insights.append(
                f"🔗 High correlation pairs found ({len(high_corr)}): {pairs} — remove redundant features."
            )
        else:
            insights.append("✅ No strong multicollinearity detected.")

    # 4. Cardinality Check
    cat_cols = df.select_dtypes(exclude=np.number).columns
    for col in cat_cols:
        uniq_ratio = df[col].nunique() / len(df)
        if uniq_ratio > 0.9:
            insights.append(
                f"🔢 '{col}' has very high uniqueness ({uniq_ratio:.2f}) — may act like an ID column."
            )

    # 5. Data Quality Score
    score = 100
    if total_missing > 0:
        score -= 10
    if (
        len(numeric_cols) > 0
        and len(df[numeric_cols].skew()[abs(df[numeric_cols].skew()) > 1]) > 0
    ):
        score -= 5
    # recompute high_corr length safely
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr().abs()
        high_corr_count = (
            sum(
                1
                for a in corr.columns
                for b in corr.columns
                if a != b and corr.loc[a, b] > 0.85
            )
            // 2
        )
        if high_corr_count > 0:
            score -= 5
    insights.append(f"📊 Overall Data Quality Score: {score}/100")

    return insights


def dataset_summary(df):
    """Generate comprehensive dataset summary"""
    summary = {
        "Total Rows": int(df.shape[0]),
        "Total Columns": int(df.shape[1]),
        "Numeric Columns": len(df.select_dtypes(include=np.number).columns),
        "Categorical Columns": len(df.select_dtypes(exclude=np.number).columns),
        "Missing Cells": int(df.isnull().sum().sum()),
        "Duplicate Rows": int(df.duplicated().sum()),
        "Memory Usage (MB)": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
    }
    return summary


# ==================== API ENDPOINTS ====================
@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify(
        {"status": "ok", "message": "Flask server running", "version": "2.0"}
    )


@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify(
        {
            "message": "Advanced EDA Flask API",
            "version": "2.0",
            "features": [
                "AI Suggestions",
                "Statistical Insights",
                "Advanced ML Prep",
                "Visualizations",
            ],
        }
    )


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Upload CSV and perform initial analysis"""
    try:
        logging.info("Received upload request")

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        # Read CSV
        df = pd.read_csv(file)
        logging.info(f"CSV loaded: {df.shape}")

        # Missing Values
        logging.info("==== Step 2: Missing Values per Column ====")
        logging.info(f"\n{df.isnull().sum().to_string()}")

        # Initial cleaning
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w\s]", "", regex=True)
        )
        df = df.drop_duplicates().reset_index(drop=True)

        # Store in state
        session_id = request.form.get("session_id", "default")
        processing_state[session_id] = {
            "df": df,
            "step": "uploaded",
            "original_shape": df.shape,
        }

        # Generate insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        summary = dataset_summary(df)

        response = {
            "status": "success",
            "message": "Dataset loaded and cleaned successfully",
            "shape": list(df.shape),
            "columns": list(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "missing_values": summary["Missing Cells"],
            "duplicates_removed": 0,
            "summary": summary,
            "head": df.head(10).to_html(classes="table table-striped"),
            "missing_heatmap": create_missing_heatmap(df),
            "next_step": "ai_suggestions",
        }

        logging.info("Upload successful")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ai_suggestions", methods=["POST"])
def ai_suggestions():
    """Generate AI-powered suggestions"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        suggestions = []

        # 1. High missing columns (>40%)
        missing_ratio = df.isnull().mean() * 100
        high_missing = missing_ratio[missing_ratio > 40].index.tolist()
        if high_missing:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "high",
                    "message": f"Columns with >40% missing: {high_missing}",
                    "columns": high_missing,
                    "reason": "Too much missing data - unreliable for analysis",
                }
            )

        # 2. High uniqueness (>90%)
        high_unique = [col for col in df.columns if df[col].nunique() > (0.9 * len(df))]
        if high_unique:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "medium",
                    "message": f"High uniqueness columns: {high_unique}",
                    "columns": high_unique,
                    "reason": "Likely ID columns - no predictive value",
                }
            )

        # 3. Low variance (<= 1 unique value)
        low_var = [col for col in numeric_cols if df[col].nunique() <= 1]
        if low_var:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "medium",
                    "message": f"Constant/low variance columns: {low_var}",
                    "columns": low_var,
                    "reason": "No variation - adds no information",
                }
            )

        # 4. Skewed features
        if len(numeric_cols) > 0:
            skew_vals = df[numeric_cols].skew()
            skewed = skew_vals[abs(skew_vals) > 1].index.tolist()
            if skewed:
                suggestions.append(
                    {
                        "type": "transform",
                        "severity": "medium",
                        "message": f"Highly skewed features ({len(skewed)}): {skewed[:5]}",
                        "columns": skewed,
                        "reason": "Skewed distribution - consider log/Box-Cox transform",
                    }
                )

        # 5. Very low variance numeric columns
        if len(numeric_cols) > 0:
            var = df[numeric_cols].var()
            very_low_var = var[var < 0.01].index.tolist()
            if very_low_var:
                suggestions.append(
                    {
                        "type": "drop_columns",
                        "severity": "low",
                        "message": f"Very low variance: {very_low_var}",
                        "columns": very_low_var,
                        "reason": "Minimal variation - little predictive value",
                    }
                )

        # All suggested drop columns
        all_drop_cols = list(set(high_missing + high_unique + low_var))

        # Statistical insights
        stat_insights = generate_statistical_insights(df)

        return jsonify(
            {
                "status": "success",
                "suggestions": suggestions,
                "suggested_drop_columns": all_drop_cols,
                "statistical_insights": stat_insights,
                "needs_user_input": True,
                "next_step": "handle_suggestions",
            }
        )

    except Exception as e:
        logging.error(f"AI suggestions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/handle_suggestions", methods=["POST"])
def handle_suggestions():
    """Handle AI suggestions based on user choice"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        action = data.get("action")  # 'auto', 'manual', 'skip'
        custom_columns = data.get("columns", [])

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        changes_made = []

        if action == "auto":
            # Auto-clean mode - comprehensive cleaning

            # Fill missing values
            for col in numeric_cols:
                if df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].median())
                    changes_made.append(f"Filled {col} missing values with median")

            for col in cat_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(
                        mode_val[0] if not mode_val.empty else "Unknown"
                    )
                    changes_made.append(f"Filled {col} missing values with mode")

            # Transform skewed features
            skew_vals = df[numeric_cols].skew()
            skewed = skew_vals[abs(skew_vals) > 1]
            for col in skewed.index:
                # Use log1p on absolute values to be safer with negatives
                df[col] = np.log1p(df[col].abs().astype(float))
                changes_made.append(f"Applied log transform to {col}")

            # Standard scaling
            if len(numeric_cols) > 0:
                scaler = StandardScaler()
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
                changes_made.append(
                    f"Applied standard scaling to {len(numeric_cols)} numeric columns"
                )

            message = (
                f"Auto-cleaning complete: {len(changes_made)} operations performed"
            )

        elif action == "manual" and custom_columns:
            # Drop custom columns
            valid_cols = [c for c in custom_columns if c in df.columns]
            invalid_cols = [c for c in custom_columns if c not in df.columns]

            if valid_cols:
                df = df.drop(columns=valid_cols)
                changes_made.append(f"Dropped columns: {valid_cols}")
                message = f"Dropped {len(valid_cols)} columns"
            else:
                message = "No valid columns to drop"

            if invalid_cols:
                message += f" (Invalid columns ignored: {invalid_cols})"

        else:
            message = "Skipped AI suggestions - no changes made"

        # Update state
        processing_state[session_id]["df"] = df
        processing_state[session_id]["changes"] = changes_made

        # Generate new insights
        stat_insights = generate_statistical_insights(df)

        return jsonify(
            {
                "status": "success",
                "message": message,
                "changes_made": changes_made,
                "shape": list(df.shape),
                "missing_values": int(df.isnull().sum().sum()),
                "missing_heatmap": create_missing_heatmap(df),
                "statistical_insights": stat_insights,
                "next_step": (
                    "choose_workflow"
                    if df.isnull().sum().sum() == 0
                    else "handle_missing"
                ),
            }
        )

    except Exception as e:
        logging.error(f"Handle suggestions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/missing_values", methods=["POST"])
def handle_missing_values():
    """Handle missing values with specified method"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        method = data.get("method")  # 'mean', 'median', 'mode'

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        if method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
            msg = "Applied mean imputation"
        elif method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
            msg = "Applied median imputation"
        else:  # mode
            for col in numeric_cols:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 0)
            msg = "Applied mode imputation"

        df[cat_cols] = df[cat_cols].fillna("Unknown")

        processing_state[session_id]["df"] = df

        return jsonify(
            {
                "status": "success",
                "message": f'{msg} for numeric columns, filled categorical with "Unknown"',
                "missing_values": int(df.isnull().sum().sum()),
                "head": df.head(10).to_html(classes="table table-striped"),
            }
        )

    except Exception as e:
        logging.error(f"Missing values error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/workflow", methods=["POST"])
def choose_workflow():
    """Choose analysis workflow"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        workflow = data.get("workflow")  # 'eda', 'ml', 'dashboard'

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        processing_state[session_id]["workflow"] = workflow

        if workflow == "eda":
            # EDA workflow
            summary = dataset_summary(df)
            insights = generate_statistical_insights(df)

            # Generate correlation for numeric columns
            numeric_df = df.select_dtypes(include=[np.number])
            strongest_corr = None
            if len(numeric_df.columns) > 1:
                corr = numeric_df.corr().abs().unstack()
                corr = corr[corr < 1].sort_values(ascending=False)
                if not corr.empty:
                    strongest_corr = {
                        "pair": str(corr.index[0]),
                        "value": float(corr.iloc[0]),
                    }

            # Generate visualizations
            pairplot = create_pairplot(df)

            return jsonify(
                {
                    "status": "success",
                    "message": "EDA workflow completed",
                    "summary": summary,
                    "insights": insights,
                    "strongest_correlation": strongest_corr,
                    "pairplot": pairplot,
                    "missing_heatmap": create_missing_heatmap(df),
                    "next_step": "save_eda",
                }
            )

        elif workflow == "ml":
            return jsonify(
                {
                    "status": "success",
                    "message": "ML workflow selected",
                    "columns": list(df.columns),
                    "numeric_columns": df.select_dtypes(
                        include=[np.number]
                    ).columns.tolist(),
                    "categorical_columns": df.select_dtypes(
                        exclude=[np.number]
                    ).columns.tolist(),
                    "needs_user_input": True,
                    "input_required": "target_column",
                    "next_step": "ml_prepare",
                }
            )

        elif workflow == "dashboard":
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

            return jsonify(
                {
                    "status": "success",
                    "message": "Dashboard workflow selected",
                    "numeric_columns": numeric_cols,
                    "categorical_columns": cat_cols,
                    "all_columns": list(df.columns),
                    "needs_user_input": True,
                    "input_required": "visualization_choice",
                    "next_step": "create_visualization",
                }
            )

    except Exception as e:
        logging.error(f"Workflow error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ml_diagnostics", methods=["POST"])
def ml_diagnostics():
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        target_col = data.get("target_column")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]

        if target_col not in df.columns:
            return jsonify({"error": "Target column not found"}), 400

        y = df[target_col].dropna()

        # Skewness
        skewness = float(y.skew())

        # QQ Plot
        plt.figure(figsize=(6, 6))
        stats.probplot(y, dist="norm", plot=plt)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        qqplot_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        # Suggest transform
        suggested = "log" if abs(skewness) > 1 else "none"

        return jsonify(
            {
                "status": "success",
                "skewness": skewness,
                "qqplot": qqplot_base64,
                "suggested_transform": suggested,
            }
        )

    except Exception as e:
        logging.error(f"ML diagnostics error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/generate_qqplot", methods=["POST"])
def generate_qqplot():
    try:
        data = request.get_json()
        session_id = data.get("session_id")
        target_col = data.get("target_column")

        if session_id not in processing_state:
            return jsonify({"status": "error", "error": "Invalid session ID"}), 400

        df = processing_state[session_id]["df"]

        if target_col not in df.columns:
            return jsonify({"status": "error", "error": "Column not found"}), 400

        # --- Generate Q-Q Plot ---
        plt.figure(figsize=(6, 6))
        stats.probplot(df[target_col].dropna(), dist="norm", plot=plt)

        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        plt.close()

        img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return jsonify({"status": "success", "qqplot": img_base64})

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)})


@app.route("/ml_prepare", methods=["POST"])
def ml_prepare():
    """
    Prepare data for machine learning.
    Responsibilities:
    - Train-test split
    - Apply transformations
    - Scaling / PCA
    - Store processed state
    """

    try:
        data = request.get_json()

        session_id = data.get("session_id", "default")
        target_col = data.get("target_column")
        transform = data.get("transform", "standard")
        gaussian_method = data.get("gaussian_method", "log")
        apply_pca = data.get("apply_pca", False)
        n_components = int(data.get("n_components", 2))

        # ---- Validate session ----
        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"].copy()

        if target_col not in df.columns:
            return jsonify({"error": "Target column not found"}), 400

        # ---- Split data ----
        X = df.drop(columns=[target_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        transformations_applied = []

        # ---- Gaussian transforms (NO scaling here) ----
        if transform == "gaussian" and numeric_cols:
            for col in numeric_cols:
                try:
                    if gaussian_method == "log":
                        X_train[col] = np.log1p(X_train[col].abs())
                        X_test[col] = np.log1p(X_test[col].abs())
                        transformations_applied.append(f"log({col})")

                    elif gaussian_method == "sqrt":
                        X_train[col] = np.sqrt(X_train[col].abs())
                        X_test[col] = np.sqrt(X_test[col].abs())
                        transformations_applied.append(f"sqrt({col})")

                    elif gaussian_method == "reciprocal":
                        X_train[col] = np.reciprocal(X_train[col].replace(0, np.nan))
                        X_test[col] = np.reciprocal(X_test[col].replace(0, np.nan))
                        transformations_applied.append(f"1/x({col})")

                    elif gaussian_method == "boxcox" and (X_train[col] > 0).all():
                        pt = PowerTransformer(method="box-cox", standardize=False)
                        X_train[col] = pt.fit_transform(X_train[[col]]).flatten()
                        X_test[col] = pt.transform(X_test[[col]]).flatten()
                        transformations_applied.append(f"boxcox({col})")

                except Exception as e:
                    logging.warning(f"Gaussian transform failed for {col}: {e}")

        # ---- Scaling (SKIPPED for gaussian) ----
        elif numeric_cols:
            scaler_map = {
                "standard": StandardScaler(),
                "minmax": MinMaxScaler(),
                "robust": RobustScaler(),
            }

            scaler = scaler_map.get(transform, StandardScaler())
            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

            transformations_applied.append(f"{scaler.__class__.__name__}")

        # ---- PCA ----
        if apply_pca and numeric_cols:
            pca = PCA(n_components=min(n_components, len(numeric_cols)))
            X_train_pca = pca.fit_transform(X_train[numeric_cols])
            X_test_pca = pca.transform(X_test[numeric_cols])

            pca_cols = [f"PCA_{i+1}" for i in range(X_train_pca.shape[1])]

            X_train = X_train.drop(columns=numeric_cols)
            X_test = X_test.drop(columns=numeric_cols)

            X_train[pca_cols] = X_train_pca
            X_test[pca_cols] = X_test_pca

            transformations_applied.append(
                f"PCA({len(pca_cols)} components, {pca.explained_variance_ratio_.sum():.2%} variance)"
            )

        # ---- Store state ----
        processing_state[session_id].update(
            {
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "target_col": target_col,
            }
        )

        return jsonify(
            {
                "status": "success",
                "message": "ML data prepared successfully",
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
                "transformations_applied": transformations_applied,
                "next_step": "ml_diagnostics",
            }
        )

    except Exception as e:
        logging.error(f"ML prepare error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/create_visualization", methods=["POST"])
def create_visualization():
    """Create visualizations"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        viz_type = data.get("viz_type")
        x_col = data.get("x_column")
        y_col = data.get("y_column")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]

        fig, ax = plt.subplots(figsize=(12, 7))

        try:
            if viz_type == "bar":
                # Handle large categories
                if df[x_col].nunique() > 20:
                    top_categories = df[x_col].value_counts().head(20).index
                    plot_df = df[df[x_col].isin(top_categories)]
                    sns.barplot(data=plot_df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Bar Chart: {y_col} vs {x_col} (Top 20 categories)")
                else:
                    sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
                    ax.set_title(f"Bar Chart: {y_col} vs {x_col}")
                plt.xticks(rotation=45, ha="right")

            elif viz_type == "line":
                sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"Line Chart: {y_col} vs {x_col}")
                plt.xticks(rotation=45, ha="right")

            elif viz_type == "scatter":
                sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.6)
                ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")

                # Add regression line if both numeric
                if pd.api.types.is_numeric_dtype(
                    df[x_col]
                ) and pd.api.types.is_numeric_dtype(df[y_col]):
                    sns.regplot(data=df, x=x_col, y=y_col, ax=ax, scatter=False)

            elif viz_type == "pie":
                value_counts = df[x_col].value_counts()
                if len(value_counts) > 10:
                    # Show top 10 and group others
                    top10 = value_counts.head(10).copy()
                    others = value_counts[10:].sum()
                    if others > 0:
                        top10["Others"] = others
                    top10.plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_title(f"Pie Chart: {x_col} (Top 10 categories)")
                else:
                    value_counts.plot.pie(autopct="%1.1f%%", ax=ax)
                    ax.set_title(f"Pie Chart: {x_col}")
                ax.set_ylabel("")

            elif viz_type == "hist":
                sns.histplot(df[x_col].dropna(), kde=True, ax=ax, bins=30)
                ax.set_title(f"Histogram: {x_col}")
                ax.set_ylabel("Frequency")

                # Add statistics
                mean_val = df[x_col].mean()
                median_val = df[x_col].median()
                ax.axvline(mean_val, linestyle="--", label=f"Mean: {mean_val:.2f}")
                ax.axvline(
                    median_val, linestyle="--", label=f"Median: {median_val:.2f}"
                )
                ax.legend()

            elif viz_type == "heatmap":
                numeric_df = df.select_dtypes(include=[np.number])
                corr_matrix = numeric_df.corr()
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt=".2f",
                    ax=ax,
                    cmap="coolwarm",
                    center=0,
                    square=True,
                    linewidths=1,
                )
                ax.set_title("Correlation Heatmap")

            else:
                raise ValueError(f"Unknown visualization type: {viz_type}")

            plt.tight_layout()
            img_base64 = fig_to_base64(fig)

            return jsonify(
                {
                    "status": "success",
                    "visualization": img_base64,
                    "type": viz_type,
                    "columns_used": {"x": x_col, "y": y_col} if y_col else {"x": x_col},
                }
            )

        except Exception as viz_error:
            plt.close(fig)
            raise viz_error

    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download_data", methods=["POST"])
def download_data():
    """Download prepared CSV data: eda, train, or test"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        file_type = data.get("file_type", "eda")  # 'eda', 'train', 'test'

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        state = processing_state[session_id]

        if file_type == "eda":
            df = state["df"]
            csv_data = df.to_csv(index=False)
            filename = "EDA_ready_data.csv"

        elif file_type == "train":
            if "X_train" not in state or "y_train" not in state:
                return (
                    jsonify(
                        {
                            "error": "ML data not prepared yet. Please prepare ML data first."
                        }
                    ),
                    400,
                )
            X_train = state["X_train"]
            y_train = state["y_train"]
            train_df = pd.concat(
                [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
            )
            csv_data = train_df.to_csv(index=False)
            filename = "ML_train_data.csv"

        elif file_type == "test":
            if "X_test" not in state or "y_test" not in state:
                return (
                    jsonify(
                        {
                            "error": "ML data not prepared yet. Please prepare ML data first."
                        }
                    ),
                    400,
                )
            X_test = state["X_test"]
            y_test = state["y_test"]
            test_df = pd.concat(
                [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
            )
            csv_data = test_df.to_csv(index=False)
            filename = "ML_test_data.csv"

        else:
            return (
                jsonify({"error": "Invalid file type. Use: eda, train, or test"}),
                400,
            )

        return jsonify(
            {
                "status": "success",
                "filename": filename,
                "data": csv_data,
                "rows": len(csv_data.split("\n")) - 1,
            }
        )

    except Exception as e:
        logging.error(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/advanced_ml_options", methods=["POST"])
def advanced_ml_options():
    """Apply advanced preprocessing options to prepared ML data"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        options = data.get("options", {})  # dict of options

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        if "X_train" not in processing_state[session_id]:
            return jsonify({"error": "Please prepare ML data first"}), 400

        X_train = processing_state[session_id]["X_train"].copy()
        X_test = processing_state[session_id]["X_test"].copy()

        numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
        operations = []

        # Gaussian transformation
        if options.get("gaussian_transform"):
            method = options.get("gaussian_method", "log")
            for col in numeric_cols:
                try:
                    if method == "log":
                        X_train[col] = np.log1p(X_train[col].abs().astype(float))
                        X_test[col] = np.log1p(X_test[col].abs().astype(float))
                        operations.append(f"Applied log transform to {col}")
                    elif method == "sqrt":
                        X_train[col] = np.sqrt(X_train[col].abs().astype(float))
                        X_test[col] = np.sqrt(X_test[col].abs().astype(float))
                        operations.append(f"Applied sqrt transform to {col}")
                    elif method == "boxcox":
                        # Box-Cox requires positive values
                        if (X_train[col] > 0).all():
                            pt = PowerTransformer(method="box-cox")
                            X_train[col] = pt.fit_transform(X_train[[col]]).flatten()
                            X_test[col] = pt.transform(X_test[[col]]).flatten()
                            operations.append(f"Applied box-cox transform to {col}")
                        else:
                            operations.append(
                                f"Skipped box-cox for {col} (non-positive values)"
                            )
                except Exception as e:
                    logging.warning(f"Transform failed for {col}: {e}")
                    operations.append(f"Transform failed for {col}: {e}")

        # Feature interaction
        if options.get("create_interactions") and len(numeric_cols) > 1:
            # Limit create to a small number to avoid explosion
            limited_numeric = numeric_cols[:4]
            for i, col1 in enumerate(limited_numeric[:-1]):
                for col2 in limited_numeric[i + 1 :]:
                    new_col = f"{col1}_x_{col2}"
                    X_train[new_col] = X_train[col1] * X_train[col2]
                    X_test[new_col] = X_test[col1] * X_test[col2]
                    operations.append(f"Created interaction: {new_col}")

        # Update state
        processing_state[session_id]["X_train"] = X_train
        processing_state[session_id]["X_test"] = X_test

        return jsonify(
            {
                "status": "success",
                "message": "Advanced preprocessing applied",
                "operations": operations,
                "new_shape": list(X_train.shape),
            }
        )

    except Exception as e:
        logging.error(f"Advanced ML options error: {e}")
        return jsonify({"error": str(e)}), 500


# Run the app (for development)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
