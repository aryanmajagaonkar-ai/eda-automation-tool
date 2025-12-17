import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")  # Use non-GUI backend
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
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = "your-secret-key-change-this-in-production"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max file size

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# Global storage for processing state
processing_state = {}


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string for web display"""
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


# ==================== API ENDPOINTS ====================


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint for connection testing"""
    return jsonify(
        {"status": "ok", "message": "Flask server is running", "version": "1.0"}
    )


@app.route("/")
def home():
    return "App running on Render!"


if __name__ == "__main__":
    app.run()


@app.route("/", methods=["GET"])
def home():
    """Root endpoint"""
    return jsonify(
        {
            "message": "EDA Flask API is running",
            "endpoints": [
                "/health",
                "/upload_csv",
                "/ai_suggestions",
                "/handle_suggestions",
                "/missing_values",
                "/workflow",
                "/ml_prepare",
                "/create_visualization",
                "/download",
            ],
        }
    )


@app.route("/upload_csv", methods=["POST"])
def upload_csv():
    """Step 1: Upload and initial processing"""
    try:
        logging.info("Received upload request")

        if "file" not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400

        logging.info(f"Processing file: {file.filename}")

        # Read CSV
        df = pd.read_csv(file)
        logging.info(f"CSV loaded: {df.shape}")

        # Initial cleaning
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_")
            .str.replace(r"[^\w\s]", "", regex=True)
        )
        df = df.drop_duplicates().reset_index(drop=True)

        # Store in session
        session_id = request.form.get("session_id", "default")
        processing_state[session_id] = {"df": df, "step": "uploaded"}
        logging.info(f"Session {session_id} created")

        # Generate initial insights
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        missing_per_column = df.isnull().sum().to_dict()

        response = {
            "status": "success",
            "message": "Dataset loaded successfully",
            "shape": df.shape,
            "columns": list(df.columns),
            "numeric_cols": numeric_cols,
            "categorical_cols": cat_cols,
            "missing_values": int(df.isnull().sum().sum()),
            "missing_per_column": missing_per_column,
            "duplicates": 0,
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
    """Step 2: AI Agent Suggestions"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")

        if session_id not in processing_state:
            return (
                jsonify({"error": "Session not found. Please upload CSV again."}),
                404,
            )

        df = processing_state[session_id]["df"]

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        suggestions = []

        # 1. High missing columns
        missing_ratio = df.isnull().mean() * 100
        high_missing = missing_ratio[missing_ratio > 40].index.tolist()
        if high_missing:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "high",
                    "message": f"Columns with >40% missing: {high_missing}",
                    "columns": high_missing,
                }
            )

        # 2. High uniqueness columns
        high_unique = [col for col in df.columns if df[col].nunique() > (0.9 * len(df))]
        if high_unique:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "medium",
                    "message": f"High uniqueness columns (likely IDs): {high_unique}",
                    "columns": high_unique,
                }
            )

        # 3. Low variance columns
        low_var = [col for col in numeric_cols if df[col].nunique() <= 1]
        if low_var:
            suggestions.append(
                {
                    "type": "drop_columns",
                    "severity": "medium",
                    "message": f"Constant/low variance columns: {low_var}",
                    "columns": low_var,
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
                        "message": f"Highly skewed features: {skewed}",
                        "columns": skewed,
                    }
                )

        # All suggested columns to drop
        all_drop_cols = list(set(high_missing + high_unique + low_var))

        return jsonify(
            {
                "status": "success",
                "suggestions": suggestions,
                "suggested_drop_columns": all_drop_cols,
                "needs_user_input": True,
                "next_step": "handle_suggestions",
            }
        )

    except Exception as e:
        logging.error(f"AI suggestions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/handle_suggestions", methods=["POST"])
def handle_suggestions():
    """Step 3: Handle AI suggestions based on user choice"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        action = data.get("action")
        custom_columns = data.get("columns", [])

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]

        if action == "auto":
            # Auto-clean mode
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            cat_cols = df.select_dtypes(exclude=[np.number]).columns

            # Fill missing values
            for col in numeric_cols:
                df[col] = df[col].fillna(df[col].median())
            for col in cat_cols:
                if df[col].isnull().any():
                    mode_val = df[col].mode()
                    df[col] = df[col].fillna(
                        mode_val[0] if not mode_val.empty else "Unknown"
                    )

            # Transform skewed features
            skew_vals = df[numeric_cols].skew()
            skewed = skew_vals[abs(skew_vals) > 1]
            for col in skewed.index:
                df[col] = np.log1p(df[col].abs())

            message = "Auto-cleaning applied: missing values filled, skewed features transformed"

        elif action == "manual" and custom_columns:
            # Drop custom columns
            valid_cols = [c for c in custom_columns if c in df.columns]
            df = df.drop(columns=valid_cols)
            message = f"Dropped columns: {valid_cols}"

        else:
            message = "Skipped AI suggestions"

        # Update state
        processing_state[session_id]["df"] = df

        return jsonify(
            {
                "status": "success",
                "message": message,
                "shape": df.shape,
                "missing_values": int(df.isnull().sum().sum()),
                "missing_heatmap": create_missing_heatmap(df),
                "next_step": "choose_workflow",
            }
        )

    except Exception as e:
        logging.error(f"Handle suggestions error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/missing_values", methods=["POST"])
def handle_missing_values():
    """Handle missing values with user's choice"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        method = data.get("method")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(exclude=[np.number]).columns

        if method == "mean":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == "median":
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        else:  # mode
            for col in numeric_cols:
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else 0)

        df[cat_cols] = df[cat_cols].fillna("Unknown")

        processing_state[session_id]["df"] = df

        return jsonify(
            {
                "status": "success",
                "message": f"Missing values handled using {method}",
                "missing_values": int(df.isnull().sum().sum()),
                "head": df.head(10).to_html(classes="table table-striped"),
            }
        )

    except Exception as e:
        logging.error(f"Missing values error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/workflow", methods=["POST"])
def choose_workflow():
    """Step 4: Choose workflow (EDA/ML/Dashboard)"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        workflow = data.get("workflow")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]
        processing_state[session_id]["workflow"] = workflow

        if workflow == "eda":
            return jsonify(
                {
                    "status": "success",
                    "message": "EDA workflow selected",
                    "summary": {
                        "total_rows": int(df.shape[0]),
                        "total_columns": int(df.shape[1]),
                        "numeric_columns": len(
                            df.select_dtypes(include=[np.number]).columns
                        ),
                        "categorical_columns": len(
                            df.select_dtypes(exclude=[np.number]).columns
                        ),
                        "missing_cells": int(df.isnull().sum().sum()),
                    },
                    "next_step": "save_eda",
                }
            )

        elif workflow == "ml":
            return jsonify(
                {
                    "status": "success",
                    "message": "ML workflow selected",
                    "columns": list(df.columns),
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
                    "needs_user_input": True,
                    "input_required": "visualization_choice",
                    "next_step": "create_visualization",
                }
            )

    except Exception as e:
        logging.error(f"Workflow error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ml_prepare", methods=["POST"])
def ml_prepare():
    """ML Preparation with train-test split"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        target_col = data.get("target_column")
        transform = data.get("transform", "standard")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]

        if target_col not in df.columns:
            return jsonify({"error": f"Column {target_col} not found"}), 400

        # Split data
        X = df.drop(columns=[target_col])
        y = df[target_col]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Apply transformation
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            if transform == "standard":
                scaler = StandardScaler()
            elif transform == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = RobustScaler()

            X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
            X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

        # Store results
        processing_state[session_id]["X_train"] = X_train
        processing_state[session_id]["X_test"] = X_test
        processing_state[session_id]["y_train"] = y_train
        processing_state[session_id]["y_test"] = y_test

        # Determine problem type
        problem_type = (
            "regression" if pd.api.types.is_numeric_dtype(y) else "classification"
        )

        return jsonify(
            {
                "status": "success",
                "message": "ML preparation complete",
                "train_shape": list(X_train.shape),
                "test_shape": list(X_test.shape),
                "problem_type": problem_type,
                "transform_applied": transform,
                "next_step": "save_ml_data",
            }
        )

    except Exception as e:
        logging.error(f"ML prepare error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/create_visualization", methods=["POST"])
def create_visualization():
    """Create visualization for dashboard"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        viz_type = data.get("viz_type")
        x_col = data.get("x_column")
        y_col = data.get("y_column")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        df = processing_state[session_id]["df"]

        fig, ax = plt.subplots(figsize=(10, 6))

        if viz_type == "bar":
            sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"Bar Chart: {y_col} vs {x_col}")

        elif viz_type == "line":
            sns.lineplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"Line Chart: {y_col} vs {x_col}")

        elif viz_type == "scatter":
            sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"Scatter Plot: {y_col} vs {x_col}")

        elif viz_type == "pie":
            df[x_col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_title(f"Pie Chart: {x_col}")
            ax.set_ylabel("")

        elif viz_type == "hist":
            sns.histplot(df[x_col], kde=True, ax=ax)
            ax.set_title(f"Histogram: {x_col}")

        elif viz_type == "heatmap":
            numeric_df = df.select_dtypes(include=[np.number])
            sns.heatmap(
                numeric_df.corr(), annot=True, fmt=".2f", ax=ax, cmap="coolwarm"
            )
            ax.set_title("Correlation Heatmap")

        img_base64 = fig_to_base64(fig)

        return jsonify(
            {"status": "success", "visualization": img_base64, "type": viz_type}
        )

    except Exception as e:
        logging.error(f"Visualization error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/download", methods=["POST"])
def download_data():
    """Generate download file"""
    try:
        data = request.get_json()
        session_id = data.get("session_id", "default")
        file_type = data.get("file_type")

        if session_id not in processing_state:
            return jsonify({"error": "Session not found"}), 404

        state = processing_state[session_id]

        if file_type == "eda":
            df = state["df"]
            csv_data = df.to_csv(index=False)
            filename = "EDA_ready_data.csv"

        elif file_type == "train":
            X_train = state.get("X_train")
            y_train = state.get("y_train")
            if X_train is None or y_train is None:
                return jsonify({"error": "ML data not prepared yet"}), 400
            train_df = pd.concat(
                [X_train.reset_index(drop=True), y_train.reset_index(drop=True)], axis=1
            )
            csv_data = train_df.to_csv(index=False)
            filename = "ML_train_data.csv"

        elif file_type == "test":
            X_test = state.get("X_test")
            y_test = state.get("y_test")
            if X_test is None or y_test is None:
                return jsonify({"error": "ML data not prepared yet"}), 400
            test_df = pd.concat(
                [X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1
            )
            csv_data = test_df.to_csv(index=False)
            filename = "ML_test_data.csv"

        else:
            return jsonify({"error": "Invalid file type"}), 400

        return jsonify({"status": "success", "filename": filename, "data": csv_data})

    except Exception as e:
        logging.error(f"Download error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 60)
    print("🚀 EDA Flask Server Starting...")
    print("=" * 60)
    print("Server URL: http://localhost:5000")
    print("Health Check: http://localhost:5000/health")
    print("=" * 60)
    app.run(debug=True, host="0.0.0.0", port=5000)
