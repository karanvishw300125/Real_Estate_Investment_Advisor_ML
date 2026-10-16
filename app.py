import pickle
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import json

# 1. CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent
MODEL_ROOT = BASE_DIR / "saved_models"
METADATA_PATH = MODEL_ROOT / "app_metadata.json"

DB_CANDIDATES = ["real_estate.db", "realestate.db", "real_estate.sqlite", "housing.db"]
EXCLUDED_DBS = {"mlflow.db", "mlflow.sqlite", "mlflow.sqlite3"}

# Files inside saved_models/<task>/ that are NOT models
ARTIFACT_FILES = {"scaler.pkl", "numeric_features.pkl", "feature_columns.pkl"}

CATEGORICAL_FIELDS = [
    "Property_Type", "Furnished_Status", "Facing",
    "Owner_Type", "Availability_Status", "State", "City",
]

st.set_page_config(page_title="Real Estate ML Analytics", page_icon="🏡", layout="wide")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    .hero {
        padding: 25px; border-radius: 16px; margin-bottom: 20px;
        background: linear-gradient(120deg, #dbeafe, #eff6ff); color: #0f172a;
    }
    .hero h1, .hero p { color: #0f172a; }
    </style>
    """,
    unsafe_allow_html=True,
)


def _stretch_kwargs():
    """Newer Streamlit uses width='stretch'; older versions use use_container_width."""
    try:
        major, minor = (int(x) for x in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 50):
            return {"width": "stretch"}
    except Exception:
        pass
    return {"use_container_width": True}


STRETCH = _stretch_kwargs()


def show_df(frame):
    st.dataframe(frame, **STRETCH)


def show_fig(figure):
    st.plotly_chart(figure, **STRETCH)


def pretty_name(value):
    return str(value).replace("_", " ").replace("-", " ").title()


# 2. DATA LOADING (cached)

def find_database():
    for name in DB_CANDIDATES:
        path = BASE_DIR / name
        if path.exists() and path.name.lower() not in EXCLUDED_DBS:
            return path

    for pattern in ("*.db", "*.sqlite", "*.sqlite3"):
        for path in BASE_DIR.glob(pattern):
            if path.name.lower() not in EXCLUDED_DBS:
                return path
    return None


@st.cache_data(show_spinner=False)
def get_tables(db_path):
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ).fetchall()
    return [row[0] for row in rows]


@st.cache_data(show_spinner="Loading data...")
def load_table(db_path, table_name):
    safe_name = '"' + table_name.replace('"', '""') + '"'
    uri = Path(db_path).resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as conn:
        return pd.read_sql_query(f"SELECT * FROM {safe_name}", conn)


@st.cache_data(show_spinner=False)
def load_metadata():
    if not METADATA_PATH.exists():
        return None
    with open(METADATA_PATH) as file:
        return json.load(file)


@st.cache_resource(show_spinner=False)
def load_artifact(path):
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as file:
            return pickle.load(file)


def get_model_files(task):
    folder = MODEL_ROOT / task.lower()
    if not folder.exists():
        return []
    return sorted(
        (p for p in folder.glob("*.pkl") if p.name.lower() not in ARTIFACT_FILES),
        key=lambda p: p.name.lower(),
    )


def load_task_artifacts(task):
    folder = MODEL_ROOT / task.lower()
    needed = {
        "scaler": "scaler.pkl",
        "numeric_features": "numeric_features.pkl",
        "feature_columns": "feature_columns.pkl",
    }
    artifacts = {}
    for key, filename in needed.items():
        path = folder / filename
        artifacts[key] = load_artifact(path) if path.exists() else None
    return artifacts


def numeric_columns(frame):
    return frame.select_dtypes(include=np.number).columns.tolist()


def categorical_columns(frame):
    return frame.select_dtypes(include=["object", "category", "bool", "string"]).columns.tolist()


db_path = find_database()
metadata = load_metadata()

data = pd.DataFrame()
db_error = None
table_name = None
available_tables = []

if db_path is not None:
    try:
        available_tables = get_tables(str(db_path))
        if available_tables:
            table_name = "properties" if "properties" in available_tables else available_tables[0]
            data = load_table(str(db_path), table_name)
    except Exception as error:
        db_error = str(error)


# 3. PREDICTION HELPERS

TRANSPORT_MAP = {"Low": 0, "Medium": 1, "High": 2}


def build_feature_row(feature_cols, meta, inputs):
    """Rebuild the exact training-time feature vector from friendly inputs."""

    size = max(float(inputs["Size_in_SqFt"]), 1.0)
    base = {
        "BHK": inputs["BHK"],
        "Size_in_SqFt": size,
        "Price_in_Lakhs": inputs["Price_in_Lakhs"],
        "Price_per_SqFt": inputs["Price_in_Lakhs"] * 100000 / size,
        "Year_Built": inputs["Year_Built"],
        "Floor_No": min(inputs["Floor_No"], inputs["Total_Floors"]),
        "Total_Floors": inputs["Total_Floors"],
        "Age_of_Property": meta["reference_year"] - inputs["Year_Built"],
        "Nearby_Schools": inputs["Nearby_Schools"],
        "Nearby_Hospitals": inputs["Nearby_Hospitals"],
        "Public_Transport_Accessibility": TRANSPORT_MAP[inputs["Public_Transport_Accessibility"]],
        "Parking_Space": int(inputs["Parking_Space"] == "Yes"),
        "Security": int(inputs["Security"] == "Yes"),
        "Amenity_Density_Score": len(inputs["Amenities"]),
        "Locality_Frequency": meta["locality_frequency"].get(inputs["Locality"], 0),
    }

    chosen = {a.lower().replace(" ", "_") for a in inputs["Amenities"]}
    for amenity in meta["amenities"]:
        base[amenity] = int(amenity in chosen)

    row = {}
    for feature in feature_cols:
        if feature in base:
            row[feature] = base[feature]
            continue

        row[feature] = 0
        for field in CATEGORICAL_FIELDS:
            if feature.startswith(field + "_"):
                row[feature] = int(feature == f"{field}_{inputs[field]}")
                break

    return pd.DataFrame([row], columns=feature_cols)


def scale_row(row, artifacts):
    """Scale ONLY the numeric columns that the scaler was fitted on."""
    scaled = row.copy()
    scaler = artifacts.get("scaler")
    numeric_features = artifacts.get("numeric_features")

    if scaler is not None and numeric_features:
        cols = [c for c in numeric_features if c in scaled.columns]
        scaled[cols] = scaler.transform(scaled[cols].astype(float))

    return scaled


def prediction_inputs(meta, task):
    """Interactive property form. Returns (inputs, submitted)."""

    cats, defaults, ranges = meta["categories"], meta["defaults"], meta["ranges"]
    inputs = {}

    st.subheader("Enter Property Details")

    left, middle, right = st.columns(3)

    with left:
        inputs["State"] = st.selectbox("State", cats["State"], key=f"{task}_state")
        city_options = [
            c for c in cats["City"] if meta["city_to_state"].get(c) == inputs["State"]
        ] or cats["City"]
        inputs["City"] = st.selectbox("City", city_options, key=f"{task}_city")
        inputs["Locality"] = st.selectbox(
            "Locality", sorted(meta["locality_frequency"]), key=f"{task}_locality"
        )
        inputs["Property_Type"] = st.selectbox("Property Type", cats["Property_Type"], key=f"{task}_ptype")
        inputs["Furnished_Status"] = st.selectbox("Furnished Status", cats["Furnished_Status"], key=f"{task}_furn")

    with middle:
        inputs["BHK"] = st.number_input("BHK", int(ranges["BHK"][0]), int(ranges["BHK"][1]), int(defaults["BHK"]), key=f"{task}_bhk")
        inputs["Size_in_SqFt"] = st.number_input("Size (SqFt)", 100, 20000, int(defaults["Size_in_SqFt"]), step=50, key=f"{task}_size")
        inputs["Price_in_Lakhs"] = st.number_input("Current Price (Lakhs)", 1.0, 5000.0, float(defaults["Price_in_Lakhs"]), step=5.0, key=f"{task}_price")
        inputs["Year_Built"] = st.number_input("Year Built", 1900, int(meta["reference_year"]), int(defaults["Year_Built"]), key=f"{task}_year")
        inputs["Total_Floors"] = st.number_input("Total Floors", 1, 100, int(defaults["Total_Floors"]), key=f"{task}_tfloors")
        inputs["Floor_No"] = st.number_input("Floor No", 0, 100, int(defaults["Floor_No"]), key=f"{task}_floor")

    with right:
        inputs["Nearby_Schools"] = st.number_input("Nearby Schools", 0, 50, int(defaults["Nearby_Schools"]), key=f"{task}_schools")
        inputs["Nearby_Hospitals"] = st.number_input("Nearby Hospitals", 0, 50, int(defaults["Nearby_Hospitals"]), key=f"{task}_hosp")
        inputs["Public_Transport_Accessibility"] = st.selectbox("Public Transport", list(TRANSPORT_MAP), index=1, key=f"{task}_transport")
        inputs["Parking_Space"] = st.selectbox("Parking Space", ["Yes", "No"], key=f"{task}_parking")
        inputs["Security"] = st.selectbox("Security", ["Yes", "No"], key=f"{task}_security")
        inputs["Facing"] = st.selectbox("Facing", cats["Facing"], key=f"{task}_facing")

    low, high = st.columns(2)
    with low:
        inputs["Owner_Type"] = st.selectbox("Owner Type", cats["Owner_Type"], key=f"{task}_owner")
        inputs["Availability_Status"] = st.selectbox("Availability", cats["Availability_Status"], key=f"{task}_avail")
    with high:
        amenity_labels = [a.replace("_", " ").title() for a in meta["amenities"]]
        inputs["Amenities"] = st.multiselect(
            "Amenities", amenity_labels, default=amenity_labels, key=f"{task}_amen"
        )

    submitted = st.button("Run Prediction", type="primary", key=f"{task}_submit")
    return inputs, submitted


# 4. SIDEBAR

st.sidebar.title("🏡 Real Estate ML")
st.sidebar.caption("Analytics and Investment Insights")

page = st.sidebar.radio("Navigation", ["Introduction", "EDA Visualization", "Prediction"])

st.sidebar.divider()
st.sidebar.caption("Real Estate Investment Advisor")


# 5. INTRODUCTION

if page == "Introduction":

    st.markdown(
        """
        <div class="hero">
            <h1>Real Estate Investment Advisor</h1>
            <p>Explore property data, visualise market patterns and predict
            investment quality and 5-year future price with trained ML models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    n_models = len(get_model_files("Classification")) + len(get_model_files("Regression"))

    c1, c2, c3 = st.columns(3)
    c1.metric("Dashboard Pages", "3")
    c2.metric("Dataset Rows", f"{len(data):,}" if not data.empty else "Not loaded")
    c3.metric("Saved Models", n_models)

    st.subheader("Platform Features")
    left, middle, right = st.columns(3)

    with left:
        st.markdown("### 📊 EDA Visualization")
        st.write("Distributions, outliers, correlations and categorical patterns.")
    with middle:
        st.markdown("### 🤖 Classification")
        st.write("Predicts whether a property is a **Good Investment**.")
    with right:
        st.markdown("### 📈 Regression")
        st.write("Predicts the property price after **5 years** (in Lakhs).")

    st.info(
        "Predictions use the same features and preprocessing as training: "
        "one-hot encoding, locality frequency, amenity flags and scaled numeric columns."
    )

    with st.expander("Project File Status"):
        st.write("Project folder:", str(BASE_DIR))
        st.write("Property database:", str(db_path) if db_path else "Not found")
        st.write("Selected table:", table_name or "Not loaded")
        st.write("App metadata:", "Found" if metadata else "Missing - run real_estate.ipynb")
        st.write("Classification models:", [p.name for p in get_model_files("Classification")] or "None")
        st.write("Regression models:", [p.name for p in get_model_files("Regression")] or "None")
        if db_error:
            st.error(db_error)


# 6. EDA VISUALIZATION

elif page == "EDA Visualization":

    st.title("📊 Exploratory Data Analysis")

    eda = pd.DataFrame()

    if db_path is not None and available_tables and not db_error:
        selected_table = st.selectbox(
            "Select database table", available_tables,
            index=available_tables.index(table_name) if table_name in available_tables else 0,
        )
        eda = load_table(str(db_path), selected_table)
    else:
        if db_error:
            st.error(f"Database error: {db_error}")
        st.info("No property database found. Upload a CSV to explore.")
        uploaded = st.file_uploader("Upload property dataset", type=["csv"])
        if uploaded is not None:
            try:
                eda = pd.read_csv(uploaded)
            except Exception as error:
                st.error(f"CSV loading error: {error}")

    if eda.empty:
        st.warning("A non-empty dataset is required for EDA.")

    else:
        num_cols = numeric_columns(eda)
        cat_cols = categorical_columns(eda)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Rows", f"{len(eda):,}")
        c2.metric("Columns", len(eda.columns))
        c3.metric("Duplicates", int(eda.duplicated().sum()))
        c4.metric("Missing Cells", int(eda.isna().sum().sum()))

        with st.expander("Dataset Preview", expanded=True):
            show_df(eda.head(100))
            if num_cols:
                st.subheader("Descriptive Statistics")
                show_df(eda[num_cols].describe().T)

        # Charts that draw every point use a sample so the app stays fast
        sample = eda.sample(n=min(len(eda), 5000), random_state=42)

        st.subheader("1. Missing Values")
        missing = eda.isna().sum()
        missing = missing[missing > 0].sort_values(ascending=False)
        if missing.empty:
            st.success("No missing values found.")
        else:
            missing_df = missing.rename("Missing Count").rename_axis("Column").reset_index()
            show_fig(px.bar(missing_df, x="Column", y="Missing Count", title="Missing Values by Column"))

        if num_cols:
            st.subheader("2. Numeric Distribution")
            selected_num = st.selectbox("Select numeric column", num_cols)
            left, right = st.columns(2)
            with left:
                show_fig(px.histogram(eda, x=selected_num, nbins=40, title=f"Distribution of {pretty_name(selected_num)}"))
            with right:
                show_fig(px.box(sample, y=selected_num, title=f"Box Plot of {pretty_name(selected_num)}"))

        if len(num_cols) >= 2:
            st.subheader("3. Correlation Heatmap")
            show_fig(px.imshow(
                eda[num_cols].corr(), text_auto=".2f", aspect="auto",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                title="Numeric Feature Correlations",
            ))

            st.subheader("4. Scatter Plot")
            left, right = st.columns(2)
            with left:
                x_axis = st.selectbox("X-axis", num_cols, key="scatter_x")
            with right:
                y_axis = st.selectbox("Y-axis", num_cols, index=1, key="scatter_y")
            color_by = st.selectbox("Color by", ["None"] + cat_cols)

            show_fig(px.scatter(
                sample, x=x_axis, y=y_axis,
                color=None if color_by == "None" else color_by,
                opacity=0.6, title=f"{pretty_name(y_axis)} vs {pretty_name(x_axis)} (5,000-row sample)",
            ))

            st.subheader("5. Scatter Matrix")
            pair_cols = st.multiselect(
                "Choose up to 5 numeric columns", num_cols,
                default=num_cols[:3], max_selections=5,
            )
            if len(pair_cols) >= 2:
                show_fig(px.scatter_matrix(
                    sample[pair_cols].dropna().head(1200),
                    dimensions=pair_cols, title="Scatter Matrix",
                ))

        if cat_cols:
            st.subheader("6. Categorical Analysis")
            selected_cat = st.selectbox("Select categorical column", cat_cols)

            counts = (
                eda[selected_cat].fillna("Missing").astype(str)
                .value_counts().head(25)
                .rename_axis(selected_cat).reset_index(name="Count")
            )

            left, right = st.columns(2)
            with left:
                show_fig(px.bar(counts, x=selected_cat, y="Count", title=f"Top Categories: {pretty_name(selected_cat)}"))
            with right:
                show_fig(px.pie(counts.head(10), names=selected_cat, values="Count", title=f"Category Share: {pretty_name(selected_cat)}"))

            if num_cols:
                st.subheader("7. Average of a Metric by Category")
                metric_col = st.selectbox("Metric", num_cols, key="group_metric")
                top_groups = eda[selected_cat].astype(str).value_counts().head(25).index
                grouped = (
                    eda[eda[selected_cat].astype(str).isin(top_groups)]
                    .groupby(selected_cat)[metric_col].mean()
                    .sort_values(ascending=False).reset_index()
                )
                show_fig(px.bar(grouped, x=selected_cat, y=metric_col, title=f"Average {pretty_name(metric_col)} by {pretty_name(selected_cat)}"))

        year_cols = [c for c in eda.columns if "year" in c.lower() and c in num_cols]
        if year_cols and num_cols:
            st.subheader("8. Trend by Year")
            left, right = st.columns(2)
            with left:
                year_col = st.selectbox("Year column", year_cols)
            with right:
                trend_metric = st.selectbox("Metric", [c for c in num_cols if c != year_col], key="trend_metric")

            trend = eda.groupby(year_col, as_index=False)[trend_metric].mean().sort_values(year_col)
            show_fig(px.line(trend, x=year_col, y=trend_metric, markers=True, title=f"Average {pretty_name(trend_metric)} by {pretty_name(year_col)}"))


# 7. PREDICTION

elif page == "Prediction":

    st.title("🤖 Property Prediction")

    task = st.radio("Select Prediction Type", ["Classification", "Regression"], horizontal=True)

    if task == "Classification":
        st.caption("Is this property a good investment?")
    else:
        st.caption("What will this property cost after 5 years (Lakhs)?")

    model_files = get_model_files(task)

    if metadata is None:
        st.error("`saved_models/app_metadata.json` not found. Run the last cell of real_estate.ipynb first.")

    elif not model_files:
        st.error(
            f"No {task.lower()} models found in `saved_models/{task.lower()}/`. "
            f"Run {task.lower()}_model.ipynb first."
        )

    else:
        model_path = st.selectbox("Select Model", model_files, format_func=lambda p: pretty_name(p.stem))

        try:
            model = load_artifact(model_path)
            artifacts = load_task_artifacts(task)
            feature_cols = list(artifacts["feature_columns"] or getattr(model, "feature_names_in_", []))

            metrics_path = MODEL_ROOT / task.lower() / "metrics.csv"
            if metrics_path.exists():
                with st.expander("Model performance on the test set"):
                    show_df(pd.read_csv(metrics_path))

            if not feature_cols:
                st.error("Feature list missing. Re-run the training notebook to save feature_columns.pkl.")

            else:
                inputs, submitted = prediction_inputs(metadata, task)

                if submitted:
                    try:
                        row = build_feature_row(feature_cols, metadata, inputs)
                        model_input = scale_row(row, artifacts)
                        prediction = model.predict(model_input)

                        st.subheader("Prediction Result")

                        if task == "Classification":
                            label = int(prediction[0])
                            if label == 1:
                                st.success("✅ Good Investment")
                            else:
                                st.error("❌ Not a Good Investment")

                            if hasattr(model, "predict_proba"):
                                probs = model.predict_proba(model_input)[0]
                                prob_df = pd.DataFrame({
                                    "Class": ["Not Good", "Good Investment"],
                                    "Probability (%)": (probs * 100).round(2),
                                })
                                show_df(prob_df)
                                show_fig(px.bar(prob_df, x="Class", y="Probability (%)", title="Class Probabilities"))
                        else:
                            future = float(prediction[0])
                            current = float(inputs["Price_in_Lakhs"])
                            growth = (future / current - 1) * 100 if current else 0.0

                            a, b = st.columns(2)
                            a.metric("Predicted Price after 5 Years", f"₹ {future:,.2f} Lakhs", f"{growth:+.1f}% vs current")
                            b.metric("Current Price", f"₹ {current:,.2f} Lakhs")

                        with st.expander("Feature vector sent to the model"):
                            show_df(row.T.rename(columns={0: "value"}))

                    except Exception as error:
                        st.error(f"Prediction failed: {error}")

        except Exception as error:
            st.error(f"Model load failed: {error}")