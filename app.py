import os
import json
import ast
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree
from pytorch_tabnet.tab_model import TabNetClassifier


st.set_page_config(
    page_title="Restaurant Success Predictor",
    layout="wide"
)

st.title("Restaurant Success Prediction System")

st.write("""
This application predicts the potential success of a restaurant location using
the same preprocessing logic used in the training notebooks: XGBoost one-hot
features, TabNet numeric features, and GNN spatial embeddings.
""")


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

def model_path(filename):
    return os.path.join(MODEL_DIR, filename)


REQUIRED_FILES = [
    "xgb_model.pkl",
    "xgb_feature_columns.json",
    "tabnet_final_model.zip",
    "tabnet_scaler.pkl",
    "tabnet_feature_columns.json",
    "gnn_embeddings.csv",
    "model_dataset.csv",
]

missing_files = [f for f in REQUIRED_FILES if not os.path.exists(model_path(f))]

if missing_files:
    st.error("Missing required files:")
    st.write(missing_files)
    st.info("Place these files in the same folder as this Streamlit app file.")
    st.stop()


@st.cache_resource
def load_artifacts():
    xgb_model = joblib.load(model_path("xgb_model.pkl"))

    tabnet_model = TabNetClassifier()
    tabnet_model.load_model(model_path("tabnet_final_model.zip"))

    tabnet_scaler = joblib.load(model_path("tabnet_scaler.pkl"))

    with open(model_path("xgb_feature_columns.json"), "r") as f:
        xgb_features = json.load(f)

    with open(model_path("tabnet_feature_columns.json"), "r") as f:
        tabnet_features = json.load(f)

    gnn_embeddings = pd.read_csv(model_path("gnn_embeddings.csv"))

    return xgb_model, tabnet_model, tabnet_scaler, xgb_features, tabnet_features, gnn_embeddings


@st.cache_data
def load_training_dataset():
    return pd.read_csv(model_path("model_dataset.csv"))


(
    xgb_model,
    tabnet_model,
    tabnet_scaler,
    xgb_features,
    tabnet_features,
    gnn_embeddings,
) = load_artifacts()

training_df = load_training_dataset()


@st.cache_data
def build_preprocessing_profile(training_df):
    df = training_df.copy()

    if "target" in df.columns:
        raw_X = df.drop(columns=["target"]).copy()
    else:
        raw_X = df.copy()

    original_columns = list(raw_X.columns)
    lower_to_original = {c.strip().lower(): c for c in original_columns}

    xgb_missing_ratio = raw_X.isnull().mean()
    xgb_high_missing = xgb_missing_ratio[xgb_missing_ratio > 0.70].index.tolist()
    xgb_base = raw_X.drop(columns=xgb_high_missing, errors="ignore").copy()

    xgb_numeric_cols = xgb_base.select_dtypes(include=[np.number]).columns.tolist()
    xgb_categorical_cols = xgb_base.select_dtypes(exclude=[np.number]).columns.tolist()

    xgb_medians = {}
    for col in xgb_numeric_cols:
        median_value = pd.to_numeric(xgb_base[col], errors="coerce").median()
        xgb_medians[col] = 0 if pd.isna(median_value) else float(median_value)

    xgb_modes = {}
    for col in xgb_categorical_cols:
        filled = xgb_base[col].fillna("Unknown").astype(str)
        mode_value = filled.mode()
        xgb_modes[col] = mode_value.iloc[0] if len(mode_value) else "Unknown"

    numeric_X = raw_X.select_dtypes(include=[np.number]).copy()
    tab_missing_ratio = numeric_X.isnull().mean()
    tab_high_missing = tab_missing_ratio[tab_missing_ratio > 0.70].index.tolist()
    tab_base = numeric_X.drop(columns=tab_high_missing, errors="ignore").copy()

    tab_medians = {}
    for col in tab_base.columns:
        median_value = pd.to_numeric(tab_base[col], errors="coerce").median()
        tab_medians[col] = 0 if pd.isna(median_value) else float(median_value)

    coords = None
    if "latitude" in lower_to_original and "longitude" in lower_to_original:
        lat_col = lower_to_original["latitude"]
        lon_col = lower_to_original["longitude"]

        coords = raw_X[[lat_col, lon_col]].copy()
        coords.columns = ["latitude", "longitude"]
        coords["latitude"] = pd.to_numeric(coords["latitude"], errors="coerce")
        coords["longitude"] = pd.to_numeric(coords["longitude"], errors="coerce")
        coords = coords.dropna()
        coords = coords[
            coords["latitude"].between(-90, 90)
            & coords["longitude"].between(-180, 180)
        ]

    return {
        "lower_to_original": lower_to_original,
        "xgb_high_missing": xgb_high_missing,
        "xgb_numeric_cols": xgb_numeric_cols,
        "xgb_categorical_cols": xgb_categorical_cols,
        "xgb_medians": xgb_medians,
        "xgb_modes": xgb_modes,
        "tab_high_missing": tab_high_missing,
        "tab_medians": tab_medians,
        "coords": coords,
    }


profile = build_preprocessing_profile(training_df)


def clean_candidate_columns(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename_map = {}
    for col in df.columns:
        lower = col.strip().lower()
        if lower in profile["lower_to_original"]:
            rename_map[col] = profile["lower_to_original"][lower]

    df = df.rename(columns=rename_map)

    aliases = {
        "price_range": "RestaurantsPriceRange2",
        "restaurantspricerange2": "RestaurantsPriceRange2",
        "expected_rating": "stars",
        "business_rating": "business_stars",
        "competition_count": "num_neighbors_1km",
        "nearby_competition_count": "num_neighbors_1km",
        "avg_competitor_rating": "avg_neighbor_rating",
        "customer_activity": "customer_activity_score",
    }

    for src, dst in aliases.items():
        matches = [c for c in df.columns if c.strip().lower() == src]
        if matches:
            df[dst] = df[matches[0]]

    return df


def parse_list_like(value):
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return [str(v).strip().lower() for v in value]

    text = str(value).strip()

    if text == "":
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v).strip().lower() for v in parsed]
    except Exception:
        pass

    return [v.strip().lower() for v in text.split(",") if v.strip()]


def add_app_engineered_features(df):
    df = df.copy()

    category_col = None
    for possible in ["categories", "category_list"]:
        matches = [c for c in df.columns if c.strip().lower() == possible]
        if matches:
            category_col = matches[0]
            break

    if category_col:
        cats = df[category_col].apply(parse_list_like)
    else:
        cats = pd.Series([[] for _ in range(len(df))], index=df.index)

    df["num_categories"] = cats.apply(len)

    def has_any(cat_list, keywords):
        return int(any(k in cat_list for k in keywords))

    df["has_nightlife"] = cats.apply(lambda x: has_any(x, ["nightlife", "bars", "cocktail bars", "pubs", "lounges"]))
    df["has_fast_food"] = cats.apply(lambda x: has_any(x, ["fast food", "burgers", "sandwiches", "pizza"]))
    df["has_breakfast"] = cats.apply(lambda x: has_any(x, ["breakfast", "brunch", "cafes", "coffee"]))
    df["has_seafood"] = cats.apply(lambda x: has_any(x, ["seafood", "sushi"]))
    df["has_asian"] = cats.apply(lambda x: has_any(x, ["asian fusion", "chinese", "japanese", "thai", "vietnamese", "korean"]))
    df["has_mexican"] = cats.apply(lambda x: has_any(x, ["mexican", "tacos", "tex-mex"]))
    df["has_italian"] = cats.apply(lambda x: has_any(x, ["italian", "pizza", "pasta"]))

    defaults = {
        "RestaurantsPriceRange2": 2,
        "NoiseLevel": 2,
        "parking_options_count": 0,
        "meal_options_count": 0,
        "ambience_score": 0,
        "music_options_count": 0,
        "best_nights_count": 0,
        "is_full_service": 1,
        "is_bar_style": 0,
        "is_family_friendly": 1,
        "is_takeout_friendly": 1,
        "is_date_spot": 0,
        "hours_open_per_week": 60,
        "days_open": 7,
        "open_late": 0,
        "open_early": 1,
        "weekend_hours": 20,
        "nearest_dist": 0.5,
        "num_neighbors_1km": 0,
        "same_cuisine_neighbors": 0,
        "avg_neighbor_rating": 3.5,
        "location_cluster": 0,
        "customer_activity_score": 100,
    }

    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["hours_open_per_week"] = pd.to_numeric(df["hours_open_per_week"], errors="coerce").fillna(60)
    df["days_open"] = pd.to_numeric(df["days_open"], errors="coerce").replace(0, np.nan).fillna(7)
    df["avg_hours_per_day"] = df["hours_open_per_week"] / df["days_open"]

    if "business_stars" not in df.columns and "stars" in df.columns:
        df["business_stars"] = df["stars"]

    if "stars" not in df.columns and "business_stars" in df.columns:
        df["stars"] = df["business_stars"]

    return df


def prepare_xgb_features(candidate_df):
    df = clean_candidate_columns(candidate_df)
    df = add_app_engineered_features(df)

    base_cols = profile["xgb_numeric_cols"] + profile["xgb_categorical_cols"]
    work = pd.DataFrame(index=df.index)

    for col in base_cols:
        if col in profile["xgb_high_missing"]:
            continue

        if col in df.columns:
            work[col] = df[col]
        else:
            if col in profile["xgb_numeric_cols"]:
                work[col] = profile["xgb_medians"].get(col, 0)
            else:
                work[col] = profile["xgb_modes"].get(col, "Unknown")

    for col in profile["xgb_numeric_cols"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(profile["xgb_medians"].get(col, 0))

    for col in profile["xgb_categorical_cols"]:
        if col in work.columns:
            work[col] = work[col].fillna("Unknown").astype(str)

    encoded = pd.get_dummies(
        work,
        columns=[c for c in profile["xgb_categorical_cols"] if c in work.columns],
        drop_first=True
    )

    for col in xgb_features:
        if col not in encoded.columns:
            encoded[col] = 0

    encoded = encoded[xgb_features]

    for col in encoded.columns:
        encoded[col] = pd.to_numeric(encoded[col], errors="coerce").fillna(0)

    return encoded


def prepare_tabnet_features(candidate_df):
    df = clean_candidate_columns(candidate_df)
    df = add_app_engineered_features(df)

    work = pd.DataFrame(index=df.index)

    for col in tabnet_features:
        if col in df.columns:
            work[col] = df[col]
        else:
            work[col] = profile["tab_medians"].get(col, 0)

    for col in work.columns:
        work[col] = pd.to_numeric(work[col], errors="coerce").fillna(profile["tab_medians"].get(col, 0))

    scaled = tabnet_scaler.transform(work)
    return scaled, work


def get_gnn_context(candidate_df):
    df = clean_candidate_columns(candidate_df)

    lat_col = None
    lon_col = None

    for c in df.columns:
        if c.strip().lower() == "latitude":
            lat_col = c
        if c.strip().lower() == "longitude":
            lon_col = c

    emb_numeric = gnn_embeddings.select_dtypes(include=[np.number]).copy()

    if emb_numeric.empty:
        return pd.DataFrame({"gnn_context_score": [0.0] * len(df)}), None

    if profile["coords"] is None or lat_col is None or lon_col is None:
        avg_emb = emb_numeric.mean(numeric_only=True)
        out = pd.DataFrame(
            [avg_emb.values] * len(df),
            columns=[f"nearest_{c}" for c in avg_emb.index]
        )
        out["gnn_context_score"] = float(avg_emb.mean())
        return out, "average"

    coords_train = profile["coords"].copy()

    n = min(len(coords_train), len(emb_numeric))
    coords_train = coords_train.iloc[:n].reset_index(drop=True)
    emb_numeric = emb_numeric.iloc[:n].reset_index(drop=True)

    train_rad = np.radians(coords_train[["latitude", "longitude"]].values)
    tree = BallTree(train_rad, metric="haversine")

    cand = df[[lat_col, lon_col]].copy()
    cand.columns = ["latitude", "longitude"]
    cand["latitude"] = pd.to_numeric(cand["latitude"], errors="coerce")
    cand["longitude"] = pd.to_numeric(cand["longitude"], errors="coerce")

    cand["latitude"] = cand["latitude"].fillna(coords_train["latitude"].median())
    cand["longitude"] = cand["longitude"].fillna(coords_train["longitude"].median())
    cand["latitude"] = cand["latitude"].clip(-90, 90)
    cand["longitude"] = cand["longitude"].clip(-180, 180)

    cand_rad = np.radians(cand[["latitude", "longitude"]].values)

    dist_rad, idx = tree.query(cand_rad, k=1)
    nearest_idx = idx.flatten()
    dist_km = dist_rad.flatten() * 6371.0

    nearest_emb = emb_numeric.iloc[nearest_idx].reset_index(drop=True)
    nearest_emb.columns = [f"nearest_{c}" for c in nearest_emb.columns]

    raw_score = nearest_emb.mean(axis=1)

    min_v = raw_score.min()
    max_v = raw_score.max()

    if max_v > min_v:
        context_score = (raw_score - min_v) / (max_v - min_v)
    else:
        context_score = pd.Series([0.5] * len(raw_score))

    nearest_emb["nearest_gnn_distance_km"] = dist_km
    nearest_emb["gnn_context_score"] = context_score.values

    return nearest_emb, "nearest"


def label_score(prob):
    if prob >= 0.75:
        return "High Potential"
    elif prob >= 0.50:
        return "Moderate Potential"
    else:
        return "Low Potential"


def predict_success(candidate_df, xgb_weight, tabnet_weight, gnn_weight):
    xgb_X = prepare_xgb_features(candidate_df)
    tabnet_X_scaled, tabnet_X_unscaled = prepare_tabnet_features(candidate_df)
    gnn_context, gnn_method = get_gnn_context(candidate_df)

    xgb_proba = xgb_model.predict_proba(xgb_X)[:, 1]
    tabnet_proba = tabnet_model.predict_proba(tabnet_X_scaled)[:, 1]
    gnn_score = gnn_context["gnn_context_score"].values

    total_weight = xgb_weight + tabnet_weight + gnn_weight

    if total_weight <= 0:
        xgb_weight, tabnet_weight, gnn_weight = 0.5, 0.5, 0.0
        total_weight = 1.0

    final_probability = (
        (xgb_weight * xgb_proba)
        + (tabnet_weight * tabnet_proba)
        + (gnn_weight * gnn_score)
    ) / total_weight

    final_probability = np.clip(final_probability, 0, 1)

    results = candidate_df.copy()
    results["xgb_probability"] = xgb_proba
    results["tabnet_probability"] = tabnet_proba
    results["gnn_context_score"] = gnn_score
    results["success_probability"] = final_probability
    results["success_score"] = (final_probability * 100).round(2)
    results["recommendation_level"] = results["success_probability"].apply(label_score)

    if "nearest_gnn_distance_km" in gnn_context.columns:
        results["nearest_gnn_distance_km"] = gnn_context["nearest_gnn_distance_km"].values

    results = results.sort_values("success_probability", ascending=False).reset_index(drop=True)
    results["rank"] = results.index + 1

    return results, xgb_X, tabnet_X_unscaled, gnn_method


def create_manual_candidate():
    st.subheader("Candidate Restaurant Location")

    with st.expander("Enter location details", expanded=True):
        col1, col2, col3 = st.columns(3)

        with col1:
            name = st.text_input("Restaurant Concept Name", "New Restaurant")
            city = st.text_input("City", "Tampa")
            state = st.text_input("State", "FL")
            postal_code = st.text_input("Postal Code", "33602")
            latitude = st.number_input("Latitude", value=27.9506, format="%.6f")
            longitude = st.number_input("Longitude", value=-82.4572, format="%.6f")

        with col2:
            expected_rating = st.slider("Expected Rating", 1.0, 5.0, 4.0)
            review_count = st.number_input("Expected Review Count", value=100, min_value=0)
            price_range = st.slider("Price Range", 1, 4, 2)
            noise_level = st.slider("Noise Level Encoded", 0, 4, 2)
            categories = st.text_input("Categories", "Restaurants, Breakfast & Brunch")

        with col3:
            competition_count = st.number_input("Nearby Competition Count", value=10, min_value=0)
            avg_competitor_rating = st.slider("Average Competitor Rating", 1.0, 5.0, 3.8)
            nearest_dist = st.number_input("Nearest Competitor Distance km", value=0.5, min_value=0.0)
            same_cuisine_neighbors = st.number_input("Same Cuisine Neighbors", value=3, min_value=0)
            customer_activity = st.number_input("Customer Activity Score", value=100, min_value=0)

    with st.expander("Optional operational details", expanded=False):
        col4, col5, col6 = st.columns(3)

        with col4:
            parking_options_count = st.number_input("Parking Options Count", value=1, min_value=0)
            meal_options_count = st.number_input("Meal Options Count", value=2, min_value=0)
            ambience_score = st.number_input("Ambience Score", value=1, min_value=0)

        with col5:
            hours_open_per_week = st.number_input("Hours Open Per Week", value=60, min_value=0)
            days_open = st.number_input("Days Open", value=7, min_value=1, max_value=7)
            weekend_hours = st.number_input("Weekend Hours", value=20, min_value=0)

        with col6:
            is_full_service = st.checkbox("Full Service", value=True)
            is_takeout_friendly = st.checkbox("Takeout Friendly", value=True)
            is_family_friendly = st.checkbox("Family Friendly", value=True)
            is_bar_style = st.checkbox("Bar Style", value=False)
            is_date_spot = st.checkbox("Date Spot", value=False)

    candidate = {
        "name": name,
        "city": city,
        "state": state,
        "postal_code": postal_code,
        "latitude": latitude,
        "longitude": longitude,
        "stars": expected_rating,
        "business_stars": expected_rating,
        "review_count": review_count,
        "RestaurantsPriceRange2": price_range,
        "NoiseLevel": noise_level,
        "categories": categories,
        "num_neighbors_1km": competition_count,
        "nearby_competition_count": competition_count,
        "avg_neighbor_rating": avg_competitor_rating,
        "avg_competitor_rating": avg_competitor_rating,
        "nearest_dist": nearest_dist,
        "same_cuisine_neighbors": same_cuisine_neighbors,
        "customer_activity_score": customer_activity,
        "parking_options_count": parking_options_count,
        "meal_options_count": meal_options_count,
        "ambience_score": ambience_score,
        "hours_open_per_week": hours_open_per_week,
        "days_open": days_open,
        "weekend_hours": weekend_hours,
        "is_full_service": int(is_full_service),
        "is_takeout_friendly": int(is_takeout_friendly),
        "is_family_friendly": int(is_family_friendly),
        "is_bar_style": int(is_bar_style),
        "is_date_spot": int(is_date_spot),
    }

    return pd.DataFrame([candidate])


st.sidebar.title("Model Settings")

st.sidebar.write("""
This  combines XGBoost,
TabNet, and GNN spatial context directly.
""")

xgb_weight = st.sidebar.slider("XGBoost Weight", 0.0, 1.0, 0.45, 0.05)
tabnet_weight = st.sidebar.slider("TabNet Weight", 0.0, 1.0, 0.45, 0.05)
gnn_weight = st.sidebar.slider("GNN Context Weight", 0.0, 1.0, 0.10, 0.05)

st.sidebar.caption("Recommended default: 0.45 XGBoost, 0.45 TabNet, 0.10 GNN context.")


st.subheader("Choose Input Method")

input_method = st.radio(
    "How do you want to test locations?",
    ["Enter one location manually", "Upload candidate locations CSV"],
    horizontal=True
)

candidate_df = None

if input_method == "Enter one location manually":
    candidate_df = create_manual_candidate()
else:
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file is not None:
        candidate_df = pd.read_csv(uploaded_file)


if candidate_df is not None:
    st.subheader("Candidate Input Preview")
    st.dataframe(candidate_df, use_container_width=True)


    if st.button("Predict Restaurant Success"):
        ranked_df, xgb_matrix, tabnet_matrix, gnn_method = predict_success(
            candidate_df,
            xgb_weight=xgb_weight,
            tabnet_weight=tabnet_weight,
            gnn_weight=gnn_weight,
        )

        st.success("Prediction complete.")

        best = ranked_df.iloc[0]
        score = best["success_score"]
        level = best["recommendation_level"]

        st.subheader("Location Success Summary")

        st.markdown(
            f"""
            <div style="
                padding: 25px;
                border-radius: 16px;
                background-color: #1f2937;
                text-align: center;
                margin-bottom: 20px;">
                <h2 style="color:white;">Predicted Success Score</h2>
                <h1 style="color:#22c55e; font-size:58px;">{score:.2f}%</h1>
                <h3 style="color:white;">{level}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric("Locations Tested", len(ranked_df))
        col2.metric("Best Success Score", f"{ranked_df['success_score'].max():.2f}%")
        col3.metric("Average Candidate Score", f"{ranked_df['success_score'].mean():.2f}%")
        col4.metric("GNN Method", str(gnn_method).title())

        st.subheader("Model Probability Comparison")

        model_compare = pd.DataFrame({
            "Model Component": ["XGBoost", "TabNet", "GNN Context", "Final Score"],
            "Score": [
                float(best["xgb_probability"]),
                float(best["tabnet_probability"]),
                float(best["gnn_context_score"]),
                float(best["success_probability"]),
            ],
        })

        st.bar_chart(model_compare.set_index("Model Component"))

        old_locations = training_df.drop(columns=["target"], errors="ignore").copy()

        if len(old_locations) > 0:
            st.subheader("Comparison to Existing Restaurant Locations")

            sample_size = min(1000, len(old_locations))
            sample_old = old_locations.sample(sample_size, random_state=42)

            old_ranked, _, _, _ = predict_success(
                sample_old,
                xgb_weight=xgb_weight,
                tabnet_weight=tabnet_weight,
                gnn_weight=gnn_weight,
            )

            candidate_score = ranked_df["success_score"].iloc[0]
            old_avg = old_ranked["success_score"].mean()
            old_top = old_ranked["success_score"].max()
            percentile = (old_ranked["success_score"] < candidate_score).mean() * 100

            c1, c2, c3 = st.columns(3)

            c1.metric("Candidate Score", f"{candidate_score:.2f}%")
            c2.metric("Average Existing Score", f"{old_avg:.2f}%")
            c3.metric("Better Than", f"{percentile:.1f}% of sampled locations")

            comparison_df = pd.DataFrame({
                "Group": [
                    "Candidate Location",
                    "Average Existing Location",
                    "Best Existing Location",
                ],
                "Success Score": [
                    candidate_score,
                    old_avg,
                    old_top,
                ],
            })

            st.bar_chart(comparison_df.set_index("Group"))
            st.caption(f"Comparison is based on a sample of {sample_size} existing locations.")

        st.subheader("Ranked Candidate Locations")

        display_cols = [
            c for c in [
                "rank",
                "name",
                "city",
                "state",
                "postal_code",
                "latitude",
                "longitude",
                "success_score",
                "recommendation_level",
                "xgb_probability",
                "tabnet_probability",
                "gnn_context_score",
                "nearest_gnn_distance_km",
            ]
            if c in ranked_df.columns
        ]

        st.dataframe(ranked_df[display_cols], use_container_width=True)

        if "latitude" in ranked_df.columns and "longitude" in ranked_df.columns:
            map_df = ranked_df[["latitude", "longitude", "success_score"]].copy()
            map_df = map_df.rename(columns={"latitude": "lat", "longitude": "lon"})

            map_df["lat"] = pd.to_numeric(map_df["lat"], errors="coerce")
            map_df["lon"] = pd.to_numeric(map_df["lon"], errors="coerce")

            map_df = map_df.dropna()
            map_df = map_df[
                map_df["lat"].between(-90, 90)
                & map_df["lon"].between(-180, 180)
            ]

            if len(map_df) > 0:
                st.subheader("Candidate Location Map")
                st.map(map_df)

        st.subheader("Interpretation")

        st.write(
            f"The best candidate location has a predicted success score of "
            f"**{best['success_score']:.2f}%**, classified as "
            f"**{best['recommendation_level']}**."
        )

        st.write("""
        The final score is calculated from the weighted combination of XGBoost,
        TabNet, and GNN context scores. XGBoost uses the reconstructed one-hot
        encoded training feature structure. TabNet uses the reconstructed numeric
        feature structure and saved scaler. The GNN component uses the nearest
        existing location embedding as a spatial proxy for the candidate.
        """)

        with st.expander("Debug: model input shapes"):
            st.write("XGBoost matrix shape:", xgb_matrix.shape)
            st.write("TabNet matrix shape:", tabnet_matrix.shape)
            st.write("Expected XGBoost features:", len(xgb_features))
            st.write("Expected TabNet features:", len(tabnet_features))

        csv = ranked_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Results",
            data=csv,
            file_name="restaurant_success_predictions.csv",
            mime="text/csv",
        )

else:
    st.info("Enter a location or upload a CSV to begin.")