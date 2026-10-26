"""app.py

Streamlit Web Application for CrowdCast.

Provides an intuitive, beginner-friendly interface for passengers or transit operators
to forecast bus crowding at an upcoming stop BEFORE the bus arrives.

Features:
- Route and stop selection (pre-populated from collected data or custom input)
- Safe user input validation (passenger counts, practical capacity, times)
- Comparison between Primary model (Decision Tree) and Comparison model (KNN)
- Multi-class predicted probabilities / confidence scores
- Summary of input scenario and plain-English explanation
- Academic disclaimer emphasizing empirical dataset limits (no fake GPS or live feeds)
"""

from datetime import datetime, time
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="CrowdCast - Bus Crowding Predictor",
    page_icon="🚌",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paths to artifacts
MODELS_DIR = Path("models")
PROCESSED_DATA_PATH = Path("data/processed/crowdcast_clean.csv")
DT_MODEL_PATH = MODELS_DIR / "decision_tree_model.joblib"
KNN_MODEL_PATH = MODELS_DIR / "knn_model.joblib"
LE_PATH = MODELS_DIR / "label_encoder.joblib"
META_PATH = MODELS_DIR / "model_metadata.json"


@st.cache_resource
def load_models_and_metadata():
    """Loads serialised pipelines, label encoder, and feature metadata."""
    if not DT_MODEL_PATH.exists() or not KNN_MODEL_PATH.exists() or not LE_PATH.exists():
        return None, None, None, None

    dt_pipeline = joblib.load(DT_MODEL_PATH)
    knn_pipeline = joblib.load(KNN_MODEL_PATH)
    label_encoder = joblib.load(LE_PATH)

    metadata = {}
    if META_PATH.exists():
        with open(META_PATH, "r") as f:
            metadata = json.load(f)

    return dt_pipeline, knn_pipeline, label_encoder, metadata


@st.cache_data
def load_distinct_routes_and_stops():
    """Loads distinct route IDs and stops from processed data if available."""
    if PROCESSED_DATA_PATH.exists():
        df = pd.read_csv(PROCESSED_DATA_PATH)
        routes = sorted(df["route_id"].dropna().unique().tolist())
        stops = sorted(
            list(set(df["previous_stop"].dropna().unique().tolist() + df["target_stop"].dropna().unique().tolist()))
        )
        return routes, stops
    return (
        ["Route_101", "Route_202", "Route_303", "Route_404"],
        ["Central Station", "Market Square", "Tech Park", "University East", "Terminal 1", "Civic Center"],
    )


def build_feature_dataframe(
    route_id: str,
    prev_stop: str,
    target_stop: str,
    day_week: str,
    pred_time_obj: time,
    earlier_passengers: float,
    capacity: float,
    travel_mins: float,
    is_peak: int,
) -> pd.DataFrame:
    """Constructs the exact pandas DataFrame required by the trained Pipeline."""
    hour = pred_time_obj.hour
    minute = pred_time_obj.minute
    decimal_time = hour + (minute / 60.0)
    earlier_ratio = earlier_passengers / capacity if capacity > 0 else 0.0

    row = {
        "passengers_before_target": float(earlier_passengers),
        "practical_capacity": float(capacity),
        "minutes_since_previous_count": float(travel_mins),
        "is_peak_period": int(is_peak),
        "earlier_occupancy_ratio": float(earlier_ratio),
        "prediction_hour": int(hour),
        "prediction_minute": int(minute),
        "prediction_time_decimal": float(decimal_time),
        "route_id": str(route_id),
        "previous_stop": str(prev_stop),
        "target_stop": str(target_stop),
        "day_of_week": str(day_week),
    }
    return pd.DataFrame([row])


def main():
    # Header Banner
    st.title("🚌 CrowdCast: Public Transport Crowding Prediction")
    st.markdown(
        """
        **Undergraduate ML-I Mini-Project** | Predicting bus passenger crowding at upcoming stops 
        *before arrival* using Decision Tree and K-Nearest Neighbors (KNN).
        """
    )
    st.divider()

    # Load artifacts
    dt_pipeline, knn_pipeline, label_encoder, metadata = load_models_and_metadata()

    if dt_pipeline is None or knn_pipeline is None:
        st.error(
            "⚠️ Trained model files not found in `models/` directory!\n\n"
            "Please run the pipeline first:\n"
            "1. `python src/prepare_data.py`\n"
            "2. `python src/train.py`\n"
            "3. Then refresh this Streamlit page."
        )
        st.stop()

    available_routes, available_stops = load_distinct_routes_and_stops()

    # Sidebar: User Inputs
    st.sidebar.header("📋 Bus Observation Inputs")
    st.sidebar.caption("Provide data observable before reaching the target stop.")

    route_id = st.sidebar.selectbox("Bus Route", available_routes)
    prev_stop = st.sidebar.selectbox("Previous Stop (Current location)", available_stops, index=0)
    
    # Filter target stop list to prevent choosing the exact same stop
    target_stop_candidates = [s for s in available_stops if s != prev_stop]
    if not target_stop_candidates:
        target_stop_candidates = available_stops
    target_stop = st.sidebar.selectbox("Target Stop (Upcoming stop)", target_stop_candidates, index=min(1, len(target_stop_candidates) - 1))

    days_list = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    today_name = datetime.today().strftime("%A")
    default_day_idx = days_list.index(today_name) if today_name in days_list else 0
    day_week = st.sidebar.selectbox("Day of Week", days_list, index=default_day_idx)

    pred_time_input = st.sidebar.time_input("Prediction Time (24-hr clock)", value=datetime.now().time())

    st.sidebar.subheader("Occupancy Parameters")
    earlier_passengers = st.sidebar.number_input(
        "Current Passengers on Bus",
        min_value=0,
        max_value=150,
        value=28,
        step=1,
        help="Observed passenger count at the previous stop.",
    )

    practical_capacity = st.sidebar.number_input(
        "Practical Bus Capacity",
        min_value=10,
        max_value=150,
        value=60,
        step=5,
        help="Verified practical capacity (seated + safe standees).",
    )

    travel_mins = st.sidebar.slider(
        "Estimated Minutes to Target Stop",
        min_value=1.0,
        max_value=60.0,
        value=6.0,
        step=0.5,
        help="Travel time in minutes between the previous and target stop.",
    )

    is_peak_bool = st.sidebar.checkbox(
        "Is Peak Travel Period?",
        value=(pred_time_input.hour in [8, 9, 17, 18] and day_week not in ["Saturday", "Sunday"]),
        help="Check if this falls within typical rush hours.",
    )
    is_peak_int = 1 if is_peak_bool else 0

    # Model Selection toggle
    selected_model_name = st.sidebar.radio(
        "Choose Model Algorithm:",
        options=["Decision Tree (Primary)", "K-Nearest Neighbors (KNN)"],
        index=0,
    )

    # Input Validation
    if earlier_passengers > practical_capacity * 1.5:
        st.warning(
            f"⚠️ Caution: Earlier passenger count ({earlier_passengers}) is unusually high "
            f"relative to practical capacity ({practical_capacity}). Please verify."
        )

    # Prediction Action
    input_df = build_feature_dataframe(
        route_id=route_id,
        prev_stop=prev_stop,
        target_stop=target_stop,
        day_week=day_week,
        pred_time_obj=pred_time_input,
        earlier_passengers=earlier_passengers,
        capacity=practical_capacity,
        travel_mins=travel_mins,
        is_peak=is_peak_int,
    )

    # Main Dashboard Columns
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.subheader("🎯 Upcoming Stop Crowding Forecast")

        active_model = dt_pipeline if "Decision Tree" in selected_model_name else knn_pipeline

        # Predict numeric class index
        pred_idx = active_model.predict(input_df)[0]
        predicted_class = label_encoder.inverse_transform([pred_idx])[0]

        # Predict probabilities
        class_names = list(label_encoder.classes_)
        if hasattr(active_model, "predict_proba"):
            probs = active_model.predict_proba(input_df)[0]
            prob_dict = {c: p for c, p in zip(class_names, probs)}
            confidence = prob_dict[predicted_class] * 100.0
        else:
            prob_dict = {c: (1.0 if c == predicted_class else 0.0) for c in class_names}
            confidence = 100.0

        # Badge styling based on prediction
        badge_colors = {
            "Low": ("#e6f4ea", "#137333", "🟢 LOW CROWDING (< 50% Occupancy)"),
            "Moderate": ("#fef7e0", "#b06000", "🟡 MODERATE CROWDING (50% - 79% Occupancy)"),
            "High": ("#fce8e6", "#c5221f", "🔴 HIGH CROWDING (≥ 80% Occupancy)"),
        }
        bg, text_c, label_text = badge_colors.get(predicted_class, ("#f1f3f4", "#202124", predicted_class))

        st.markdown(
            f"""
            <div style="background-color: {bg}; border-radius: 8px; padding: 18px 24px; border: 2px solid {text_c}; margin-bottom: 20px;">
                <h2 style="color: {text_c}; margin: 0; font-size: 26px;">{label_text}</h2>
                <p style="color: {text_c}; margin-top: 6px; margin-bottom: 0; font-size: 15px;">
                    Model Confidence: <strong>{confidence:.1f}%</strong> | Predicted by: <strong>{selected_model_name}</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("##### Prediction Probabilities Across All Classes:")
        for c in ["Low", "Moderate", "High"]:
            if c in prob_dict:
                p_val = prob_dict[c]
                col_label, col_bar = st.columns([1, 3])
                with col_label:
                    st.write(f"**{c}**")
                with col_bar:
                    st.progress(float(p_val), text=f"{p_val * 100:.1f}%")

        # Explain the prediction
        st.markdown("##### 💡 Diagnostic Explanation:")
        earlier_ratio_pct = (earlier_passengers / practical_capacity) * 100.0
        st.write(
            f"- At **{prev_stop}**, the bus had **{earlier_passengers} passengers** on board "
            f"({earlier_ratio_pct:.1f}% practical occupancy)."
        )
        st.write(
            f"- Prediction interval: **{travel_mins} minutes** before arriving at **{target_stop}** "
            f"({'during' if is_peak_int else 'outside'} peak rush hours)."
        )
        if predicted_class == "High":
            st.info(
                "Notice: Model forecasts significant boarding or high residual passenger load "
                "surpassing 80% capacity at the target stop. Passengers may find seating unavailable."
            )
        elif predicted_class == "Moderate":
            st.info(
                "Notice: Expected occupancy is balanced between 50% and 79%. Seating may be tight, "
                "but ample standing space should remain."
            )
        else:
            st.info(
                "Notice: Occupancy expected to remain below 50%. Boarding passengers have a high "
                "likelihood of obtaining seats."
            )

    with col2:
        st.subheader("🔍 Input Feature Summary")
        summary_data = {
            "Feature Name": [
                "Route ID",
                "Previous Stop",
                "Target Stop",
                "Day of Week",
                "Prediction Time",
                "Earlier Passengers",
                "Practical Capacity",
                "Earlier Occupancy %",
                "Travel Time (Mins)",
                "Peak Period?",
            ],
            "Value": [
                route_id,
                prev_stop,
                target_stop,
                day_week,
                pred_time_input.strftime("%H:%M"),
                f"{earlier_passengers} persons",
                f"{practical_capacity} seats/standees",
                f"{earlier_ratio_pct:.1f}%",
                f"{travel_mins:.1f} mins",
                "Yes (Rush Hour)" if is_peak_int else "No (Off-Peak)",
            ],
        }
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

        if metadata:
            st.markdown("##### 📊 Benchmark Model Metrics (Holdout Trips)")
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric(
                    "Decision Tree Macro F1",
                    f"{metadata.get('dt_metrics', {}).get('macro_f1', 0):.2f}",
                    delta="Primary",
                )
            with col_m2:
                st.metric(
                    "KNN Macro F1",
                    f"{metadata.get('knn_metrics', {}).get('macro_f1', 0):.2f}",
                    delta="Comparison",
                    delta_color="off",
                )

    # Disclaimer Section (Academic Integrity requirement)
    st.divider()
    st.caption(
        "ℹ️ **Project & Academic Disclaimer:** CrowdCast predictions are statistical estimates computed "
        "by machine learning models trained on fixed survey observations. This interface does **not** "
        "connect to fake live GPS hardware, synthetic telemetry, or real-time transit dispatch systems. "
        "Predictive performance is bounded by the routes and times captured during the student survey period."
    )


if __name__ == "__main__":
    main()
