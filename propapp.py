import json
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px


st.set_page_config(page_title="Property Price Predictor", page_icon="🏠", layout="wide")
# Step 2 — Path constants
MODEL_PATH = "models/best_model.joblib"
META_PATH = "models/meta.json"
COMPARISON_PATH = "outputs/model_comparison.csv"

# Step 3 = Cache Loader
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_meta():
    with open(META_PATH) as f:
        return json.load(f)


@st.cache_data
def load_comparison():
    return pd.read_csv(COMPARISON_PATH)

# Step 4 - Load model and metadata
model = load_model()
meta = load_meta()

st.title("🏠 Property Price Predictor")
st.caption(
    f"Predicts residential property price (in Lacs ₹) for Indian listings. "
    f"Best model: **{meta['best_model']}** (Test R² = {meta['test_r2']:.3f})"
)

tab_predict, tab_compare = st.tabs(["🔮 Predict Price", "📊 Model Comparison"])

# ----------------------------------------------------------------------
# Step 5: TAB 1: Prediction Form
# ----------------------------------------------------------------------
with tab_predict:
    st.subheader("Enter property details")

    col1, col2, col3 = st.columns(3)

    with col1:
        posted_by = st.selectbox("Posted by", meta["posted_by_options"])
        bhk_or_rk = st.selectbox("BHK or RK", meta["bhk_or_rk_options"])
        bhk_no = st.number_input("Number of bedrooms (BHK)", min_value=1, max_value=10, value=2, step=1)

    with col2:
        city = st.selectbox("City", meta["cities_used"], index=meta["cities_used"].index("Mumbai") if "Mumbai" in meta["cities_used"] else 0)
        sqft = st.slider(
            "Area (sq. ft.)",
            min_value=int(meta["sqft_min"]), max_value=int(meta["sqft_max"]),
            value=1000, step=10,
        )
        resale = st.radio("Resale property?", ["Yes", "No"], horizontal=True)

    with col3:
        under_construction = st.radio("Under construction?", ["No", "Yes"], horizontal=True)
        rera = st.radio("RERA registered?", ["Yes", "No"], horizontal=True)
        ready_to_move = st.radio("Ready to move in?", ["Yes", "No"], horizontal=True)

 # Step 6 — Coordinate auto-fill (optional, collapsible section)
 # Auto-fill lat/long from the selected city's average, with manual override
    coords = meta["city_coords"].get(city, {"LATITUDE": 20.5937, "LONGITUDE": 78.9629})
    with st.expander("Advanced: fine-tune exact coordinates (optional)"):
        lat = st.number_input("Latitude", value=float(coords["LATITUDE"]), format="%.4f")
        lon = st.number_input("Longitude", value=float(coords["LONGITUDE"]), format="%.4f")

 # Step 7 — Predict button and Prediction Logic

        st.divider()

    if st.button("Predict Price", type="primary", use_container_width=True):
        input_df = pd.DataFrame([{
            "POSTED_BY": posted_by,
            "UNDER_CONSTRUCTION": 1 if under_construction == "Yes" else 0,
            "RERA": 1 if rera == "Yes" else 0,
            "BHK_NO.": bhk_no,
            "BHK_OR_RK": bhk_or_rk,
            "SQUARE_FT": sqft,
            "READY_TO_MOVE": 1 if ready_to_move == "Yes" else 0,
            "RESALE": 1 if resale == "Yes" else 0,
            "LONGITUDE": lon,
            "LATITUDE": lat,
            "CITY": city,
        }])

        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 0)  # guard against tiny negative predictions

        st.success(f"### Estimated Price: ₹ {prediction:,.2f} Lacs (~₹ {prediction * 100000:,.0f})")

        low, high = prediction * 0.85, prediction * 1.15
        st.caption(f"Typical range for similar listings: ₹ {low:,.1f} – {high:,.1f} Lacs")


# ----------------------------------------------------------------------
# Step 8: TAB 2: Model comparison
# ----------------------------------------------------------------------
with tab_compare:
    st.subheader("Model performance comparison")
    comp_df = load_comparison()
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        comp_df.sort_values("Test_R2"),
        x="Test_R2", y="Model", orientation="h",
        title="Test R² by model", text="Test_R2",
        color="Test_R2", color_continuous_scale="Tealgrn",
    )
    fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(
        comp_df.sort_values("RMSE_Lacs"),
        x="RMSE_Lacs", y="Model", orientation="h",
        title="RMSE (Lacs) by model — lower is better", text="RMSE_Lacs",
        color="RMSE_Lacs", color_continuous_scale="Reds",
    )
    fig2.update_traces(texttemplate="%{text:.1f}", textposition="outside")
    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "Metrics computed on a held-out 20% test split, with 5-fold cross-validation "
        "R² shown as a stability check. Target values above the 99th percentile and below "
        "the 1st percentile were clipped during preprocessing to limit the influence of extreme outliers."
    )