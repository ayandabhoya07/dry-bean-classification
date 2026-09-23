import streamlit as st
import joblib
import pandas as pd


# ============================================================
# LOAD TRAINED MODEL AND LABEL ENCODER
# ============================================================

model = joblib.load("dry_bean_xgboost.pkl")
label_encoder = joblib.load("label_encoder.pkl")


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🫘",
    layout="centered"
)


# ============================================================
# TITLE
# ============================================================

st.title("🫘 Dry Bean Classification")

st.write(
    "Enter the morphological measurements of a dry bean "
    "to predict its class using the trained XGBoost model."
)

st.divider()


# ============================================================
# INPUT SECTION
# ============================================================

st.header("Bean Measurements")

st.write(
    "Enter the morphological measurements of the bean."
)

col1, col2 = st.columns(2)


# ============================================================
# LEFT COLUMN
# ============================================================

with col1:

    area = st.number_input(
        "Area",
        min_value=20420.00000,
        max_value=254616.00000,
        value=50000.00000,
        step=0.00001,
        format="%.5f"
    )

    major_axis_length = st.number_input(
        "Major Axis Length",
        min_value=183.601165,
        max_value=738.860153,
        value=300.00000,
        step=0.00001,
        format="%.5f"
    )

    minor_axis_length = st.number_input(
        "Minor Axis Length",
        min_value=122.512653,
        max_value=460.198497,
        value=200.00000,
        step=0.00001,
        format="%.5f"
    )

    aspect_ration = st.number_input(
        "Aspect Ration",
        min_value=1.024868,
        max_value=2.430306,
        value=1.50000,
        step=0.00001,
        format="%.5f"
    )

    extent = st.number_input(
        "Extent",
        min_value=0.555315,
        max_value=0.866195,
        value=0.75000,
        step=0.00001,
        format="%.5f"
    )

    solidity = st.number_input(
        "Solidity",
        min_value=0.919246,
        max_value=0.994677,
        value=0.98000,
        step=0.00001,
        format="%.5f"
    )


# ============================================================
# RIGHT COLUMN
# ============================================================

with col2:

    roundness = st.number_input(
        "Roundness",
        min_value=0.489618,
        max_value=0.990685,
        value=0.80000,
        step=0.00001,
        format="%.5f"
    )

    compactness = st.number_input(
        "Compactness",
        min_value=0.640577,
        max_value=0.987303,
        value=0.70000,
        step=0.00001,
        format="%.5f"
    )

    shape_factor1 = st.number_input(
        "Shape Factor 1",
        min_value=0.002778,
        max_value=0.010451,
        value=0.00600,
        step=0.00001,
        format="%.5f"
    )

    shape_factor2 = st.number_input(
        "Shape Factor 2",
        min_value=0.000564,
        max_value=0.003665,
        value=0.00200,
        step=0.00001,
        format="%.5f"
    )

    shape_factor4 = st.number_input(
        "Shape Factor 4",
        min_value=0.947687,
        max_value=0.999733,
        value=0.99000,
        step=0.00001,
        format="%.5f"
    )

# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

new_bean = pd.DataFrame([{

    "Area": area,

    "MajorAxisLength": major_axis_length,

    "MinorAxisLength": minor_axis_length,

    "AspectRation": aspect_ration,

    "Extent": extent,

    "Solidity": solidity,

    "roundness": roundness,

    "Compactness": compactness,

    "ShapeFactor1": shape_factor1,

    "ShapeFactor2": shape_factor2,

    "ShapeFactor4": shape_factor4

}])


# ============================================================
# PREDICTION BUTTON
# ============================================================

st.divider()

if st.button(
    "🔍 Predict Bean Class",
    use_container_width=True
):

    # --------------------------------------------------------
    # Make prediction
    # --------------------------------------------------------

    prediction = model.predict(new_bean)

    # Convert encoded number back to original class name
    predicted_class = label_encoder.inverse_transform(
        prediction
    )[0]


    # --------------------------------------------------------
    # Get prediction probabilities
    # --------------------------------------------------------

    probabilities = model.predict_proba(new_bean)[0]

    class_names = label_encoder.classes_

    probability_df = pd.DataFrame({

        "Bean Class": class_names,

        "Probability": probabilities

    })

    probability_df = probability_df.sort_values(
        by="Probability",
        ascending=False
    )


    # --------------------------------------------------------
    # Display prediction
    # --------------------------------------------------------

    st.success(
        f"🫘 Predicted Bean Class: **{predicted_class}**"
    )


    # --------------------------------------------------------
    # Display confidence
    # --------------------------------------------------------

    confidence = probability_df.iloc[0]["Probability"] * 100

    st.metric(
        "Prediction Probability",
        f"{confidence:.2f}%"
    )


    # --------------------------------------------------------
    # Display probability chart
    # --------------------------------------------------------

    st.subheader("Prediction Probabilities")

    probability_display = probability_df.copy()

    probability_display["Probability"] = (
        probability_display["Probability"] * 100
    ).round(2)

    probability_display = probability_display.set_index(
        "Bean Class"
    )

    st.bar_chart(
        probability_display["Probability"]
    )


    # --------------------------------------------------------
    # Display probability table
    # --------------------------------------------------------

    st.subheader("Detailed Results")

    probability_display = probability_display.rename(
        columns={
            "Probability": "Probability (%)"
        }
    )

    st.dataframe(
        probability_display,
        use_container_width=True
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("ℹ️ About this model"):

    st.write(
        """
        **Model:** XGBoost Classifier

        **Task:** Multiclass Dry Bean Classification

        **Number of classes:** 7

        **Input features:** 11

        **Model selection:** Multiple classification
        algorithms were evaluated before selecting XGBoost
        for further optimization.

        **Hyperparameter optimization:** RandomizedSearchCV
        with 5-fold Stratified Cross-Validation.

        **Final test accuracy:** 92.32%

        **Final test Macro F1-score:** 93.57%
        """
    )