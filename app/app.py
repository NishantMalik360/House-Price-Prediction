"""
=========================================================
AI House Price Prediction System
Streamlit Application
=========================================================
"""

import streamlit as st

from src.predict import (
    predict_house,
    get_locations,
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="AI House Price Prediction",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("🏠 House Price Prediction")

st.sidebar.info(
    """
AI-powered Bengaluru House Price Prediction

Developed using:
- Python
- Scikit-Learn
- Gradient Boosting
- Streamlit
"""
)

# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🏠 AI House Price Prediction")

st.write(
    "Predict Bengaluru house prices using Machine Learning."
)

st.divider()

# --------------------------------------------------
# Input Section
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    area_type = st.selectbox(
        "Area Type",
        [
            "Super built-up Area",
            "Built-up Area",
            "Plot Area",
            "Carpet Area",
        ],
    )

    bhk = st.number_input(
        "BHK",
        min_value=1,
        value=2,
    )

    bath = st.number_input(
        "Bathrooms",
        min_value=1,
        value=2,
    )

with col2:

    total_sqft = st.number_input(
        "Total Sqft",
        min_value=100.0,
        value=1200.0,
    )

    balcony = st.number_input(
        "Balconies",
        min_value=0,
        value=1,
    )

    location = st.selectbox(
        "Location",
        get_locations(),
    )

st.divider()

# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("🔍 Predict Price", use_container_width=True):

    user_input = {
        "total_sqft": total_sqft,
        "bath": bath,
        "balcony": balcony,
        "bhk": bhk,
        "area_type": area_type,
        "location": location,
    }

    try:

        with st.spinner("Predicting house price..."):

            prediction = predict_house(user_input)

        st.metric(
            label="🏡 Predicted House Price",
            value=f"₹ {prediction:.2f} Lakhs",
        )

        st.success("Prediction completed successfully!")

    except Exception as e:

        st.error(f"Error: {e}")

# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "AI House Price Prediction System | Built with Streamlit & Scikit-Learn"
)