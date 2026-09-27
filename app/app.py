import joblib
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("SuperKart Sales Prediction")
st.write("Enter product and store details to estimate sales total.")

product_weight = st.number_input("Product Weight", min_value=0.0, value=12.0)
product_sugar = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
product_area = st.number_input("Product Allocated Area", min_value=0.0, value=0.05)
product_type = st.selectbox("Product Type", ["Frozen Foods", "Dairy", "Canned", "Baking Goods", "Health and Hygiene"])
product_mrp = st.number_input("Product MRP", min_value=0.0, value=150.0)
store_est_year = st.number_input("Store Establishment Year", min_value=1900, max_value=2050, value=2000)
store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
city_type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Supermarket Type1", "Supermarket Type2", "Departmental Store", "Food Mart"])

if st.button("Predict Sales"):
    if not MODEL_PATH.exists():
        st.error("Model file not found. Please train the model first by running `python src/train_model.py`.")
    else:
        model = load_model()
        sample = pd.DataFrame(
            [{
                "Product_Weight": product_weight,
                "Product_Sugar_Content": product_sugar,
                "Product_Allocated_Area": product_area,
                "Product_Type": product_type,
                "Product_MRP": product_mrp,
                "Store_Establishment_Year": store_est_year,
                "Store_Size": store_size,
                "Store_Location_City_Type": city_type,
                "Store_Type": store_type,
            }]
        )

        sample = pd.get_dummies(sample)
        model_columns = model.feature_names_in_
        missing_cols = set(model_columns) - set(sample.columns)
        for col in missing_cols:
            sample[col] = 0
        sample = sample[model_columns]

        prediction = model.predict(sample)[0]
        st.success(f"Estimated total sales: ₹{prediction:,.2f}")
