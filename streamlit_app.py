import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="✈️ USA Market Airline Fare Prediction Dashboard", layout="wide")

@st.cache_data
def load_data():
    # Replace with your data file path
    data = pd.read_csv(r"C:\Users\Admin\Desktop\Rerai_Project\MarketFarePredictionData.csv")
    return data

# Load data
data = load_data()

st.title("✈️ USA Market Airline Fare Prediction Dashboard")

# Show raw data toggle
if st.checkbox("Show raw data (first 100 rows)"):
    st.dataframe(data.head(100))

# Define features and target based on available columns (update if needed)
target = 'Average_Fare'
# Example feature list, adjust to your dataset
all_features = [
    'MktCoupons', 'OriginCityMarketID', 'DestCityMarketID', 'OriginAirportID', 'DestAirportID',
    'NonStopMiles', 'RoundTrip', 'Pax', 'CarrierPax', 'Market_share', 'Market_HHI', 'LCC_Comp',
    'Multi_Airport', 'Circuity', 'Slot', 'Non_Stop', 'MktMilesFlown', 'OriginCityMarketID_freq',
    'DestCityMarketID_freq', 'OriginAirportID_freq', 'DestAirportID_freq', 'Carrier_freq',
    'ODPairID_freq', 'DayOfWeek', 'CircuityRatio', 'Carrier_MarketImpact'
]

# Check if all_features exist in data, else remove missing ones
all_features = [f for f in all_features if f in data.columns]

# Prepare features X and target y
X = data[all_features].copy()
y = data[target].copy()

# Convert all features to numeric, coerce errors, fill NaNs with median
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(X.median())

assert not any(X.dtypes == 'object'), "There are still non-numeric columns in features!"

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_models(X_train, y_train):
    xgb = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)

    cat = CatBoostRegressor(verbose=0, random_state=42)
    cat.fit(X_train, y_train)

    return xgb, cat

with st.spinner("Training models..."):
    model_xgb, model_cat = train_models(X_train, y_train)

st.success("Models trained successfully!")

# Model evaluation
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return mse, r2

mse_xgb, r2_xgb = evaluate_model(model_xgb, X_test, y_test)
mse_cat, r2_cat = evaluate_model(model_cat, X_test, y_test)

st.subheader("Model Performance on Test Set")
st.write(f"**XGBoost Regressor**: MSE = {mse_xgb:.2f}, R² = {r2_xgb:.3f}")
st.write(f"**CatBoost Regressor**: MSE = {mse_cat:.2f}, R² = {r2_cat:.3f}")

# Visualization: Fare trends by DayOfWeek
if 'DayOfWeek' in data.columns:
    fare_by_day = data.groupby('DayOfWeek')[target].mean()
    st.subheader("Average Fare by Day of Week")
    fig, ax = plt.subplots()
    fare_by_day.plot(kind='bar', ax=ax)
    ax.set_ylabel('Average Fare ($)')
    ax.set_xlabel('Day of Week (1=Monday)')
    st.pyplot(fig)

# User input for prediction
st.subheader("Predict Average Fare")

input_data = {}
for feature in all_features:
    min_val = float(X[feature].min())
    max_val = float(X[feature].max())
    mean_val = float(X[feature].mean())
    input_data[feature] = st.number_input(
        f"{feature}", min_value=min_val, max_value=max_val, value=mean_val
    )

input_df = pd.DataFrame([input_data])

# Predict button
if st.button("Predict Fare"):
    # Ensure numeric and no missing
    input_df = input_df.apply(pd.to_numeric, errors='coerce').fillna(X.median())
    
    pred_xgb = model_xgb.predict(input_df)[0]
    pred_cat = model_cat.predict(input_df)[0]
    
    st.write(f"**XGBoost Predicted Fare:** ${pred_xgb:.2f}")
    st.write(f"**CatBoost Predicted Fare:** ${pred_cat:.2f}")

st.write("---")
st.caption("Data source: US Department of Transportation Bureau of Transportation Statistics (May 2025)")

