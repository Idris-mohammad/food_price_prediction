import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

# Load data and model
df = pd.read_csv('data/cleaned_food_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])

model = joblib.load('models/food_price_model.pkl')

st.title("🛒 Food Price Prediction App")
st.markdown("Predict prices for essential food items like rice, wheat, sugar, onion, milk, and oil.")

# User Inputs
selected_item = st.selectbox("Select an item", df['Item'].unique())
selected_date = st.date_input("Select a future date for prediction")

# Prepare input for prediction
input_day = selected_date.day
input_month = selected_date.month
input_year = selected_date.year

# ✅ Updated function: combines historical and predicted point into a single line
def plot_price_trend(item, cutoff_date=None, predicted_price=None):
    item_df = df[df["Item"] == item].copy()
    item_df = item_df.sort_values("Date")  # Ensure correct time order

    if cutoff_date and predicted_price is not None:
        # Add predicted row
        predicted_row = pd.DataFrame({
            "Date": [pd.to_datetime(cutoff_date)],
            "Price": [predicted_price]
        })
        combined_df = pd.concat([item_df[["Date", "Price"]], predicted_row], ignore_index=True)
        combined_df = combined_df.sort_values("Date")
    else:
        combined_df = item_df[["Date", "Price"]]

    # Plot as single line (dots will now connect)
    fig, ax = plt.subplots()
    ax.plot(combined_df["Date"], combined_df["Price"], marker='o', linestyle='-', label='Price Trend')
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.set_title(f"{item} Price Over Time")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# Show the trend before prediction
plot_price_trend(selected_item)

# Predict Button
if st.button("Predict Price"):
    input_df = pd.DataFrame({
        'Item': [selected_item],
        'Day': [input_day],
        'Month': [input_month],
        'Year': [input_year]
    })

    # Predict
    predicted_price = model.predict(input_df)[0]
    st.success(f"Predicted price of **{selected_item}** on {selected_date.strftime('%Y-%m-%d')} is ₹{predicted_price:.2f}")

    # Show updated graph with predicted value included in the line
    plot_price_trend(selected_item, cutoff_date=selected_date, predicted_price=predicted_price)
