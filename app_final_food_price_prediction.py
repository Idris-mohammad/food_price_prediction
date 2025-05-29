
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime

st.set_page_config(page_title="Food Price Prediction App", page_icon="🛒", layout="wide")

# Sidebar navigation
st.sidebar.title("📂 Navigation")
selection = st.sidebar.radio("Go to", ["🏠 Home", "📈 Data Visualization", "🔮 Price Prediction", "ℹ️ About Us"])

# Load data and model
@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_food_prices.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def load_model():
    return joblib.load("models/food_price_model.pkl")

df = load_data()

# Home Page
if selection == "🏠 Home":
    st.title("🛒 Food Price Prediction App")
    st.markdown("""
    Welcome to the **Food Price Prediction** app!

    This tool helps you:
    - Explore historical price trends for essential food items
    - Predict future prices using machine learning
    - Understand patterns in food price movements

    ---
    Use the sidebar to explore:
    - 📈 Visualize past price trends
    - 🔮 Predict future prices
    - ℹ️ Learn about the team and tech stack
    """)

# Data Visualization Page
elif selection == "📈 Data Visualization":
    st.title("📈 Food Price Trends")
    selected_item = st.selectbox("Select a food item:", df["Item"].unique())

    filtered_df = df[df["Item"] == selected_item]

    if not filtered_df.empty:
        fig, ax = plt.subplots()
        ax.plot(filtered_df["Date"], filtered_df["Price"], marker='o', linestyle='-')
        ax.set_title(f"{selected_item} Price Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.warning("No data found for the selected item.")

# Price Prediction Page
elif selection == "🔮 Price Prediction":
    st.title("🔮 Predict Food Price")

    model = load_model()

    selected_item = st.selectbox("Select a food item to predict:", df["Item"].unique())
    selected_date = st.date_input("Select a future date to predict the price for:")

    # Extract date features
    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    # Prepare input as raw features
    input_df = pd.DataFrame({
        "Item": [selected_item],
        "Day": [day],
        "Month": [month],
        "Year": [year]
    })

    # Predict
    predicted_price = model.predict(input_df)[0]
    st.success(f"📌 Predicted Price of {selected_item} on {selected_date}: ₹{predicted_price:.2f}")

    # Plot trend with predicted point
    item_hist = df[df["Item"] == selected_item]
    fig, ax = plt.subplots()
    ax.plot(item_hist["Date"], item_hist["Price"], marker='o', linestyle='-', label="Historical")
    ax.scatter(pd.to_datetime(selected_date), predicted_price, color='red', label="Predicted", zorder=5)
    ax.set_title(f"{selected_item} Price Trend + Prediction")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

# About Us Page
elif selection == "ℹ️ About Us":
    st.title("ℹ️ About Us")

    st.markdown("""
    **Team :**  
    - Mohammed Idris: Model developer & Deployment lead
    - Naveen Kumar: Backend developer
    - Praneetha :  QA Tester & Documentation Lead

    **Technologies Used:**  
    - Python, Pandas, Scikit-learn, Matplotlib, Streamlit

    **Use Cases:**  
    - Useful for retailers, students, researchers, and budget planners
    
    **Connect with us:**
    - LinkedIn :
        - Idris chennari
        - Naveen kumar
        - Praneetha
    """)
