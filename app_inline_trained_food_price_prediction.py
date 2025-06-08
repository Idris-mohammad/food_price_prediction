
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Food Price Prediction App", page_icon="🛒", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_food_prices.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def train_model():
    df = load_data()
    df["Day"] = df["Date"].dt.day
    df["Month"] = df["Date"].dt.month
    df["Year"] = df["Date"].dt.year

    X = df[["Item", "Day", "Month", "Year"]]
    y = df["Price"]

    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), ["Item"])],
        remainder='passthrough'
    )
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    model_pipeline.fit(X, y)
    return model_pipeline

# Load data
df = load_data()
model = train_model()

# Sidebar navigation
st.sidebar.title("📂 Navigation")
selection = st.sidebar.radio("Go to", ["🏠 Home", "📈 Data Visualization", "🔮 Price Prediction", "ℹ️ About Us"])

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

elif selection == "🔮 Price Prediction":
    st.title("🔮 Predict Food Price")

    selected_item = st.selectbox("Select a food item to predict:", df["Item"].unique())
    selected_date = st.date_input("Select a future date to predict the price for:")

    day = selected_date.day
    month = selected_date.month
    year = selected_date.year

    input_df = pd.DataFrame({
        "Item": [selected_item],
        "Day": [day],
        "Month": [month],
        "Year": [year]
    })

    predicted_price = model.predict(input_df)[0]
    st.success(f"📌 Predicted Price of {selected_item} on {selected_date}: ₹{predicted_price:.2f}")

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
    - LinkedIn:
        - Idris chennari
        - Naveen kumar
        - Praneetha
    """)
