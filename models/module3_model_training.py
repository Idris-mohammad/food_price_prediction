import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import joblib

# Create models directory if not exists
if not os.path.exists('models'):
    os.makedirs('models')

# Load and prepare data
df = pd.read_csv('data/cleaned_food_prices.csv')
df['Date'] = pd.to_datetime(df['Date'])
df['Day'] = df['Date'].dt.day
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

X = df[['Item', 'Day', 'Month', 'Year']]
y = df['Price']

# Preprocessing and pipeline
preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(), ['Item'])],
    remainder='passthrough'
)
model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model_pipeline.fit(X_train, y_train)

# Save model (safe format)
joblib.dump(model_pipeline, 'models/food_price_model.pkl')
print("✅ Model retrained and saved.")
