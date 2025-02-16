import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from transformers import pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Set Streamlit page config
st.set_page_config(page_title="Sales Forecasting", layout="wide")

st.title("📊 Sales Forecasting & Recommendations")

# Load datasets
@st.cache_data
def load_data():
    foot_traffic = pd.read_csv("foot_traffic.csv")  # Columns: [date, time, people_in, people_out]
    sales_data = pd.read_csv("sales_data.csv")  # Columns: [date, product_id, sales, price, stock_level]
    product_data = pd.read_csv("product_data.csv")  # Columns: [product_id, category, shelf_location]
    
    sales_data['date'] = pd.to_datetime(sales_data['date'])
    foot_traffic['date'] = pd.to_datetime(foot_traffic['date'])
    
    # Merge datasets
    df = sales_data.merge(foot_traffic, on='date', how='left')
    
    # Feature engineering
    df['conversion_rate'] = df['sales'] / (df['people_in'] + 1)  # Avoid division by zero
    df.fillna(0, inplace=True)
    
    return df

df = load_data()

st.subheader("📂 Preview of Merged Dataset")
st.dataframe(df.head())

# Normalize data
scaler = MinMaxScaler()
df[['sales', 'people_in', 'conversion_rate']] = scaler.fit_transform(df[['sales', 'people_in', 'conversion_rate']])

# Split data
train, test = train_test_split(df, test_size=0.2, shuffle=False)

st.subheader("📈 Data Distribution")
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.histplot(df['sales'], bins=30, ax=ax[0], kde=True)
ax[0].set_title("Sales Distribution")
sns.histplot(df['people_in'], bins=30, ax=ax[1], kde=True)
ax[1].set_title("Foot Traffic Distribution")
st.pyplot(fig)

# Prepare time-series data
sequence_length = 30
train_gen = TimeseriesGenerator(train[['sales']].values, train['sales'].values, length=sequence_length, batch_size=16)
test_gen = TimeseriesGenerator(test[['sales']].values, test['sales'].values, length=sequence_length, batch_size=16)

# Define LSTM model
st.subheader("🛠 Training LSTM Model")
with st.spinner("Training LSTM model..."):
    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(sequence_length, 1)),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(train_gen, epochs=20, validation_data=test_gen, verbose=1)

# Make predictions
predictions = model.predict(test_gen)

st.subheader("📉 Sales Forecasting")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(test['date'][sequence_length:], test['sales'][sequence_length:], label="Actual Sales")
ax.plot(test['date'][sequence_length:], predictions, label="Predicted Sales", linestyle="dashed")
ax.legend()
st.pyplot(fig)

# Load GPT model for recommendations
@st.cache_resource
def load_gpt_model():
    return pipeline("text-generation", model="gpt2")

gpt = load_gpt_model()

# Function to generate insights
def generate_sales_insight(trend, product_name):
    prompt = f"Sales for {product_name} have {trend}. Suggest why and how to improve it."
    response = gpt(prompt, max_length=50)
    return response[0]['generated_text']

# Select a product
product_id = st.selectbox("🔍 Select a Product", df['product_id'].unique())

# Predict sales for selected product
st.subheader(f"📊 Forecast & Insights for Product {product_id}")
predicted_sales = model.predict(test_gen)[-1][0]
trend = "expected to increase" if predicted_sales > test['sales'].iloc[-1] else "expected to decrease"
insight = generate_sales_insight(trend, f"Product {product_id}")

st.metric(label="📈 Predicted Sales", value=round(predicted_sales, 2))
st.text_area("💡 AI-Generated Recommendation:", insight)

st.success("✅ Analysis Complete! Adjust pricing, stock, and promotions accordingly.")

