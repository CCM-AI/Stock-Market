import streamlit as st
import pandas as pd
import requests
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Function to fetch stock data using Alpha Vantage API
def get_stock_data(symbol, api_key):
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}"
    response = requests.get(url)
    data = response.json()

    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df['Close'] = pd.to_numeric(df['Close'])
        df['Open'] = pd.to_numeric(df['1. open'])
        df['Date'] = pd.to_datetime(df.index)
        df = df[['Date', 'Open', 'Close']].sort_values(by='Date')
        return df
    else:
        st.error(f"Error: {data.get('Note', 'No data found.')}")
        return None

# Feature engineering to add previous day's data and moving averages
def add_features(df):
    df['Prev_Close'] = df['Close'].shift(1)
    df['Pct_Change'] = df['Close'].pct_change() * 100
    df['5_SMA'] = df['Close'].rolling(window=5).mean()
    df['10_SMA'] = df['Close'].rolling(window=10).mean()
    df.dropna(inplace=True)
    return df

# Train the Random Forest Regressor model
def train_model(stock_data):
    X = stock_data[['Prev_Close', 'Pct_Change', '5_SMA', '10_SMA']]
    y = stock_data['Open']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    return model, mae

# Streamlit interface
st.title("Saudi Stock Market Daily Opening Price Prediction")

# Input for API key and stock symbol
api_key = st.text_input("Enter your Alpha Vantage API Key")
symbol = st.text_input("Enter Stock Symbol (e.g., TADAWUL:1010)")

if api_key and symbol:
    stock_data = get_stock_data(symbol, api_key)
    if stock_data is not None:
        st.write("Stock Data Preview:")
        st.write(stock_data.tail())
        
        # Add features to the data
        stock_data = add_features(stock_data)
        
        # Train the model
        model, mae = train_model(stock_data)
        
        # Display the model's Mean Absolute Error
        st.write(f"Model's Mean Absolute Error (MAE): {mae:.2f}")
        
        # Make a prediction for the next day's opening price
        last_row = stock_data.iloc[-1]
        last_features = np.array([[
            last_row['Prev_Close'],
            last_row['Pct_Change'],
            last_row['5_SMA'],
            last_row['10_SMA']
        ]])

        predicted_open = model.predict(last_features)[0]
        st.write(f"Predicted Opening Price for {symbol} tomorrow: SAR {predicted_open:.2f}")
