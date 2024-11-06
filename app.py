import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import requests

# --- Streamlit App Title and Description ---
st.title("Stock Market Prediction App")
st.write("""
This app predicts the price movement (increase or decrease) of stocks based on historical data.
You can upload your stock data as a CSV, and the app will perform feature engineering, training, and display predictions.
""")

# --- File Upload Section ---
uploaded_file = st.file_uploader("Upload your stock data CSV file", type=["csv"])

# --- Real-time Stock Data Fetching (Optional) ---
def get_realtime_data(stock_symbol):
    api_key = 'your_alpha_vantage_api_key'  # Replace with your Alpha Vantage API key
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    
    if "Time Series (Daily)" in data:
        df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
        df['Close'] = pd.to_numeric(df['Close'])
        df['Date'] = df.index
        df = df[['Date', 'Close']].sort_values(by='Date')
        return df
    else:
        st.error("Unable to fetch real-time data. Please check the stock symbol or API key.")
        return None

# --- If CSV is uploaded ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.write(df.head())

    # --- Preprocess Data ---
    df['Prev_Close'] = df['Close'].shift(1)
    df['Price_Movement'] = np.where(df['Close'] > df['Prev_Close'], 1, 0)  # 1 = Increase, 0 = Decrease
    df['Pct_Change'] = df['Close'].pct_change()  # Percentage change in closing price
    df.dropna(inplace=True)  # Drop rows with missing values

    # --- Feature Engineering ---
    df['5_SMA'] = df['Close'].rolling(window=5).mean()  # 5-day simple moving average
    df['5_EMA'] = df['Close'].ewm(span=5, adjust=False).mean()  # 5-day exponential moving average
    df['10_SMA'] = df['Close'].rolling(window=10).mean()  # 10-day moving average
    df['10_EMA'] = df['Close'].ewm(span=10, adjust=False).mean()  # 10-day EMA

    # --- Display Preprocessed Data ---
    st.write("Data After Feature Engineering:")
    st.write(df.tail())

    # --- Split Data for Training ---
    X = df[['Close', 'Prev_Close', 'Pct_Change', '5_SMA', '5_EMA', '10_SMA', '10_EMA']]  # Features
    y = df['Price_Movement']  # Target variable

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- Train Model ---
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Make Predictions ---
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # --- Display Model Accuracy ---
    st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

    # --- Display Predictions and Probabilities ---
    pred_proba = model.predict_proba(X_test)
    pred_df = pd.DataFrame({'Predicted_Movement': y_pred, 'Pred_Prob_Up': pred_proba[:, 1]})
    st.write(pred_df.head())

    # --- Visualize Stock Data ---
    fig = px.line(df, x='Date', y='Close', title='Stock Closing Prices')
    st.plotly_chart(fig)

    # --- Visualize Prediction Probabilities ---
    st.write("Prediction Probabilities for Price Increase:")
    st.write(pred_df[['Predicted_Movement', 'Pred_Prob_Up']])

    # --- Display Prediction Confidence ---
    confidence = pred_proba[:, 1].mean()
    st.write(f"Average Confidence of Prediction: {confidence * 100:.2f}%")

# --- Real-Time Data Option ---
if st.button("Get Real-Time Data (Example: AAPL)"):
    stock_symbol = "AAPL"  # Example: Apple stock
    real_time_df = get_realtime_data(stock_symbol)
    
    if real_time_df is not None:
        st.write("Real-Time Stock Data:")
        st.write(real_time_df.tail())

        # --- Feature Engineering on Real-Time Data ---
        real_time_df['Prev_Close'] = real_time_df['Close'].shift(1)
        real_time_df['Price_Movement'] = np.where(real_time_df['Close'] > real_time_df['Prev_Close'], 1, 0)
        real_time_df['Pct_Change'] = real_time_df['Close'].pct_change()
        real_time_df['5_SMA'] = real_time_df['Close'].rolling(window=5).mean()
        real_time_df['5_EMA'] = real_time_df['Close'].ewm(span=5, adjust=False).mean()

        # --- Prepare Features for Prediction ---
        real_time_df.dropna(inplace=True)
        X_real_time = real_time_df[['Close', 'Prev_Close', 'Pct_Change', '5_SMA', '5_EMA']]
        
        # --- Make Predictions on Real-Time Data ---
        real_time_pred = model.predict(X_real_time)
        real_time_pred_prob = model.predict_proba(X_real_time)[:, 1]

        # --- Display Real-Time Prediction ---
        real_time_df['Predicted_Movement'] = real_time_pred
        real_time_df['Prediction_Confidence'] = real_time_pred_prob
        st.write(real_time_df[['Date', 'Close', 'Predicted_Movement', 'Prediction_Confidence']].tail())

        # --- Visualize Real-Time Data ---
        fig = px.line(real_time_df, x='Date', y='Close', title=f'Real-Time {stock_symbol} Stock Data')
        st.plotly_chart(fig)

else:
    st.write("Upload a CSV file or press the button to get real-time data.")

