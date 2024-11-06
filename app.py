import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Title and Description
st.title("Saudi Stock Market Price Prediction")
st.write("""
This app predicts the likelihood of stock price increase at the next market opening based on historical data.
""")

# File upload for CSV with closing prices
st.subheader("Upload Daily Closing Prices CSV")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Display the first few rows of the data
    st.write("Here is the data you've uploaded:")
    st.write(df.head())

    # Check if the required columns (Date, Stock, Closing Price) are in the uploaded file
    if 'Date' in df.columns and 'Stock' in df.columns and 'Close' in df.columns:
        # Feature Engineering and Predictions Logic
        st.write("Proceeding with feature engineering and predictions...")

        # Simple feature engineering example
        df['Prev_Close'] = df['Close'].shift(1)
        df['Volume_Change'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close'] * 100
        
        # Mock prediction model (RandomForestClassifier)
        model = RandomForestClassifier(n_estimators=100)
        X = df[['Prev_Close', 'Volume_Change']]  # Example features
        y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # Example target (price increase)
        model.fit(X, y)

        predictions = model.predict_proba(X)
        confidence = predictions[:, 1]  # Confidence of price increase

        # Display confidence of price increase
        st.subheader("Confidence of Price Increase at Next Market Open")
        st.write(confidence)

        # Plotting confidence scores for each stock
        fig, ax = plt.subplots()
        ax.plot(df['Date'], confidence, label='Confidence of Increase')
        ax.set_xlabel("Date")
        ax.set_ylabel("Confidence")
        ax.set_title("Predicted Confidence for Stock Price Increase")
        st.pyplot(fig)

    else:
        st.error("Uploaded CSV must contain 'Date', 'Stock', and 'Close' columns.")
