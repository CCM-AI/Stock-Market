import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# File uploader for CSV with closing prices
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    # Load the CSV file into a DataFrame
    df = pd.read_csv(uploaded_file)
    
    # Convert the 'Close' column to numeric (in case it's a string)
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    # Feature Engineering: calculate previous close and volume change
    df['Prev_Close'] = df['Close'].shift(1)
    df['Volume_Change'] = (df['Close'] - df['Prev_Close']) / df['Prev_Close'] * 100

    # Drop the first row since it will have NaN values in 'Prev_Close'
    df = df.dropna()

    # Check if 'Close' and 'Prev_Close' columns are available
    if 'Prev_Close' in df.columns:
        # Prepare the target (price increase or decrease)
        y = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)  # 1 if price increases, else 0
        
        # Ensure there's both 1 and 0 in the target variable (y)
        if len(np.unique(y)) > 1:  # Check if both classes (increase and decrease) exist
            # Train the model using a Random Forest Classifier
            model = RandomForestClassifier(n_estimators=100)
            X = df[['Prev_Close', 'Volume_Change']]  # Example features
            model.fit(X, y)

            # Get predicted probabilities
            predictions = model.predict_proba(X)

            # Check the number of classes
            if predictions.shape[1] > 1:
                confidence = predictions[:, 1]  # Confidence of price increase (class 1)
            else:
                confidence = predictions[:, 0]  # If only one class, return the confidence for class 0

            # Display the confidence of price increase
            st.subheader("Confidence of Price Increase at Next Market Open")
            st.write(confidence)

            # Plotting confidence scores
            fig, ax = plt.subplots()
            ax.plot(df['Prev_Close'], confidence, label='Confidence of Increase')
            ax.set_xlabel("Previous Close Price")
            ax.set_ylabel("Confidence")
            ax.set_title("Predicted Confidence for Stock Price Increase")
            st.pyplot(fig)
        else:
            st.error("The dataset does not contain both price increase and decrease classes. Please ensure that your dataset has diverse stock movements.")
    else:
        st.error("The required columns ('Close' and 'Prev_Close') are missing in the data.")
