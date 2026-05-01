# Ride_Cancellation_Analysis
This notebook contains data preprocessing, feature engineering, and model training for predicting ride cancellations.

# Ride Cancellation Prediction App

## Overview

This project is a machine learning-based web application that predicts whether a ride will be cancelled based on ride conditions such as time, location, weather, and pricing.

The application is built using Streamlit and provides an interactive interface for users to input ride details and get real-time predictions.

## Features

* Predicts ride cancellation using a trained Random Forest model
* User-friendly interface with clean layout
* Manual time input (HH:MM format)
* Real-time prediction with confidence score
* Handles categorical data using encoding

## Tech Stack

* Python
* Pandas
* Scikit-learn
* Streamlit

## Project Structure

├── app.py
├── ride.csv
├── requirements.txt
├── Ride_Cancellation_Analysis.ipynb
├── README.md

## How It Works

1. The dataset is preprocessed using one-hot encoding
2. A Random Forest model is trained on historical ride data
3. User inputs are transformed to match training format
4. The model predicts whether a ride will be cancelled
5. The result is displayed with a confidence score

## Installation & Run Locally

1. Clone the repository
2. Install dependencies

pip install -r requirements.txt

3. Run the app
python -m streamlit run app.py

## Deployment

This app can be deployed easily using Streamlit Community Cloud by connecting your GitHub repository.

## Future Improvements

* Add model accuracy metrics
* Include feature importance visualization
* Improve UI with advanced styling
* Add recommendation system for optimal booking time

## Author
Abdul Khan
