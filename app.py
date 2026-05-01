import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Ride Cancellation Predictor", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv("ride.csv")

df = load_data()
  
df_encoded = pd.get_dummies(
    df,
    columns=['pickup_location', 'drop_location', 'weather', 'day_of_week'],
    drop_first=True
)

X = df_encoded.drop("cancelled", axis=1)
y = df_encoded["cancelled"]

@st.cache_resource
def train_model(X, y):
    model = RandomForestClassifier()
    model.fit(X, y)
    return model

model = train_model(X, y)

st.markdown("""
    <h1 style='text-align: center;'>Ride Cancellation Predictor</h1>
    <p style='text-align: center; color: gray;'>
    Smart prediction system based on ride conditions
    </p>
""", unsafe_allow_html=True)

st.info("This model predicts ride cancellation based on historical data")

st.markdown("---")

left, right = st.columns([1, 1])

with left:
    st.subheader("Ride Details")

   
    time_input = st.text_input("Booking Time (HH:MM)", "10:30")

    try:
        hour, minute = map(int, time_input.split(":"))
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            booking_hour = hour
            time_valid = True
        else:
            st.error("Enter valid time (00:00 to 23:59)")
            time_valid = False
    except:
        st.error("Format should be HH:MM")
        time_valid = False


    day_of_week = st.selectbox("Day of Week", df["day_of_week"].unique())

    ride_distance = st.number_input("Ride Distance (km)", 1.0, 50.0, 5.0)

    driver_rating = st.selectbox("Driver Rating", [1, 2, 3, 4, 5])

    surge = st.selectbox("Surge Level", [1, 2, 3, 4, 5])

    pickup = st.selectbox("Pickup Location", df["pickup_location"].unique())
    drop = st.selectbox("Drop Location", df["drop_location"].unique())
    weather = st.selectbox("Weather", df["weather"].unique())

    predict_btn = st.button("Predict Cancellation")

with right:
    st.subheader("Prediction")

    if predict_btn and time_valid:

        input_data = pd.DataFrame({
            "booking_hour": [booking_hour],
            "day_of_week": [day_of_week],
            "ride_distance_km": [ride_distance],
            "driver_rating": [driver_rating],
            "surge_pricing": [surge],
            "pickup_location": [pickup],
            "drop_location": [drop],
            "weather": [weather]
        })

        combined = pd.concat([df.drop("cancelled", axis=1), input_data], ignore_index=True)

        combined_encoded = pd.get_dummies(
            combined,
            columns=['pickup_location', 'drop_location', 'weather', 'day_of_week'],
            drop_first=True
        )

        input_encoded = combined_encoded.tail(1)
        input_encoded = input_encoded.reindex(columns=X.columns, fill_value=0)

        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        st.markdown("### Result")

        if prediction == 1:
            st.error("High cancellation risk")
            st.metric("Risk Score", f"{probability:.2f}")
        else:
            st.success("Ride likely to complete")
            st.metric("Success Score", f"{1 - probability:.2f}")

    elif predict_btn and not time_valid:
        st.warning("Please fix the time input before predicting")

    else:
        st.write("Provide ride details and click Predict")

st.markdown("---")
st.caption("Machine Learning powered ride prediction system")