import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import datetime

# Load synthetic data
data = pd.read_csv('data/synthetic_call_center_data.csv', parse_dates=['date'])

# One-hot encode the weather variable
encoder = OneHotEncoder(sparse_output=False)
weather_encoded = encoder.fit_transform(data[['weather']])
weather_encoded_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather']))

# Combine encoded weather with original data
data = pd.concat([data, weather_encoded_df], axis=1)

# Split data into features and target
X = data[['day_of_week'] + list(weather_encoded_df.columns)]
y = data['call_volume']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

# Streamlit app
st.title("Call Center Forecasting Simulation with Weather Data")

# Show synthetic data
st.subheader("Synthetic Call Center Data (Last 90 Days)")
st.write(data.tail())

# Model performance
st.subheader("Model Performance")
st.write(f'Training MAE: {mae_train:.2f}, Training RMSE: {rmse_train:.2f}')
st.write(f'Testing MAE: {mae_test:.2f}, Testing RMSE: {rmse_test:.2f}')

# Predictions vs Actuals
st.subheader("Predictions vs Actuals (Last 90 Days)")
fig, ax = plt.subplots()
ax.plot(data['date'], y, label='Actual', linestyle='--')
ax.plot(data['date'], model.predict(X), label='Predicted', linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Call Volume')
ax.set_title('Actual vs Predicted Call Volume (Last 90 Days)')
ax.legend()
st.pyplot(fig)

# Generate random weather forecast
st.subheader("Generate Random Weather Forecast for the Next 14 Days")
if st.button("Generate Random Forecast"):
    future_weather = []
    weather_conditions = ["sunny", "rainy", "snowy"]
    weather_symbols = {"sunny": "☀️", "rainy": "🌧️", "snowy": "❄️"}
    for i in range(14):
        day_weather = np.random.choice(weather_conditions)
        future_weather.append(day_weather)
    
    # Encode future weather
    future_weather_encoded = encoder.transform(np.array(future_weather).reshape(-1, 1))
    
    # Prepare future dates
    future_dates = pd.date_range(start=datetime.datetime.now(), periods=14, freq='D')
    future_data = pd.DataFrame(future_dates, columns=['date'])
    future_data['day_of_week'] = future_data['date'].dt.dayofweek
    
    # Combine future data with weather
    future_data_encoded = pd.concat([future_data, pd.DataFrame(future_weather_encoded, columns=encoder.get_feature_names_out(['weather']))], axis=1)
    
    # Generate future predictions
    future_predictions = model.predict(future_data_encoded[['day_of_week'] + list(encoder.get_feature_names_out(['weather']))])
    
    # Generate non-weather influenced predictions
    future_data_non_weather = future_data_encoded[['day_of_week']].copy()
    for col in encoder.get_feature_names_out(['weather']):
        future_data_non_weather[col] = 0  # Set weather columns to zero
    future_predictions_non_weather = model.predict(future_data_non_weather)
    
    # Plot future predictions
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(future_dates, future_predictions, label='Predicted Call Volume (With Weather)', color='blue')
    ax.plot(future_dates, future_predictions_non_weather, label='Predicted Call Volume (No Weather)', color='orange', linestyle='--')
    
    # Add weather symbols
    weather_df = pd.DataFrame({'date': future_dates, 'weather': future_weather})
    for weather, group in weather_df.groupby('weather'):
        ax.scatter(group['date'], [min(future_predictions)]*len(group), label=f"{weather.capitalize()} {weather_symbols[weather]}", alpha=0.6, marker='o')
    
    ax.set_xlabel('Date')
    ax.set_ylabel('Call Volume')
    ax.set_title('Predicted Call Volume for the Next 14 Days')
    ax.set_ylim(0, 1200)  # Set y-axis limits
    ax.legend()
    st.pyplot(fig)
