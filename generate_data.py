import pandas as pd
import numpy as np

# Generate synthetic call center data
np.random.seed(42)

# Define parameters
n_days = 90

# Create a time series
date_rng = pd.date_range(start='2023-01-01', periods=n_days, freq='D')

# Weekly pattern: Call volumes for each day of the week
daily_volumes = {
    0: 1000,  # Monday
    1: 900,   # Tuesday
    2: 850,   # Wednesday
    3: 800,   # Thursday
    4: 850,   # Friday
    5: 500,   # Saturday
    6: 300    # Sunday
}

# Generate call volumes with random variance
call_volumes = np.array([
    daily_volumes[day_of_week] * (1 + np.random.uniform(-0.04, 0.04))
    for day_of_week in date_rng.dayofweek
])

# Generate random weather data with correlation
weather_conditions = ['sunny', 'rainy', 'snowy']
weather = np.random.choice(weather_conditions, n_days, p=[0.6, 0.3, 0.1])

# Adjust call volumes based on weather
weather_effect = {'sunny': 1.1, 'rainy': 1.0, 'snowy': 0.95}
call_volumes = (call_volumes * np.vectorize(weather_effect.get)(weather)).astype(int)

# Create a DataFrame
data = pd.DataFrame(date_rng, columns=['date'])
data['call_volume'] = call_volumes
data['day_of_week'] = data['date'].dt.dayofweek
data['weather'] = weather

# Save synthetic data to CSV
data.to_csv('data/synthetic_call_center_data.csv', index=False)
