import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder

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

print(f'Training MAE: {mae_train}, Training RMSE: {rmse_train}')
print(f'Testing MAE: {mae_test}, Testing RMSE: {rmse_test}')
