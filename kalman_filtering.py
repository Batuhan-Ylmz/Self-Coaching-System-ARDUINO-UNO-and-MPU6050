import pandas as pd
import numpy as np

def kalman_filter(data, process_variance, measurement_variance):
    n = len(data)
    xhat = np.zeros(n)  # A posteriori estimate
    P = np.zeros(n)  # A posteriori error estimate
    xhat_minus = np.zeros(n)  # A priori estimate
    P_minus = np.zeros(n)  # A priori error estimate
    K = np.zeros(n)  # Kalman gain

    for k in range(1, n):
        # Time update (prediction)
        xhat_minus[k] = xhat[k - 1]
        P_minus[k] = P[k - 1] + process_variance

        # Measurement update (correction)
        K[k] = P_minus[k] / (P_minus[k] + measurement_variance)
        xhat[k] = xhat_minus[k] + K[k] * (data[k] - xhat_minus[k])
        P[k] = (1 - K[k]) * P_minus[k]

    return xhat

# Read raw data from CSV
raw_data = pd.read_csv('raw_data.csv')

# Columns to apply Kalman filter
columns_to_filter = ['accX', 'accY', 'accZ', 'gyroX', 'gyroY', 'gyroZ']

# Tune parameters based on requirements
process_variance = 1e-3
measurement_variance = 1

# Apply Kalman filter to all columns
for column in columns_to_filter:
    axis_data = raw_data[column].values
    filtered_data = kalman_filter(axis_data, process_variance, measurement_variance)
    raw_data[column] = filtered_data

# Save the preprocessed data to a new CSV file
raw_data.to_csv('kalman_filtered_data.csv', index=False)
