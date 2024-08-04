import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import ruptures as rpt

# Configuration
timeframe = '15m'
symbol = 'EURUSDm'
pages = 7

# Load the data
file_path = f'COLLECT CANDLES/{timeframe}/{symbol}{timeframe}{str(pages)}.csv'
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found.")

df = pd.read_csv(file_path)

# Ensure time is in datetime format
df['time'] = pd.to_datetime(df['time'])

# Display the first few rows of the dataframe
print(df.head())
print(df.columns)

# Check if the dataframe is empty
if df.empty:
    raise ValueError("DataFrame is empty. Check the data source.")

# Calculate rolling mean, standard deviation, and deviations
rolling_window = 20
df['rolling_mean'] = df['close'].rolling(window=rolling_window).mean()
df['rolling_std'] = df['close'].rolling(window=rolling_window).std()
df['upper_band'] = df['rolling_mean'] + 2 * df['rolling_std']
df['lower_band'] = df['rolling_mean'] - 2 * df['rolling_std']
df['deviation'] = df['close'] - df['rolling_mean']

# CUSUM parameters
threshold = 2
k = 0.7

# CUSUM calculation
df['cusum_pos'] = (df['deviation'] - k).cumsum().clip(lower=0)
df['cusum_neg'] = (df['deviation'] + k).cumsum().clip(upper=0)
df['cusum_shift'] = ((df['cusum_pos'] > threshold) | (df['cusum_neg'] < -threshold)).astype(int)

# Bayesian change point detection
signal = df['close'].dropna().values
model = "rbf"
algo = rpt.Pelt(model=model).fit(signal)
result = algo.predict(pen=10)

# Filter out-of-bound indices from result
result = [cp for cp in result if cp < len(df)]

# Prepare features for One-Class SVM
df = df.dropna()
features = df[['close', 'tickVolume', 'open', 'high', 'low', 'rolling_mean', 'rolling_std']]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Apply One-Class SVM
ocsvm = OneClassSVM(nu=0.01, kernel='rbf', gamma=0.1)
df['anomaly'] = ocsvm.fit_predict(features_scaled)
df['is_anomaly'] = df['anomaly'] == -1

# Identify critical anomalies
df['anomaly_score'] = np.abs(df['close'].diff().fillna(0))
df['is_critical_anomaly'] = df['is_anomaly'] & (df['anomaly_score'] > df['anomaly_score'].quantile(0.90))

# Save the results
df.to_csv(f'COLLECT CANDLES/{timeframe}/{symbol}{timeframe}{str(pages)}_tagged.csv', index=False)

# Visualization
plt.figure(figsize=(14, 7))

# Plot close prices
plt.plot(df['time'], df['close'], label='Close Price', color='blue', alpha=0.7)

# Plot rolling mean and Bollinger Bands
plt.plot(df['time'], df['rolling_mean'], label='Rolling Mean', color='green', linestyle='--')
plt.fill_between(df['time'], df['upper_band'], df['lower_band'], color='orange', alpha=0.2, label='Bollinger Bands')

# Plot anomalies
anomalies = df[df['is_anomaly']]
if not anomalies.empty:
    plt.scatter(anomalies['time'], anomalies['close'], color='red', label='Anomalies', marker='o', alpha=0.7)

# Plot critical anomalies
critical_anomalies = df[df['is_critical_anomaly']]
if not critical_anomalies.empty:
    plt.scatter(critical_anomalies['time'], critical_anomalies['close'], color='magenta', label='Critical Anomalies', marker='x', s=100, alpha=0.9)

# Plot CUSUM shifts
shift_points = df[df['cusum_shift'] == 1]
plt.scatter(shift_points['time'], shift_points['close'], color='purple', label='CUSUM Shifts', marker='s', alpha=0.7)

# Plot Bayesian change points
for cp in result:
    if 0 <= cp < len(df):  # Ensure the change point index is within bounds
        plt.axvline(x=df['time'].iloc[cp], color='cyan', linestyle='--', label='Change Point' if cp == result[0] else "")

# Labels and legend
plt.title('Close Prices with Anomalies, CUSUM Shifts, and Change Points')
plt.xlabel('Time')
plt.ylabel('Close Price')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

# Show plot
plt.show()

# Print detected critical anomalies
print(df[df['is_critical_anomaly']])
