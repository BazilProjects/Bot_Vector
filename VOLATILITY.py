import pandas as pd
import numpy as np
import os
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configuration
timeframe = '15m'
symbol = 'EURUSDm'
pages = 7

# Construct the file path
file_path = f'COLLECT CANDLES/{timeframe}/{symbol}{timeframe}{str(pages)}.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found.")

# Load the data
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

# Calculate returns
df['Returns'] = df['close'].pct_change()

# Calculate historical volatility (standard deviation of returns)
volatility = df['Returns'].std() * np.sqrt(252)  # Annualized volatility
print(f'Historical Volatility: {volatility}')

# Drop NaN values
df.dropna(inplace=True)

# Prepare data for ExtraTrees model
features = ['rolling_mean', 'rolling_std', 'upper_band', 'lower_band', 'deviation', 'Returns']
target = 'Returns'

# Shift the target to forecast the next period's returns
df['Target'] = df['Returns'].shift(-1)

# Drop the last row since it will have NaN in the target column after shifting
df.dropna(inplace=True)

X = df[features]
y = df['Target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the ExtraTrees model
model = ExtraTreesRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Display the first few rows of the updated dataframe
print(df.head())
