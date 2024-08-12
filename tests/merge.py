import pandas as pd

# Example DataFrame
data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=3000, freq='5T'),
    'value_1h': [0] * 24 + [None] * (3000 - 24),
    'value_4h': [0] * 6 + [None] * (3000 - 6),
    'value_1d': [0] * 2 + [None] * (3000 - 2),
    'value_30m': [0] * 48 + [None] * (3000 - 48),
    'value_15m': [0] * 96 + [None] * (3000 - 96),
    'value_5m': list(range(3000))
}
df_merged = pd.DataFrame(data).set_index('timestamp')

# Handling missing values
df_filled_forward = df_merged.fillna(method='ffill')  # Forward fill
df_filled_backward = df_merged.fillna(method='bfill')  # Backward fill
df_interpolated = df_merged.interpolate(method='linear')  # Linear interpolation

# Drop rows with missing values (optional)
df_dropped = df_merged.dropna()

# Display the cleaned DataFrames
print("Forward Filled DataFrame:")
print(df_filled_forward.head(10))

print("\nBackward Filled DataFrame:")
print(df_filled_backward.head(10))

print("\nInterpolated DataFrame:")
print(df_interpolated.head(10))

print("\nDataFrame with Dropped NaNs:")
print(df_dropped.head(10))
