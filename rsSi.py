import os
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuration
timeframe = '15m'
symbol = 'XAUUSDm'
pages = 8

# Construct the file path
file_path = f'COLLECT CANDLES/{timeframe}/{symbol}{timeframe}{str(pages)}.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"File {file_path} not found.")

# Load the data
df = pd.read_csv(file_path)

# Ensure time is in datetime format
df['time'] = pd.to_datetime(df['time'])


# Ensure the DataFrame has a 'close' column
if 'close' not in df.columns:
    raise ValueError("The DataFrame does not contain a 'close' column.")

def calculate_rsi(data, period=14):
    # Calculate price changes
    delta = data['close'].diff()

    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate average gains and losses using SMA
    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


# Calculate RSI and add it to the DataFrame
df['rsi'] = calculate_rsi(df)

# Create the subplots: two rows, one column
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1, row_heights=[0.7, 0.3])

# Plot the closing prices on the first row
fig.add_trace(go.Scatter(x=df['time'], y=df['close'], mode='lines', name='Close Price'), row=1, col=1)

# Plot the RSI on the second row
fig.add_trace(go.Scatter(x=df['time'], y=df['rsi'], mode='lines', name='RSI'), row=2, col=1)

# Add the 70 and 30 level lines to the RSI plot
fig.add_shape(type='line', x0=df['time'].min(), x1=df['time'].max(), y0=70, y1=70,
              line=dict(color='green', dash='dash'), row=2, col=1)
fig.add_shape(type='line', x0=df['time'].min(), x1=df['time'].max(), y0=30, y1=30,
              line=dict(color='red', dash='dash'), row=2, col=1)

# Update layout
fig.update_layout(title=f'{symbol} Price and RSI',
                  yaxis_title='Price',
                  yaxis2_title='RSI',
                  xaxis2_title='Time')

# Add RSI range
fig.update_yaxes(range=[0, 100], row=2, col=1)

# Show the plot
fig.show()
print(df['rsi'].iloc[-1])