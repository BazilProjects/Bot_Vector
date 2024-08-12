import pandas as pd
import pandas_ta as ta
import matplotlib.pyplot as plt

# Define your variables (timeframe, symbol, pages)
timeframe = '15m'  # Example timeframe
symbol = 'EURUSDm'   # Example symbol
pages =8        # Example page number

# Read the data from the CSV file
df = pd.read_csv(f'../COLLECT CANDLES/{timeframe}/{symbol}{timeframe}{str(pages)}.csv')

# Split the DataFrame: keep the last row separate (actual) and work with the rest
actual = df.tail(1)
df = df.head(len(df) - 1)

# If the DataFrame is not empty, proceed with processing
if not df.empty:
    print(f'Symbol: {symbol}')
    
    dt=df

    # Calculate the 10-day and 30-day Simple Moving Averages (SMA)
    dt['sma_10'] = ta.sma(dt['close'], length=10)
    dt['sma_30'] = ta.sma(dt['close'], length=30)

    # Buy Signal: When sma_10 crosses above sma_30
    dt['buy_signal'] = (dt['sma_10'] > dt['sma_30']) & (dt['sma_10'].shift(1) <= dt['sma_30'].shift(1))

    # Sell Signal: When sma_10 crosses below sma_30
    dt['sell_signal'] = (dt['sma_10'] < dt['sma_30']) & (dt['sma_10'].shift(1) >= dt['sma_30'].shift(1))

    # Plotting the Close Price along with the SMAs and Buy/Sell Signals
    plt.figure(figsize=(14, 7))
    plt.plot(dt['close'], label='Close Price', color='blue')
    plt.plot(dt['sma_10'], label='SMA 10', color='green')
    plt.plot(dt['sma_30'], label='SMA 30', color='red')

    # Plot buy signals
    plt.plot(dt.index[dt['buy_signal']], dt['close'][dt['buy_signal']], '^', markersize=10, color='g', lw=0, label='Buy Signal')

    # Plot sell signals
    plt.plot(dt.index[dt['sell_signal']], dt['close'][dt['sell_signal']], 'v', markersize=10, color='r', lw=0, label='Sell Signal')

    plt.title(f'{symbol} Close Price with SMA Buy and Sell Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Display the signals
    print(dt[['close', 'sma_10', 'sma_30', 'buy_signal', 'sell_signal']].tail(10))
else:
    print(f"No data available for {symbol}.")
