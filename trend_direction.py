def get_trend_direction(df, ema_length=10, macd_fast=12, macd_slow=26, macd_signal=9):

    # Calculate EMA
    df['EMA'] = ta.ema(df['close'], length=ema_length)
    
    # Calculate MACD
    macd = ta.macd(df['close'], fast=macd_fast, slow=macd_slow, signal=macd_signal)
    df = pd.concat([df, macd], axis=1)
    df=df.dropna()
    # Ensure there are no NaN values in the MACD and EMA columns before checking the trend
    if df[['EMA', 'MACD_12_26_9', 'MACDs_12_26_9']].isnull().values.any():
        return "Not enough data to determine trend"
    
    # Determine the latest trend
    latest_row = df.iloc[-1]
    
    if latest_row['close'] > latest_row['EMA'] and latest_row['MACD_12_26_9'] > latest_row['MACDs_12_26_9']:
        return 1
    elif latest_row['close'] < latest_row['EMA'] and latest_row['MACD_12_26_9'] < latest_row['MACDs_12_26_9']:
        return -1
    else:
        return 0