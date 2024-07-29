import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor#,ExtraTressRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.graph_objs as go
import plotly.io as pio
from itertools import combinations
from metaapi_cloud_sdk import MetaApi
import asyncio
import joblib
import sklearn
import os
from datetime import datetime, timedelta
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJlNWZkNGRkYWZmZmIyMDk2YTAyMWYzNjZiY2YxYjYwYSIsInBlcm1pc3Npb25zIjpbXSwiYWNjZXNzUnVsZXMiOlt7ImlkIjoidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpIiwibWV0aG9kcyI6WyJ0cmFkaW5nLWFjY291bnQtbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZXN0LWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1ycGMtYXBpIiwibWV0aG9kcyI6WyJtZXRhYXBpLWFwaTp3czpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZWFsLXRpbWUtc3RyZWFtaW5nLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFzdGF0cy1hcGkiLCJtZXRob2RzIjpbIm1ldGFzdGF0cy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoicmlzay1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsicmlzay1tYW5hZ2VtZW50LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJjb3B5ZmFjdG9yeS1hcGkiLCJtZXRob2RzIjpbImNvcHlmYWN0b3J5LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtdC1tYW5hZ2VyLWFwaSIsIm1ldGhvZHMiOlsibXQtbWFuYWdlci1hcGk6cmVzdDpkZWFsaW5nOio6KiIsIm10LW1hbmFnZXItYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6ImJpbGxpbmctYXBpIiwibWV0aG9kcyI6WyJiaWxsaW5nLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19XSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6ImU1ZmQ0ZGRhZmZmYjIwOTZhMDIxZjM2NmJjZjFiNjBhIiwiaWF0IjoxNzE4NjAwMTc5fQ.P_Q7S9gllY-A0ygrF72pmpyUrno0VZ6_gBXIT31fLOwKFcEEeqopfcEH7yUL6upwTe69YAitfuy3OcDfjBNL7D7Vnuh1FUUUb2EbuGEnBi-B3GSazhZ83uSeAa89zutuNsr7DrptYf-ZHIUw10NSbIHZjhTKOsT9GoV-lv6QLsaxe87wJ8hbv5ajWvo2VKYhJKv0fQFDP2SwsboGnjC4ioqMGoGFhAv2BNLs3nXZL8SWe0tEYIcLmfWB1sWTlPDrsnaOzRdyOzUwbRljoJzns1BUyHW375eQTR93oNjN_P1zJNL9J_V8rqc-nIJcQPhIFytwaFDy-Z3DJHZQJk6mAbDLkphjQAvGeXrwjjk_uQNuY0WzmJmv21dFeQ3aFJhW-wH7l_KYEYjwqxP_H2lU0_AXpiGxn0ZWOoRbyp2Uer8X2hBg_psUd7RlhmfVqXQqTkJLpFpZrP980rq1S_deFWWiZPihXqkzTsFwUTl2DWMyuTxwsQnCBi-Dbt11XY-1kXe7uABqL9L6YbaclohAUVQF1pIVONbELTkTzUqfzqf30TLsxT67xScvd51-smHTDBYyvSDjzWLQ-HCl9gMJlZ43NcJtA9fckE47yf7O5YaaaA3QXhNSO86jiDvJ_spVijZXi62P_d4d71fCpUZTum1WSwRAiF5scvIjwFMSL98'
accountId = os.getenv('ACCOUNT_ID') or '653d65c4-a70f-49ac-a6de-deea63238808'
#symbol_list =['BTCUSDm',]
timeframe='30m'
pages=7
n_estimators=100
min_samples_leaf=1
max_depth=50
def generate_new_features(df,column1,column2):
    # Check if required columns are present
    required_columns = [str(column1),str(column2)]#[f'{column1}', f'{column2}']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain '{col}' column.")

    # Addition/Subtraction
    df[f'{column1}_plus_{column2}'] = df[column1] + df[column2]
    df[f'{column1}_minus_{column2}'] = df[column1] - df[column2]

    # Multiplication/Division
    df[f'{column1}_times_{column2}'] = df[column1] * df[column2]
    df[f'{column1}_div_{column2}'] = df[column1] / df[column2]

    # Mean and Standard Deviation
    df[f'mean_column_{column1}{column2}'] = df[[column1, column2]].mean(axis=1)
    df[f'std_column_{column1}{column2}'] = df[[column1, column2]].std(axis=1)

    # Sum
    df[f'sum_column_{column1}{column2}'] = df[[column1, column2]].sum(axis=1)

    # Logarithm and Exponential
    df[f'log_column1_{column1}{column2}'] = np.log(df[column1])
    df[f'exp_column1_{column1}{column2}'] = np.exp(df[column1])

    # Square Root
    df[f'sqrt_column1_{column1}{column2}'] = np.sqrt(df[column1])

    # Rolling Mean and Cumulative Sum
    df[f'rolling_mean_column1_{column1}{column2}'] = df[column1].rolling(window=3).mean()
    df[f'cumulative_sum_column1_{column1}{column2}'] = df[column1].cumsum()

    # Polynomial Features
    poly = PolynomialFeatures(degree=2)
    poly_features = poly.fit_transform(df[[column1, column2]])
    df[f'poly_column1_2_{column1}{column2}'] = poly_features[:, 2]  # x1 * x2 term
    df[f'poly_column1_2_squared_{column1}{column2}'] = poly_features[:, 3]  # x1^2 term
    df[f'poly_column2_squared_{column1}{column2}'] = poly_features[:, 4]  # x2^2 term

    # Interaction Term
    df[f'interaction_{column1}{column2}'] = df[column1] * df[column2]
    #if ('close' in [column1,column2]) and ('open' in [column1,column2]):
    # Binning
    df[f'binned_column_{column1}{column2}'] = pd.cut(df[column1], bins=3, labels=['low', 'medium', 'high'])
    #df[f'binned_column_2_{column1}{column2}'] = pd.cut(df[column2], bins=3, labels=['low', 'medium', 'high'])

    # Lagged Features
    df[f'lagged_column1_{column1}{column2}'] = df[column1].shift(1)
    #df[f'lagged_column2_{column1}{column2}'] = df[column1].shift(2)
    #df[f'lagged_column3_{column1}{column2}'] = df[column1].shift(3)

    # Custom Function Example
    df[f'custom_column1_{column1}{column2}'] = df[column1].apply(lambda x: x * 2)
    #df[f'custom_column2_{column1}{column2}'] = df[column2].apply(lambda x: x * 2)

    return df

def decimal_places(number):
    # Convert the number to a string
    num_str = str(number)
    
    # Check if there is a decimal point
    if '.' in num_str:
        # Find the index of the decimal point
        decimal_index = num_str.index('.')
        
        # Count the characters after the decimal point
        num_decimal_places = len(num_str) - decimal_index - 1
        
        return num_decimal_places
    else:
        # If there is no decimal point, return 0
        return 0

def add_stop_losse(df):
    pass
    # Initialize new columns
    df['type'] = None
    df['sell_high'] = None
    df['buy_low'] = None

    # Determine the candle type dynamically
    for i in range(1, len(df)):
        if df.at[i, 'close'] > df.at[i-1, 'close']:
            df.at[i, 'type'] = 'buy'
        else:
            df.at[i, 'type'] = 'sell'

    # Set the first candle type as 'buy' or 'sell' as needed
    df.at[0, 'type'] = 'buy' if df.at[0, 'close'] > df.at[0, 'open'] else 'sell'


    # Loop through the DataFrame to fill Sell_High and Buy_Low
    for index, row in df.iterrows():
        if row['type'] == 'sell':
            df.at[index, 'sell_high'] = row['high']
        elif row['type'] == 'buy':
            df.at[index, 'buy_low'] = row['low']
    # Loop through the DataFrame to fill Sell_High and Buy_Low
    for index, row in df.iterrows():
        if row['type'] == 'sell':
            df.at[index, 'sell_low'] = row['low']
        elif row['type'] == 'buy':
            df.at[index, 'buy_high'] = row['high']
    # Combine Sell_High and Buy_Low into one column
    df['stop_losses'] = df['sell_high'].combine_first(df['buy_low'])
    df['trade_max'] = df['sell_low'].combine_first(df['buy_high'])
    df['sell_high']= df['sell_high'].fillna(1)
    df['buy_low']= df['buy_low'].fillna(1)
    df['sell_low']= df['sell_low'].fillna(1)
    df['buy_high']= df['buy_high'].fillna(1)

    # Drop the individual Sell_High and Buy_Low columns
    #df.drop(columns=['sell_high', 'buy_low',], inplace=True)
    return df


   
async def get_candles_m(timeframe,symbol,pages):
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']
    timeframe=timeframe
    symbol=symbol
    if initial_state not in deployed_states:
        # wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
        
    try:
        # Create an empty dataframe to store the candlestick data
        df = pd.DataFrame()


        # retrieve last 10K100
        pages = pages
        #print(f'Downloading {pages}K latest candles for {symbol}')
        started_at = datetime.now().timestamp()
        start_time = None
        candles = None
        for i in range(pages):
            try:
                # the API to retrieve historical market data is currently available for G1 only
                candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=start_time)
                print(f'Downloaded {i}000 historical candles for {symbol}')
                
                
                if not candles:
                    pass
                else:
                    #Create a new dataframe for each iteration and add it to the main dataframe
                    new_df = pd.DataFrame(candles)
                    df = pd.concat([df, new_df], ignore_index=True)
                    print(f'Candles added to dataframe')
                df.to_csv(f'{symbol}{timeframe}{str(pages)}.csv', index=False)
            except:
                pass

        return df

    except Exception as e:
        raise e
        
async def main2(timeframe,pages):

    for symbol in symbol_list:
        try:

            df=await get_candles_m(timeframe,symbol,pages)

        except Exception as e:
            #print(f'{symbol} failed')
            raise e
            pass
asyncio.run(main2(timeframe,pages))