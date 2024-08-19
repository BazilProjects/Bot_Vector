import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
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
import docx
from sklearn.model_selection import cross_val_score
import pandas_ta as ta

from datetime import datetime, timedelta
token = os.getenv('TOKEN') or 'eyJhbGciOiJSUzUxMiIsInR5cCI6IkpXVCJ9.eyJfaWQiOiJlNWZkNGRkYWZmZmIyMDk2YTAyMWYzNjZiY2YxYjYwYSIsInBlcm1pc3Npb25zIjpbXSwiYWNjZXNzUnVsZXMiOlt7ImlkIjoidHJhZGluZy1hY2NvdW50LW1hbmFnZW1lbnQtYXBpIiwibWV0aG9kcyI6WyJ0cmFkaW5nLWFjY291bnQtbWFuYWdlbWVudC1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZXN0LWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1ycGMtYXBpIiwibWV0aG9kcyI6WyJtZXRhYXBpLWFwaTp3czpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciIsIndyaXRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoibWV0YWFwaS1yZWFsLXRpbWUtc3RyZWFtaW5nLWFwaSIsIm1ldGhvZHMiOlsibWV0YWFwaS1hcGk6d3M6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6Im1ldGFzdGF0cy1hcGkiLCJtZXRob2RzIjpbIm1ldGFzdGF0cy1hcGk6cmVzdDpwdWJsaWM6KjoqIl0sInJvbGVzIjpbInJlYWRlciJdLCJyZXNvdXJjZXMiOlsiKjokVVNFUl9JRCQ6KiJdfSx7ImlkIjoicmlzay1tYW5hZ2VtZW50LWFwaSIsIm1ldGhvZHMiOlsicmlzay1tYW5hZ2VtZW50LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJjb3B5ZmFjdG9yeS1hcGkiLCJtZXRob2RzIjpbImNvcHlmYWN0b3J5LWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIiwid3JpdGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19LHsiaWQiOiJtdC1tYW5hZ2VyLWFwaSIsIm1ldGhvZHMiOlsibXQtbWFuYWdlci1hcGk6cmVzdDpkZWFsaW5nOio6KiIsIm10LW1hbmFnZXItYXBpOnJlc3Q6cHVibGljOio6KiJdLCJyb2xlcyI6WyJyZWFkZXIiLCJ3cml0ZXIiXSwicmVzb3VyY2VzIjpbIio6JFVTRVJfSUQkOioiXX0seyJpZCI6ImJpbGxpbmctYXBpIiwibWV0aG9kcyI6WyJiaWxsaW5nLWFwaTpyZXN0OnB1YmxpYzoqOioiXSwicm9sZXMiOlsicmVhZGVyIl0sInJlc291cmNlcyI6WyIqOiRVU0VSX0lEJDoqIl19XSwidG9rZW5JZCI6IjIwMjEwMjEzIiwiaW1wZXJzb25hdGVkIjpmYWxzZSwicmVhbFVzZXJJZCI6ImU1ZmQ0ZGRhZmZmYjIwOTZhMDIxZjM2NmJjZjFiNjBhIiwiaWF0IjoxNzE4NjAwMTc5fQ.P_Q7S9gllY-A0ygrF72pmpyUrno0VZ6_gBXIT31fLOwKFcEEeqopfcEH7yUL6upwTe69YAitfuy3OcDfjBNL7D7Vnuh1FUUUb2EbuGEnBi-B3GSazhZ83uSeAa89zutuNsr7DrptYf-ZHIUw10NSbIHZjhTKOsT9GoV-lv6QLsaxe87wJ8hbv5ajWvo2VKYhJKv0fQFDP2SwsboGnjC4ioqMGoGFhAv2BNLs3nXZL8SWe0tEYIcLmfWB1sWTlPDrsnaOzRdyOzUwbRljoJzns1BUyHW375eQTR93oNjN_P1zJNL9J_V8rqc-nIJcQPhIFytwaFDy-Z3DJHZQJk6mAbDLkphjQAvGeXrwjjk_uQNuY0WzmJmv21dFeQ3aFJhW-wH7l_KYEYjwqxP_H2lU0_AXpiGxn0ZWOoRbyp2Uer8X2hBg_psUd7RlhmfVqXQqTkJLpFpZrP980rq1S_deFWWiZPihXqkzTsFwUTl2DWMyuTxwsQnCBi-Dbt11XY-1kXe7uABqL9L6YbaclohAUVQF1pIVONbELTkTzUqfzqf30TLsxT67xScvd51-smHTDBYyvSDjzWLQ-HCl9gMJlZ43NcJtA9fckE47yf7O5YaaaA3QXhNSO86jiDvJ_spVijZXi62P_d4d71fCpUZTum1WSwRAiF5scvIjwFMSL98'
accountId = os.getenv('ACCOUNT_ID') or 'df662a60-74f5-4f40-a356-622e3f20c88d'
#accountId = os.getenv('ACCOUNT_ID') or 'fe52db4a-44df-461a-847e-6325e62ab55d'

symbol_list = [
    'XAUUSDm',  # Gold/US Dollar (Commodity)


]

data = {}
timeframe='5m'

def extract_simulation_values(currency_pair_file):
    # Load the .docx file
    doc = docx.Document(currency_pair_file)
    
    # Initialize variables to store extracted values
    best_sma_10 = None
    best_sma_30 = None
    best_rsi_period = None
    
    # Iterate through each line in the document
    for para in doc.paragraphs:
        line = para.text.strip()
        
        # Extract values based on specific keywords
        if "best_sma_10" in line:
            best_sma_10 = line.split(":")[1].strip()
        elif "best_sma_30" in line:
            best_sma_30 = line.split(":")[1].strip()
        elif "Best rsi_period" in line:
            best_rsi_period = line.split(":")[1].strip()
    
    return best_sma_10, best_sma_30, best_rsi_period


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

def prepare(df):
    #df=add_stop_losse(df)
    df=df.drop(columns=['symbol','timeframe','brokerTime'])

    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].astype(int)// 10**99 
    # Generate new features and handle data preparation
    df_new =df

    return df_new
async def main2():
    print('Up and runing')
    api = MetaApi(token)
    account = await api.metatrader_account_api.get_account(accountId)
    initial_state = account.state
    deployed_states = ['DEPLOYING', 'DEPLOYED']

    if initial_state not in deployed_states:
        # wait until account is deployed and connected to broker
        print('Deploying account')
        await account.deploy()
        print('Waiting for API server to connect to broker (may take a few minutes)')
        await account.wait_connected()
    # Connect to MetaApi API
    connection = account.get_rpc_connection()
    await connection.connect()

    # Wait until terminal state synchronized to the local state
    print('Waiting for SDK to synchronize to terminal state (may take some time depending on your history size)')
    await connection.wait_synchronized()
    
    for symbol in symbol_list:
        print(symbol)
        trades =await connection.get_positions()#connection.get_orders()
        if len(trades)>=40:
            print(f'There are more than 10 runing trades, Total is :{len(trades)}')

        else:
            #sentiment_results = float(cal_sentiment_textblob(symbol))
            #if sentiment_results>0.25 or sentiment_results<0.25:
            try:
                try:
                    # Fetch historical price data
                    candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=50)

                    print('Fetched the latest candle data successfully')
                except Exception as e:
                    raise e
                try:
                    if not isinstance(candles, str):
                        df=pd.DataFrame(candles)
                    else:
                        
                        df=pd.DataFrame()
                except Exception as e:
                    raise e

                if not df.empty:
                
                    df_new=prepare(df)
                    currency_pairs = [
                        "GBPUSDm0.001_simulation_analysis.docx",
                        "EURUSDm0.0007_simulation_analysis.docx",
                        "BTCUSDm600_simulation_analysis.docx",
                        "XAUUSDm4_simulation_analysis.docx",
                        "GBPAUDm0.0015_simulation_analysis.docx"
                    ]


                    # Loop through each currency pair file
                    for pair_file in currency_pairs:
                        if symbol==pair_file[:7]:
                            added_lag=pair_file[:-25]
                            added_lag=float(added_lag[7:])

                            full_path = os.path.join('', pair_file)
                    
                    # Extract the values from the file
                    best_sma_10, best_sma_30, best_rsi_period = extract_simulation_values(full_path)
                    df_new['RSI'] = ta.rsi(df_new['close'], length=int(best_rsi_period))
                    df_new['sma_10'] = ta.sma(df['close'], length=int(best_sma_10))
                    df_new['sma_30'] = ta.sma(df['close'], length=int(best_sma_30))
                    # Buy Signal: When sma_10 crosses above sma_30
                    df_new['buy_signal'] = ((df_new['sma_10'] > df_new['sma_30']) & 
                                            (df_new['sma_10'].shift(1) <= df_new['sma_30'].shift(1))).astype(int)

                    # Sell Signal: When sma_10 crosses below sma_30
                    df_new['sell_signal'] = ((df_new['sma_10'] < df_new['sma_30']) & 
                                             (df_new['sma_10'].shift(1) >= df_new['sma_30'].shift(1))).astype(int)
                    predictions_features=np.array(df_new.iloc[-1]).reshape(1, -1)
                    print(df_new)
                    model,_= joblib.load(f'Regressors/{timeframe}/{symbol}model.pkl')

                    prediction=model.predict(predictions_features)
                    previous_close= df_new['close'].iloc[-1]
                    sma_10=df_new['sma_10'].iloc[-1]
                    sma_30=df_new['sma_30'].iloc[-1]
                    rsi=df_new['RSI'].iloc[-1]
                    stop_loss_away=prediction[0][1]
                    next_close=prediction[0][0]
                    print(stop_loss_away, next_close)
                    prices = await connection.get_symbol_price(symbol)
                    print(prices)
                    # Extract bid and ask prices
                    bid_price =float(prices['bid'])
                    ask_price = float(prices['ask'])
                    current_market_price=((bid_price+ask_price)/2)
                    current_open=current_market_price
                    if next_close>previous_close and rsi<50 and sma_10>sma_30 and (next_close-current_open)>(added_lag*2):
                        stop_loss=current_open-stop_loss_away
                        #take_profit=trademax-(lag_size/2)
                        try:
                            
                            result = await connection.create_market_buy_order(
                                symbol=symbol,
                                volume=0.01,
                                stop_loss=stop_loss,
                                take_profit=next_close,
                            )
                            print(f'Buy_Signal (T)   :Buy Trade successful For Symbol :{symbol}')
                            
                            Trader_success=True
                        except Exception as err:
                            print('Trade failed with error:')
                            print(api.format_error(err))
                    elif next_close<previous_close and rsi>50 and sma_10<sma_30  and (current_open-next_close)>(added_lag*2):
                        stop_loss=current_open+stop_loss_away
                        #take_profit=trademax+(lag_size/2)
                        try:
                            
                            result = await connection.create_market_sell_order(
                                symbol=symbol,
                                volume=0.01,
                                stop_loss=stop_loss,
                                take_profit=next_close,
                            )
                            print(f'Sell Signal (T)   :Sell Trade successful For Symbol :{symbol}')
                            Trader_success=True

                        except Exception as err:
                            #raise err
                            print('Trade failed with error:')
                            print(api.format_error(err))
                    else:
                        print('No trade conditions passed, so no trade placed')
                print('*'*20)
                print('*'*20)
                print('*'*20)
            except Exception as e:
                raise e
                print(f"An error occurred: {e}")
#def main():
asyncio.run(main2())