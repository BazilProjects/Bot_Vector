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
accountId = os.getenv('ACCOUNT_ID') or '7416410e-1803-4778-bead-73b66d695bb5'
#symbol_list =['EURUSDm', 'GBPUSDm','AUDCHFm', 'NZDUSDm','GBPTRYm','XAUUSDm','XAGUSDm',]

symbol_list = [
    'EURUSDm',  # Euro/US Dollar (Major)
    'GBPUSDm',  # British Pound/US Dollar (Major)
    'AUDCHFm',  # Australian Dollar/Swiss Franc (Minor)
    'NZDUSDm',  # New Zealand Dollar/US Dollar (Major)
    'GBPTRYm',  # British Pound/Turkish Lira (Exotic)
    'XAUUSDm',  # Gold/US Dollar (Commodity)
    'XAGUSDm',  # Silver/US Dollar (Commodity)
    'USDCHFm',  # US Dollar/Swiss Franc (Major)
    'AUDUSDm',  # Australian Dollar/US Dollar (Major)
    #'EURGBPm',  # Euro/British Pound (Minor)
    'GBPCHFm',  # British Pound/Swiss Franc (Minor)
    #'AUDJPYm',  # Australian Dollar/Japanese Yen (Minor)
    #'AUDNZDm',  # Australian Dollar/New Zealand Dollar (Minor)
    ##'EURCHFm',  # Euro/Swiss Franc (Minor)
    'EURAUDm',  # Euro/Australian Dollar (Minor)
    'EURCADm',  # Euro/Canadian Dollar (Minor)
    'GBPAUDm' ,  # British Pound/Australian Dollar (Minor)
    'BTCUSDm'

]

"""

"""
timeframe='15m'
pages=7
n_estimators=1
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
    #df[f'close_dif{column1}']=df[column1]//df[column1].min()
    #df[f'close_dif{column2}']=df[column2]//df[column2].min()
    # Multiplication/Division
    df[f'{column1}_times_{column2}'] = df[column1] * df[column2]
    df[f'{column1}_div_{column2}'] = df[column1] / df[column2]

    # Mean and Standard Deviation
    df[f'mean_column_{column1}{column2}'] = df[[column1, column2]].mean(axis=1)
    df[f'std_column_{column1}{column2}'] = df[[column1, column2]].std(axis=1)

    # Sum
    df[f'sum_column_{column1}{column2}'] = df[[column1, column2]].sum(axis=1)

    # Logarithm and Exponential
    

    df[column1] = df[column1].apply(lambda x: x if x > 1e-40 else 1e-40)  # Replace non-positive values
    df[column2] = df[column2].apply(lambda x: x if x > 1e-40 else 1e-40)  # Replace non-positive values

    # Trigonometric Functions
   
    df[f'log_column1_{column1}{column2}'] = np.log(df[column1])
    df[f'exp_column1_{column1}{column2}'] = np.exp(df[column1])

    # Square Root
    df[f'sqrt_column1_{column1}{column2}'] = np.sqrt(df[column1])
    #df[column2] = df[column2].apply(lambda x: x if x > 1e-30 else 1e-30)
    #df[f'sqrt_column1_2{column1}{column2}'] = np.sqrt(df[column2])

    # Rolling Mean and Cumulative Sum
    df[f'rolling_mean_column1_{column1}{column2}'] = df[column1].rolling(window=3).mean()
    df[f'cumulative_sum_column1_{column1}{column2}'] = df[column1].cumsum()

    #df[f'rolling_mean_column1_2{column1}{column2}'] = df[column2].rolling(window=3).mean()
    #df[f'cumulative_sum_column1_2{column1}{column2}'] = df[column2].cumsum()

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
    df[f'lagged_column2_{column1}{column2}'] = df[column2].shift(1)
    #df[f'lagged_column3_{column1}{column2}'] = df[column1].shift(2)

    # Custom Function Example
    df[f'custom_column1_{column1}{column2}'] = df[column1].apply(lambda x: x * 2)
    #df[f'custom_column2_{column1}{column2}'] = df[column2].apply(lambda x: x * 2)
     # Adding trigonometric function results to DataFrame
    df[f'sin_close_{column1}{column2}'] =  np.sin(df[column1])
    df[f'sin_open_{column1}{column2}'] = np.sin(df[column2])

    df[f'cos_close_{column1}{column2}'] = np.cos(df[column1])
    df[f'cos_open_{column1}{column2}'] = np.cos(df[column2])

    df[f'tan_close_{column1}{column2}'] = np.tan(df[column1])
    df[f'tan_open_{column1}{column2}'] = np.tan(df[column2])



    # Adding hyperbolic function results to DataFrame
    df[f'sinh_close_{column1}{column2}'] = np.sinh(df[column1])
    df[f'sinh_open_{column1}{column2}'] = np.sinh(df[column2])

    df[f'cosh_close_{column1}{column2}'] = np.cosh(df[column1])
    df[f'cosh_open_{column1}{column2}'] = np.cosh(df[column2])

    df[f'tanh_close_{column1}{column2}'] = np.tanh(df[column1])
    df[f'tanh_open_{column1}{column2}'] = np.tanh(df[column2])

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


# Function to add nested dictionary content to a .docx file
def add_nested_dict_to_docx(info_dict, filename=f'output{pages}{timeframe}.docx'):
    doc = Document()
    doc.add_heading('AI analysis and market forecast For timeframe', 0)
    for symbol, details in info_dict.items():
        doc.add_heading(symbol, level=1)
        for key, value in details.items():
            doc.add_paragraph(f"{key}: {value}", style='List Number')
    doc.save(filename)
    print(f"Information added to {filename} successfully.")


def add_stop_losse(df):
    pass
    # Initialize new columns
    df['type'] = None
    df['sell_high'] = None
    df['buy_low'] = None

    # Determine the candle type dynamically
    for i in range(df.index[1], len(df)):
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
    df['return'] = df['close'].pct_change()
    df['volatility'] = df['return'].rolling(window=10).std()
    df['moving_avg'] = df['close'].rolling(window=10).mean()
    # Drop the individual Sell_High and Buy_Low columns
    #df.drop(columns=['sell_high', 'buy_low',], inplace=True)
    return df
# Function to add data to the dictionary
def add_symbol_data(symbol, r2_2, mse_2, r2_2_1, mse_2_1, r2_2_2, mse_2_2, next_close, next_low, next_high, actual_close, actual_low, actual_high, diff2, diff1):
    if symbol not in data:
        data[symbol] = {}
    
    data[symbol] = {
        'r2_2': r2_2,
        'mse_2': mse_2,
        'r2_2_1': r2_2_1,
        'mse_2_1': mse_2_1,
        'r2_2_2': r2_2_2,
        'mse_2_2': mse_2_2,
        'next_close': next_close,
        'next_low': next_low,
        'next_high': next_high,
        'actual': {
            'close': actual_close,
            'low': actual_low,
            'high': actual_high
        },
        'diff2': diff2,
        'diff1': diff1
    }

def prepare(df):
    df=add_stop_losse(df)
    df=df.drop(columns=['symbol','timeframe','brokerTime'])

    df['time'] = pd.to_datetime(df['time'])
    df['time'] = df['time'].astype(int) #// 10**99 
    # Generate new features and handle data preparation
    df_new =df
    #print(df_new)
    #print(df.columns)
    # Columns to consider
    columns = ['high', 'low', 'close', 'open', 'tickVolume','time','stop_losses','trade_max','sell_high','sell_low','buy_low','buy_high']

    # Generate combinations of 2 pairs
    combinations_2 = list(combinations(columns, 2))

    # Print all combinations
    for comb in combinations_2:
        first_member, second_member = comb
        df_new = generate_new_features(df_new,first_member,second_member)
    
    time=df_new['time']
    df_new = df_new.dropna()  # Drop NaN values

    # Round numerical columns based on decimal places of the last 'close' value
    decimal_places = lambda number: len(str(number).split('.')[1])
    #df_new = df_new.round(decimal_places(df['close'].iloc[-1]))

    # Check for infinite values in each column
    inf_mask = df_new.isin([np.inf, -np.inf])

    # Drop columns with infinite values
    df_new = df_new.loc[:, ~inf_mask.any()]

    # Set a threshold for numerical columns
    threshold = 1e40  # Adjust the threshold as needed

    # Filter columns where all values are greater than the threshold
    columns_to_drop = df_new.select_dtypes(include=np.number).columns[(df_new.select_dtypes(include=np.number) > threshold).all()]
    #df_new['time']=time
    #print(columns_to_drop)
    #print(df_new.drop(columns=['Prediction']))
    df_new.drop(columns=columns_to_drop)
    return df_new
# Define a threshold value
threshold = 1e-40

# Define a function to replace values below the threshold and handle infinity
def replace_values(x):
    if isinstance(x, (int, float)):
        if x <= threshold:
            return threshold
        elif x == np.inf:
            return np.finfo(np.float64).max
        elif x == -np.inf:
            return -np.finfo(np.float64).max
    return x

# Apply the function only to columns of type int or float
def prepare_30m(candles):
    
    df=pd.DataFrame(candles)
    df_new=prepare(df)
    #print(df_new[['Prediction','trade_max','stop_losses','stop_losses_predictions']])
    df_new = df_new.apply(lambda col: col.map(replace_values) if col.dtype in [np.int64, np.float64] else col)

    
    X =df_new
    print(len(X.columns))
    #print(X.columns)
    column_list=[]
    for column in X.columns:
        try:
            if str(column[:6])=='binned':
                column_list.append(column)
        except Exception as e:
            pass
    for i in ['type']:
        column_list.append(i)

    
    #print(last_row)
    # One-hot encode the 'binned_column1' feature
    transformer = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(), column_list)],
        remainder='passthrough'
    )
    X = transformer.fit_transform(X)
    last_row = np.array(X[-1]).reshape(1, -1)
    return last_row

async def main2(timeframe,pages):
    
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
        if len(trades)>=10:
            pass

        else:
            try:
                try:
                    # Fetch historical price data
                    candles = await account.get_historical_candles(symbol=symbol, timeframe=timeframe, start_time=None, limit=300)

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
                    #print(df)
                    df_new=prepare(df)
                    #print(df_new[['Prediction','trade_max','stop_losses','stop_losses_predictions']])
                    df_new = df_new.apply(lambda col: col.map(replace_values) if col.dtype in [np.int64, np.float64] else col)

                    
                    X =df_new
                    print(len(X.columns))
                    #print(X.columns)
                    column_list=[]
                    for column in X.columns:
                        try:
                            if str(column[:6])=='binned':
                                column_list.append(column)
                        except Exception as e:
                            pass
                    for i in ['type']:
                        column_list.append(i)
 
                    
                    #print(last_row)
                    # One-hot encode the 'binned_column1' feature
                    transformer = ColumnTransformer(
                        transformers=[('cat', OneHotEncoder(), column_list)],
                        remainder='passthrough'
                    )
                    X = transformer.fit_transform(X)
                    last_row = np.array(X[-1]).reshape(1, -1)
                    #print(len(last_row.columns))

                    model_close, sklearn_version = joblib.load(f'Regressors/Close/model{symbol}{timeframe}close.pkl')
                    model_low,_ = joblib.load(f'Regressors/Low/model{symbol}{timeframe}low.pkl')
                    model_high,_ = joblib.load(f'Regressors/High/model{symbol}{timeframe}high.pkl')


                    classifiers_15m, _= joblib.load(f'Classifiers/15m/ExtraTrees{symbol}.pkl')
                    classifiers_30m,_ = joblib.load(f'Classifiers/30m/ExtraTrees{symbol}.pkl')
                    """
                    trained_feature_names = model_close.feature_names_in_

                    # Get the feature names from the prediction data
                    prediction_feature_names = last_row.columns

                    # Identify extra features
                    extra_features = set(prediction_feature_names) - set(trained_feature_names)

                    # Drop extra features from the prediction data
                    last_row = last_row.drop(columns=extra_features)
                    
                    # Now you can safely use the cleaned prediction data
                    next_close = model_close.predict(prediction_data_cleaned)
                    """

                    next_close = model_close.predict(last_row)
                    next_low = model_low.predict(last_row)
                    next_high = model_high.predict(last_row)

                    classifiers_15m_pred=classifiers_15m.predict(last_row)[0]
                    candles = await account.get_historical_candles(symbol=symbol, timeframe='30m', start_time=None, limit=300)
                    last_row_30m=prepare_30m(candles)
                    classifiers_30m_pred=classifiers_30m.predict(last_row_30m)[0]
                    """
                    if classifiers_15m_pred==0:
                        classifiers_15m_pred_proba=classifiers_15m.predict_proba(last_row)[0][0]
                    else:
                        classifiers_15m_pred_proba=classifiers_15m.predict_proba(last_row)[0][1]
                    if classifiers_30m_pred==0:
                        classifiers_30m_pred_proba=classifiers_30m.predict_proba(last_row)[0][0]
                    else:
                        classifiers_30m_pred_proba=classifiers_30m.predict_proba(last_row)[0][1]
                    """
                    # Get predicted probabilities for both classifiers
                    classifiers_15m_pred_proba = classifiers_15m.predict_proba(last_row)[0][classifiers_15m_pred]
                    classifiers_30m_pred_proba = classifiers_30m.predict_proba(last_row_30m)[0][classifiers_30m_pred]

                    next_close=round(next_close[0],decimal_places(df['close'].iloc[-1]))
                    next_low=round(next_low[0],decimal_places(df['close'].iloc[-1]))
                    next_high=round(next_high[0],decimal_places(df['close'].iloc[-1]))


                    print(f"Next predicted close price: {next_close}")
                    print(f"Next predicted low price: {next_low}")
                    print(f"Next predicted high price: {next_high}")

                    previous_close= df_new['close'].iloc[-1]
                    lag_size=0.0004
                    

                    if (classifiers_15m_pred==classifiers_30m_pred):
                        if (classifiers_15m_pred==1) and (classifiers_15m_pred_proba>0.9) and (classifiers_30m_pred_proba>0.9):
                            if (next_close > previous_close and
                                (next_close>next_low) and
                                (next_close - previous_close) > lag_size) and (next_close<next_high):
                                stop_loss=None#next_low-(lag_size*4)
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
                        elif (classifiers_15m_pred==0) and (classifiers_15m_pred_proba>0.9) and (classifiers_30m_pred_proba>0.9):
                            if (next_close<previous_close and 
                                (next_close<next_high) and
                                (previous_close-next_close)>lag_size) and (next_close<next_high):
                                stop_loss=None#(next_high+lag_size*4)
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
                #print(f'{symbol} failed')
                #raise e
                pass
def main():
    asyncio.run(main2(timeframe,pages))