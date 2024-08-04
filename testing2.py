import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, OneHotEncoder
from sklearn.compose import ColumnTransformer
import plotly.graph_objs as go
import plotly.io as pio
from itertools import combinations
from metaapi_cloud_sdk import MetaApi
import asyncio
import joblib
import pandas_ta as ta
import sklearn
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, max_error
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

from datetime import datetime, timedelta
symbol_list =['EURUSDm',]

timeframe='15m'
pages=100
n_estimators=7
min_samples_leaf=1
max_depth=50
threshold=1e-10  

async def main2(timeframe,pages):
    print('Up and runing')
    for symbol in symbol_list:
        try:

            
            df=pd.read_csv(f'COLLECT CANDLES/100m/{symbol}{timeframe}{str(pages)}.csv')
            df=df.drop(columns=['symbol','timeframe','brokerTime'])
            df['time'] = pd.to_datetime(df['time'])
            df['time'] = df['time'].astype(int) #// 10**99
            if not df.empty:
                #decimal_places_found=decimal_places(df['close'].iloc[-1])
                min_degree = df['close'].min()
                max_degree = df['close'].max()
                df['close_normalized'] = (df['close'] - min_degree) / (max_degree - min_degree)
                df_new=df
                #print(df_new)
                dt=pd.DataFrame(df_new)
                
                df_new['Candle_close'] = df_new['close'].shift(-1)
                df_new=df_new.dropna()

                X =df_new.drop(columns=['Candle_close'])
                y= df_new['Candle_close']

                
                
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                

                # Split the data into training and testing sets




                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, shuffle=False)
                
                
                # Define a dictionary of models
                models = {
                    'SVR': SVR(kernel='sigmoid'),
                    'DecisionTree': DecisionTreeRegressor(),
                    'RandomForest': RandomForestRegressor(n_estimators=1, min_samples_leaf=2, max_depth=500, random_state=42),
                    #'LinearRegression': LinearRegression(),
                    'ExtraTrees': ExtraTreesRegressor(n_estimators=1, random_state=42),

                    'gradient_boosting_regressor':GradientBoostingRegressor(n_estimators=1),
                    'hist_gradient_boosting_regressor':HistGradientBoostingRegressor(max_iter=1000),
                    'ada_boost_regressor':AdaBoostRegressor(n_estimators=5),
                    #'bagging_regressor' : BaggingRegressor(base_estimator=DecisionTreeRegressor(), n_estimators=1),

                }

                # Train each model, make predictions, and calculate metrics
                results = {}
                print(symbol)
                for name, model in models.items():
                    # Create a pipeline
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),  # Scaling
                        ('regressor', model)  # Model
                    ])
                    
                    # Train the model
                    pipeline.fit(X_train, y_train)
                    
                    # Predict on the test set
                    y_pred = pipeline.predict(X_test)

                    # Convert the list of predictions to a numpy array
                    y_pred = np.array(y_pred)
                    # Compute metrics
                    mse = mean_squared_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results[name] = {'MSE': mse, 'R^2': r2}
                    
                    #print(f'{name} Mean Squared Error: {mse}')
                    print(f'{name} R^2 Score: {r2}')


                
        except Exception as e:
            #print(f'{symbol} failed')
            raise e
            pass
asyncio.run(main2(timeframe,pages))