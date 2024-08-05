import sklearn
from datetime import datetime, timedelta
from textblob import TextBlob
import yfinance as yf
import pandas as pd

def cal_sentiment_textblob(symbol):
    total_sentiment = 0
    
    try:
        # Retrieve news headlines using yfinance
        stock = yf.Ticker(symbol)
        news_df = stock.news
        news_df = pd.DataFrame(news_df)
        print(news_df)
        # Sort news by publication time in descending order (handling potential exceptions)
        try:
            news_df = news_df.sort_values('providerPublishTime', ascending=False)
        except KeyError:
            news_df = news_df.sort_values('uuid', ascending=False)
        
        print(f"Sentiment analysis for symbol: {symbol}")
        print("--------------------------------------")
        print(news_df)
        # Perform sentiment analysis on the latest news headlines
        for index, row in news_df.iterrows():
            headline = row['title']
            
            # Perform sentiment analysis using TextBlob
            blob = TextBlob(headline)
            sentiment = blob.sentiment.polarity
            total_sentiment += sentiment
            
            # Print each headline and its sentiment score
            print(headline)
            print(f"Sentiment Score: {sentiment}")
        
    except Exception as e:
        raise e
        print(f"Error occurred: {str(e)}")
        return total_sentiment
    
    return total_sentiment
symbol='BTCUSDm'

sentiment_results = float(cal_sentiment_textblob(symbol))
print(sentiment_results)