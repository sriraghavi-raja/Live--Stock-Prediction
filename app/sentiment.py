import pandas as pd
import numpy as np
import re
import json
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
import time
from functools import lru_cache

# Cache for Twitter sentiment data to reduce API calls
sentiment_cache = {}
cache_expiry = {}
CACHE_DURATION = 1800  # 30 minutes

@lru_cache(maxsize=100)
def clean_tweet(tweet):
    """
    Clean tweet text by removing links, special characters
    """
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_tweet_sentiment(tweet):
    """
    Classify sentiment of passed tweet using TextBlob
    """
    # Clean the tweet
    cleaned_tweet = clean_tweet(tweet)
    
    # Create TextBlob object
    analysis = TextBlob(cleaned_tweet)
    
    # Return sentiment polarity: -1 to 1
    return analysis.sentiment.polarity

def get_tweets_sentiment(ticker, count=100):
    """
    Get sentiment score for tweets about a stock ticker
    
    Args:
        ticker (str): Stock ticker symbol
        count (int): Number of tweets to analyze
    
    Returns:
        float: Average sentiment score between -1 (negative) and 1 (positive)
    """
    current_time = datetime.now()
    cache_key = f"{ticker}_sentiment"
    
    # Return cached sentiment if it exists and is not expired
    if cache_key in sentiment_cache and cache_expiry[cache_key] > current_time:
        return sentiment_cache[cache_key]
    
    try:
        # In a real implementation, this would use Twitter API
        # Since we can't actually connect to Twitter here, we'll simulate the data
        
        # Simulate fetching tweets
        simulated_sentiment = simulate_tweet_sentiment(ticker)
        
        # Cache the result
        sentiment_cache[cache_key] = simulated_sentiment
        cache_expiry[cache_key] = current_time + timedelta(seconds=CACHE_DURATION)
        
        return simulated_sentiment
    
    except Exception as e:
        print(f"Error fetching tweets for {ticker}: {e}")
        # Return neutral sentiment if there's an error
        return 0.0

def simulate_tweet_sentiment(ticker):
    """
    Simulate tweet sentiment for a stock ticker
    This function creates realistic but synthetic sentiment data
    based on the ticker symbol
    """
    # Create a pseudo-random but consistent sentiment for each ticker
    # This ensures the same ticker always gets the same sentiment in a session
    np.random.seed(sum(ord(c) for c in ticker) + int(datetime.now().strftime("%Y%m%d")))
    
    # Base sentiment - slightly positive on average
    base_sentiment = np.random.normal(0.1, 0.3)
    
    # Add some stock-specific bias
    ticker_bias = {
        'AAPL': 0.2,   # Apple tends to have positive sentiment
        'MSFT': 0.15,  # Microsoft tends to have positive sentiment
        'GOOGL': 0.1,  # Google tends to have positive sentiment
        'AMZN': 0.1,   # Amazon tends to have positive sentiment
        'FB': -0.1,    # Facebook tends to have negative sentiment
        'TSLA': 0.3,   # Tesla tends to have very positive sentiment (or negative)
        'NFLX': 0.05,  # Netflix tends to have slightly positive sentiment
        'TWTR': -0.05, # Twitter tends to have slightly negative sentiment
    }
    
    # Apply ticker bias if available
    specific_bias = ticker_bias.get(ticker, 0)
    
    # Calculate final sentiment
    sentiment = base_sentiment + specific_bias
    
    # Ensure sentiment is between -1 and 1
    sentiment = max(-0.95, min(0.95, sentiment))
    
    return sentiment

def get_signal(sentiment_score):
    """
    Convert sentiment score to a trading signal
    
    Args:
        sentiment_score (float): Sentiment score between -1 and 1
    
    Returns:
        str: Trading signal (Strong Buy, Buy, Neutral, Sell, Strong Sell)
    """
    if sentiment_score >= 0.6:
        return "Strong Buy"
    elif sentiment_score >= 0.2:
        return "Buy"
    elif sentiment_score <= -0.6:
        return "Strong Sell"
    elif sentiment_score <= -0.2:
        return "Sell"
    else:
        return "Neutral"

def clear_sentiment_cache():
    """Clear expired items from sentiment cache"""
    current_time = datetime.now()
    keys_to_remove = [key for key, expiry_time in cache_expiry.items() if expiry_time <= current_time]
    
    for key in keys_to_remove:
        if key in sentiment_cache:
            del sentiment_cache[key]
        if key in cache_expiry:
            del cache_expiry[key]