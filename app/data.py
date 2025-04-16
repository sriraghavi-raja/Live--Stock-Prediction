import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ta
from apscheduler.schedulers.background import BackgroundScheduler
import pytz

# Cache for storing data to reduce API calls
data_cache = {}
cache_expiry = {}
CACHE_DURATION = 300  

def get_stock_data(ticker, period="1d", interval="1m"):
    """
    Fetch stock data with caching mechanism to avoid rate limiting
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        interval (str): Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
    
    Returns:
        pd.DataFrame: DataFrame containing stock data
    """
    cache_key = f"{ticker}_{period}_{interval}"
    current_time = datetime.now()
    
   
    if cache_key in data_cache and cache_expiry[cache_key] > current_time:
        return data_cache[cache_key]
    
    try:
        stock = yf.Ticker(ticker)
        
        # For intraday data, use history method
        if interval in ["1m", "2m", "5m", "15m", "30m", "60m", "90m", "1h"]:
           
            if interval == "1m" and period not in ["1d", "5d", "7d"]:
                period = "1d" 
                
            df = stock.history(period=period, interval=interval)
        else:
           
            if period == "max":
                df = stock.history(period=period)
            else:
                df = stock.history(period=period, interval=interval)
        
        # Add technical indicators
        if not df.empty:
            df = add_technical_indicators(df)
        
        # Cache the result
        data_cache[cache_key] = df
        cache_expiry[cache_key] = current_time + timedelta(seconds=CACHE_DURATION)
        
        return df
    
    except Exception as e:
        print(f"Error fetching stock data for {ticker}: {e}")
        return pd.DataFrame()

def add_technical_indicators(df):
    """Add technical indicators to the dataframe"""
    if len(df) < 14:  # Need at least 14 data points for some indicators
        return df
    
    # Moving Averages
    df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20, fillna=True)
    df['SMA_50'] = ta.trend.sma_indicator(df['Close'], window=50, fillna=True)
    df['EMA_12'] = ta.trend.ema_indicator(df['Close'], window=12, fillna=True)
    df['EMA_26'] = ta.trend.ema_indicator(df['Close'], window=26, fillna=True)
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    # RSI
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14, fillna=True)
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bollinger.bollinger_hband()
    df['BB_Low'] = bollinger.bollinger_lband()
    df['BB_Mid'] = bollinger.bollinger_mavg()
    
    # Volume indicators
    df['Volume_SMA'] = ta.trend.sma_indicator(df['Volume'], window=20, fillna=True)
    
    # Average True Range (ATR)
    df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14, fillna=True)
    
    return df

def get_stock_info(ticker):
    """Get fundamental information about a stock"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Extract relevant information
        stock_info = {
            'name': info.get('shortName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'industry': info.get('industry', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'eps': info.get('trailingEps', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A'),
            'target_price': info.get('targetMeanPrice', 'N/A'),
            'recommendation': info.get('recommendationKey', 'N/A'),
            'description': info.get('longBusinessSummary', 'No description available.')
        }
        
        # Format market cap
        if isinstance(stock_info['market_cap'], (int, float)) and stock_info['market_cap'] != 'N/A':
            if stock_info['market_cap'] >= 1e12:
                stock_info['market_cap'] = f"${stock_info['market_cap']/1e12:.2f}T"
            elif stock_info['market_cap'] >= 1e9:
                stock_info['market_cap'] = f"${stock_info['market_cap']/1e9:.2f}B"
            elif stock_info['market_cap'] >= 1e6:
                stock_info['market_cap'] = f"${stock_info['market_cap']/1e6:.2f}M"
                
        # Format dividend yield as percentage
        if isinstance(stock_info['dividend_yield'], (int, float)) and stock_info['dividend_yield'] != 'N/A':
            stock_info['dividend_yield'] = f"{stock_info['dividend_yield']*100:.2f}%"
        
        return stock_info
    
    except Exception as e:
        print(f"Error fetching stock info for {ticker}: {e}")
        return {
            'name': ticker,
            'sector': 'N/A',
            'industry': 'N/A',
            'market_cap': 'N/A',
            'pe_ratio': 'N/A',
            'eps': 'N/A',
            'dividend_yield': 'N/A',
            'target_price': 'N/A',
            'recommendation': 'N/A',
            'description': 'No description available.'
        }

def get_price_prediction(df, forecast_periods=5):
    """Simple price prediction based on ARIMA-like approach"""
    if df.empty or len(df) < 30:
        return None
    
    try:
       
        close_prices = df['Close'].values
        
        # returns
        returns = np.diff(close_prices) / close_prices[:-1]
        
        # exponentially weighted average return
        alpha = 0.3  
        weighted_return = returns[-20:] * np.power(1-alpha, np.arange(20))
        avg_return = np.sum(weighted_return) / np.sum(np.power(1-alpha, np.arange(20)))
        
        # Generate predictions
        last_price = close_prices[-1]
        predictions = []
        
        for i in range(1, forecast_periods + 1):
            next_price = last_price * (1 + avg_return)
            predictions.append(next_price)
            last_price = next_price
        
        # dates for predictions
        last_date = df.index[-1]
        dates = []
        

        if isinstance(last_date, pd.Timestamp) and last_date.minute != 0:
            # Intraday data
            for i in range(1, forecast_periods + 1):
                next_date = last_date + timedelta(minutes=i)
                # Only include trading hours (9:30 AM - 4:00 PM)
                while next_date.hour < 9 or (next_date.hour == 9 and next_date.minute < 30) or next_date.hour > 16:
                    next_date += timedelta(minutes=1)
                dates.append(next_date)
        else:
            # Daily data
            for i in range(1, forecast_periods + 1):
                dates.append(last_date + timedelta(days=i))
        
        return pd.DataFrame({'Date': dates, 'Predicted_Close': predictions})
    
    except Exception as e:
        print(f"Error generating price prediction: {e}")
        return None

def create_candlestick_chart(df, ticker, indicators=None):
    """Create an interactive candlestick chart with indicators"""
    if df.empty:
        return None
    
   
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.7, 0.3],
                        subplot_titles=(f'{ticker} Stock Price', 'Volume'))
    

    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Price',
            increasing_line_color='#26a69a', 
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # indicators if specified
    if indicators:
        if 'sma' in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='blue', width=1), name='SMA 20'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'), row=1, col=1)
        
        if 'ema' in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_12'], line=dict(color='purple', width=1), name='EMA 12'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['EMA_26'], line=dict(color='green', width=1), name='EMA 26'), row=1, col=1)
        
        if 'bb' in indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_High'], line=dict(color='rgba(0,0,255,0.3)', width=1), name='BB Upper'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Low'], line=dict(color='rgba(0,0,255,0.3)', width=1), name='BB Lower'), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['BB_Mid'], line=dict(color='rgba(0,0,255,0.5)', width=1), name='BB Middle'), row=1, col=1)
    
    #  volume bar chart
    colors = ['#26a69a' if df['Close'][i] >= df['Open'][i] else '#ef5350' for i in range(len(df))]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['Volume'],
            marker_color=colors,
            name='Volume'
        ),
        row=2, col=1
    )
    
    # volume moving average
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df['Volume_SMA'],
            line=dict(color='blue', width=1.5),
            name='Volume SMA'
        ),
        row=2, col=1
    )
    
  
    fig.update_layout(
        height=600,
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_dark"
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig

def create_technical_indicator_charts(df, ticker):
    """Create charts for technical indicators"""
    if df.empty or len(df) < 14:
        return None
    
    #  figure with secondary y-axis
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.1, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=(f'{ticker} Technical Indicators', 'MACD', 'RSI'))
    
    # price line to first subplot
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Close'], name='Close Price', line=dict(color='#26a69a')),
        row=1, col=1
    )
    
    #EMA lines
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_12'], name='EMA 12', line=dict(color='purple')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['EMA_26'], name='EMA 26', line=dict(color='orange')),
        row=1, col=1
    )
    
    #  Bollinger Bands
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_High'], name='BB Upper', line=dict(color='rgba(173, 216, 230, 0.5)')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Low'], name='BB Lower', line=dict(color='rgba(173, 216, 230, 0.5)')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['BB_Mid'], name='BB Middle', line=dict(color='rgba(173, 216, 230, 0.8)')),
        row=1, col=1
    )
    
    #  MACD 
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='#26a69a')),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['MACD_Signal'], name='Signal', line=dict(color='#ef5350')),
        row=2, col=1
    )
    
    #  MACD histogram
    colors = ['#26a69a' if val >= 0 else '#ef5350' for val in df['MACD_Hist']]
    fig.add_trace(
        go.Bar(x=df.index, y=df['MACD_Hist'], name='Histogram', marker_color=colors),
        row=2, col=1
    )
    
   
    fig.add_trace(
        go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='#26a69a')),
        row=3, col=1
    )
    
    # RSI reference 
    fig.add_shape(
        type="line", line_color="red", line_width=1, opacity=0.5, line_dash="dash",
        x0=df.index[0], x1=df.index[-1], y0=70, y1=70,
        xref="x3", yref="y3"
    )
    fig.add_shape(
        type="line", line_color="green", line_width=1, opacity=0.5, line_dash="dash",
        x0=df.index[0], x1=df.index[-1], y0=30, y1=30,
        xref="x3", yref="y3"
    )
    
    
    fig.update_layout(
        height=700,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0),
        template="plotly_dark"
    )
    
   
    fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    
    return fig

def get_stock_performance_summary(df):
    """Calculate performance metrics for a stock"""
    if df.empty:
        return {}
    
    try:
        #  daily returns
        df['Daily_Return'] = df['Close'].pct_change() * 100
        
        # Get current price and calculate changes
        current_price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[-2] if len(df) > 1 else None
        day_change = current_price - prev_close if prev_close else None
        day_change_pct = (day_change / prev_close * 100) if prev_close else None
        
        # price range
        period_high = df['High'].max()
        period_low = df['Low'].min()
        
        #volatility (standard deviation of returns)
        volatility = df['Daily_Return'].std()
        
        # average volume
        avg_volume = df['Volume'].mean()
        
        # Technical signals
        rsi = df['RSI'].iloc[-1] if 'RSI' in df.columns else None
        macd = df['MACD'].iloc[-1] if 'MACD' in df.columns else None
        macd_signal = df['MACD_Signal'].iloc[-1] if 'MACD_Signal' in df.columns else None
        
        # Determine signals
        rsi_signal = "Overbought" if rsi and rsi > 70 else "Oversold" if rsi and rsi < 30 else "Neutral"
        macd_signal_val = "Bullish" if macd and macd_signal and macd > macd_signal else "Bearish" if macd and macd_signal and macd < macd_signal else "Neutral"
        
        # Trend based on moving averages
        trend = "Uptrend" if 'SMA_20' in df.columns and 'SMA_50' in df.columns and df['SMA_20'].iloc[-1] > df['SMA_50'].iloc[-1] else \
                "Downtrend" if 'SMA_20' in df.columns and 'SMA_50' in df.columns and df['SMA_20'].iloc[-1] < df['SMA_50'].iloc[-1] else "Neutral"
        
        summary = {
            'current_price': current_price,
            'day_change': day_change,
            'day_change_pct': day_change_pct,
            'period_high': period_high,
            'period_low': period_low,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'rsi': rsi,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal_val,
            'trend': trend
        }
        
        return summary
    
    except Exception as e:
        print(f"Error calculating performance summary: {e}")
        return {}

def clear_cache():
    """Clear expired items from cache"""
    current_time = datetime.now()
    keys_to_remove = [key for key, expiry_time in cache_expiry.items() if expiry_time <= current_time]
    
    for key in keys_to_remove:
        if key in data_cache:
            del data_cache[key]
        if key in cache_expiry:
            del cache_expiry[key]

# Set up scheduler to clear cache periodically
scheduler = BackgroundScheduler(timezone=pytz.utc)
scheduler.add_job(clear_cache, 'interval', minutes=5)
scheduler.start()

