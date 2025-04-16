import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import numpy as np
from sentiment import get_tweets_sentiment, get_signal
from data import (
    get_stock_data, 
    get_stock_info, 
    get_price_prediction,
    create_candlestick_chart,
    create_technical_indicator_charts,
    get_stock_performance_summary,
    clear_cache
)

# Page configuration
st.set_page_config(
    page_title="Advanced Stock Analysis Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for better styling
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0;
        color: #ffffff;
    }
    .subheader {
        font-size: 1.2rem;
        font-weight: 400;
        color: #a3a8b8;
        margin-top: 0;
    }
    .metric-card {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 15px;
        margin: 5px 0;
        border-left: 4px solid #26a69a;
    }
    .metric-card.negative {
        border-left: 4px solid #ef5350;
    }
    .metric-title {
        font-size: 0.9rem;
        font-weight: 600;
        color: #a3a8b8;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        margin: 5px 0;
    }
    .metric-delta {
        font-size: 0.9rem;
    }
    .positive {
        color: #26a69a;
    }
    .negative {
        color: #ef5350;
    }
    .neutral {
        color: #a3a8b8;
    }
    .info-container {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 15px;
        margin: 10px 0;
    }
    .tab-subheader {
        font-size: 1.3rem;
        font-weight: 600;
        margin: 15px 0 5px 0;
    }
    div[data-testid="stHorizontalBlock"] > div {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 15px;
    }
    .sentiment-gauge {
        margin: 10px auto;
        width: 200px;
        height: 200px;
    }
    .stButton>button {
        width: 100%;
        background-color: #26a69a;
        color: white;
        font-weight: 500;
    }
    .chart-container {
        margin-top: 20px;
    }
    .news-card {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #4285f4;
    }
    .news-title {
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .news-meta {
        font-size: 0.8rem;
        color: #a3a8b8;
        margin-bottom: 10px;
    }
    .news-summary {
        font-size: 0.9rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1a1c24;
        border-radius: 4px 4px 0 0;
        gap: 1px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #26a69a;
        color: white;
    }
    .watchlist-card {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 10px;
        margin: 5px 0;
    }
    .prediction-card {
        background-color: #1a1c24;
        border-radius: 6px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #ff9800;
    }
    .prediction-header {
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 10px;
    }
    div[data-testid="stDecoration"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state for storing data
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now() - timedelta(minutes=5)
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
if 'refresh_interval' not in st.session_state:
    st.session_state.refresh_interval = 60  # seconds
if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = False

# Function to check if data refresh is needed
def should_refresh():
    return (datetime.now() - st.session_state.last_update).total_seconds() >= st.session_state.refresh_interval

# Helper function to format percentage changes
def format_change(value):
    if value is None:
        return "N/A"
    return f"{value:.2f}%"

# Helper function to add color based on value
def color_change(value):
    if value is None:
        return "neutral"
    return "positive" if value > 0 else "negative" if value < 0 else "neutral"

# Function to display metrics in cards
def metric_card(title, value, delta=None, prefix="", suffix="", delta_prefix="", delta_suffix=""):
    delta_color = color_change(delta)
    card_class = "metric-card" if delta_color != "negative" else "metric-card negative"
    
    delta_html = ""
    if delta is not None:
        delta_icon = "↑" if delta > 0 else "↓" if delta < 0 else ""
        delta_html = f"""
        <span class="metric-delta {delta_color}">
            {delta_icon} {delta_prefix}{abs(delta):.2f}{delta_suffix}
        </span>
        """
    
    st.markdown(f"""
    <div class="{card_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{prefix}{value}{suffix}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)

# Function to display a news card
def news_card(title, publisher, date, summary, link):
    st.markdown(f"""
    <div class="news-card">
        <div class="news-title">{title}</div>
        <div class="news-meta">
            {publisher} • {date}
        </div>
        <div class="news-summary">
            {summary}
        </div>
        <a href="{link}" target="_blank">Read more</a>
    </div>
    """, unsafe_allow_html=True)

# Function to create a sentiment gauge chart
def create_sentiment_gauge(sentiment_score):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=sentiment_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Twitter Sentiment", 'font': {'size': 16, 'color': 'white'}},
        gauge={
            'axis': {'range': [-1, 1], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [-1, -0.5], 'color': '#ef5350'},
                {'range': [-0.5, -0.3], 'color': '#ff9800'},
                {'range': [-0.3, 0.3], 'color': '#9e9e9e'},
                {'range': [0.3, 0.5], 'color': '#8bc34a'},
                {'range': [0.5, 1], 'color': '#26a69a'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.8,
                'value': sentiment_score
            }
        }
    ))
    
    fig.update_layout(
        height=230,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': 'white'},
    )
    
    return fig

# Function to display a watchlist item
def watchlist_item(ticker):
    try:
        df = get_stock_data(ticker, period="1d", interval="5m")
        if df.empty:
            st.warning(f"No data available for {ticker}")
            return
        
        info = get_stock_info(ticker)
        price = df['Close'].iloc[-1]
        prev_close = df['Close'].iloc[0]
        change = price - prev_close
        change_pct = (change / prev_close) * 100
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write(f"**{info['name']}** ({ticker})")
        with col2:
            # Mini sparkline chart
            fig = px.line(df, x=df.index, y='Close', height=50)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False, zeroline=False),
                yaxis=dict(showticklabels=False, showgrid=False, zeroline=False)
            )
            fig.update_traces(line_color='#26a69a' if change_pct >= 0 else '#ef5350')
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        with col3:
            st.write(f"${price:.2f}")
            st.write(f"<span class='{color_change(change_pct)}'>{format_change(change_pct)}</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading {ticker}: {e}")

# Main app header
st.markdown('<h1 class="main-header">Advanced Stock Analysis Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Real-time market data with advanced technical analysis and sentiment tracking</p>', unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("## Dashboard Settings")
    
    # Stock search
    ticker_input = st.text_input("Enter Stock Ticker Symbol", "AAPL").upper()
    
    # Time period selection
    period_options = {
        "1 Day": "1d",
        "5 Days": "5d",
        "1 Month": "1mo",
        "3 Months": "3mo",
        "6 Months": "6mo",
        "1 Year": "1y"
    }
    selected_period = st.selectbox("Select Time Period", list(period_options.keys()))
    period = period_options[selected_period]
    
    # Time interval selection (adaptive based on period)
    interval_options = {
        "1 Day": ["1m", "2m", "5m", "15m", "30m", "60m"],
        "5 Days": ["5m", "15m", "30m", "60m", "90m"],
        "1 Month": ["30m", "60m", "1d"],
        "3 Months": ["1d", "5d", "1wk"],
        "6 Months": ["1d", "5d", "1wk"],
        "1 Year": ["1d", "5d", "1wk", "1mo"]
    }
    
    interval_labels = {
        "1m": "1 Minute",
        "2m": "2 Minutes",
        "5m": "5 Minutes",
        "15m": "15 Minutes",
        "30m": "30 Minutes",
        "60m": "1 Hour",
        "90m": "90 Minutes",
        "1d": "1 Day",
        "5d": "5 Days",
        "1wk": "1 Week",
        "1mo": "1 Month"
    }
    
    available_intervals = interval_options[selected_period]
    interval_labels_filtered = {k: interval_labels[k] for k in available_intervals}
    selected_interval_label = st.selectbox("Select Time Interval", list(interval_labels_filtered.values()))
    selected_interval = list(interval_labels.keys())[list(interval_labels.values()).index(selected_interval_label)]
    
    # Chart type selection
    chart_indicators = st.multiselect(
        "Select Technical Indicators", 
        ["sma", "ema", "bb", "rsi", "macd"],
        default=["sma", "ema"]
    )
    
    # Data refresh settings
    st.markdown("## Data Refresh Settings")
    refresh_interval_options = {
        "30 seconds": 30,
        "1 minute": 60,
        "5 minutes": 300,
        "15 minutes": 900,
        "30 minutes": 1800
    }
    selected_refresh = st.selectbox(
        "Refresh Interval", 
        list(refresh_interval_options.keys()),
        index=1
    )
    st.session_state.refresh_interval = refresh_interval_options[selected_refresh]
    
    st.session_state.auto_refresh = st.checkbox("Auto Refresh Data", value=st.session_state.auto_refresh)
    
    if st.button("Refresh Now"):
        st.session_state.last_update = datetime.now() - timedelta(minutes=5)
        st.rerun()

    
    # Watchlist management
    st.markdown("## Watchlist")
    new_ticker = st.text_input("Add to Watchlist").upper()
    if new_ticker and st.button("Add"):
        if new_ticker not in st.session_state.watchlist:
            st.session_state.watchlist.append(new_ticker)
    
    if st.button("Clear Watchlist"):
        st.session_state.watchlist = []

# Auto-refresh logic
if st.session_state.auto_refresh and should_refresh():
    st.session_state.last_update = datetime.now()
    clear_cache()  # Clear the cache to fetch fresh data

# Main content area
if ticker_input:
    # Fetch data
    try:
        df = get_stock_data(ticker_input, period=period, interval=selected_interval)
        if df.empty:
            st.error(f"No data found for {ticker_input}. Please check the ticker symbol.")
        else:
            # Get additional data
            stock_info = get_stock_info(ticker_input)
            performance = get_stock_performance_summary(df)
            sentiment_score = get_tweets_sentiment(ticker_input)
            signal = get_signal(sentiment_score)
            
            # Display company info and current stats
            col1, col2, col3 = st.columns([3, 1, 2])
            
            with col1:
                st.markdown(f"# {stock_info['name']} ({ticker_input})")
                st.markdown(f"**Sector:** {stock_info['sector']} | **Industry:** {stock_info['industry']}")
                st.markdown(f"**Market Cap:** {stock_info['market_cap']} | **P/E Ratio:** {stock_info['pe_ratio']}")
            
            with col2:
                st.markdown("### Current Stats")
                current_price = performance.get('current_price', 'N/A')
                day_change_pct = performance.get('day_change_pct', None)
                
                if isinstance(current_price, (int, float)):
                    price_display = f"${current_price:.2f}"
                else:
                    price_display = current_price
                
                metric_card("Price", price_display, delta=day_change_pct, delta_suffix="%")
            
            with col3:
                st.markdown("### Sentiment Analysis")
                st.plotly_chart(create_sentiment_gauge(sentiment_score), use_container_width=True, config={'displayModeBar': False})
                st.markdown(f"<div style='text-align:center; font-size:1.2rem; font-weight:600;' class='{color_change(sentiment_score)}'>{signal}</div>", unsafe_allow_html=True)
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Chart", "📊 Technical Analysis", "🔮 Predictions", "📰 News", "📋 Company Info"])
            
            with tab1:
                # Price chart with selected indicators
                st.markdown("### Price Chart")
                chart = create_candlestick_chart(df, ticker_input, chart_indicators)
                if chart:
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Insufficient data to create chart")
                
                # Key metrics row
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    metric_card("Period High", f"${performance.get('period_high', 'N/A'):.2f}" if isinstance(performance.get('period_high'), (int, float)) else 'N/A')
                
                with col2:
                    metric_card("Period Low", f"${performance.get('period_low', 'N/A'):.2f}" if isinstance(performance.get('period_low'), (int, float)) else 'N/A')
                
                with col3:
                    metric_card("Avg Volume", f"{performance.get('avg_volume', 'N/A'):,.0f}" if isinstance(performance.get('avg_volume'), (int, float)) else 'N/A')
                
                with col4:
                    metric_card("Volatility", f"{performance.get('volatility', 'N/A'):.2f}%" if isinstance(performance.get('volatility'), (int, float)) else 'N/A')
            
            with tab2:
                # Technical analysis charts and indicators
                st.markdown("### Technical Indicators")
                
                tech_chart = create_technical_indicator_charts(df, ticker_input)
                if tech_chart:
                    st.plotly_chart(tech_chart, use_container_width=True)
                else:
                    st.warning("Insufficient data to create technical analysis")
                
                # Technical signals
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    signal_class = "neutral"
                    if performance.get('rsi_signal') == "Overbought":
                        signal_class = "negative"
                    elif performance.get('rsi_signal') == "Oversold":
                        signal_class = "positive"
                    
                    st.markdown(f"""
                    <div class="info-container">
                        <div class="metric-title">RSI Signal</div>
                        <div class="metric-value {signal_class}">{performance.get('rsi_signal', 'N/A')}</div>
                        <div>Current RSI: {performance.get('rsi', 'N/A'):.2f}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    signal_class = "neutral"
                    if performance.get('macd_signal') == "Bullish":
                        signal_class = "positive"
                    elif performance.get('macd_signal') == "Bearish":
                        signal_class = "negative"
                    
                    st.markdown(f"""
                    <div class="info-container">
                        <div class="metric-title">MACD Signal</div>
                        <div class="metric-value {signal_class}">{performance.get('macd_signal', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    signal_class = "neutral"
                    if performance.get('trend') == "Uptrend":
                        signal_class = "positive"
                    elif performance.get('trend') == "Downtrend":
                        signal_class = "negative"
                    
                    st.markdown(f"""
                    <div class="info-container">
                        <div class="metric-title">Price Trend</div>
                        <div class="metric-value {signal_class}">{performance.get('trend', 'N/A')}</div>
                    </div>
                    """, unsafe_allow_html=True)
            
            with tab3:
                # Price predictions
                st.markdown("### Price Predictions")
                
                # Generate predictions
                predictions = get_price_prediction(df, forecast_periods=5)
                if predictions is not None and not predictions.empty:
                    # Create prediction chart
                    fig = go.Figure()
                    
                    # Historical data
                    fig.add_trace(go.Scatter(
                        x=df.index[-30:],
                        y=df['Close'][-30:],
                        mode='lines',
                        name='Historical Price',
                        line=dict(color='#26a69a')
                    ))
                    
                    # Prediction data
                    fig.add_trace(go.Scatter(
                        x=predictions['Date'],
                        y=predictions['Predicted_Close'],
                        mode='lines+markers',
                        name='Predicted Price',
                        line=dict(color='#ff9800', dash='dash'),
                        marker=dict(size=8)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker_input} Price Prediction - Next 5 Periods",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=500,
                        template="plotly_dark",
                        margin=dict(l=0, r=0, t=30, b=0),
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Display prediction table
                    st.markdown("#### Predicted Prices")
                    
                    # Format prediction dataframe
                    pred_df = predictions.copy()
                    pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d %H:%M')
                    pred_df['Predicted_Close'] = pred_df['Predicted_Close'].round(2)
                    
                    # Add calculation for percent change from current price
                    current_price = df['Close'].iloc[-1]
                    pred_df['Change'] = ((pred_df['Predicted_Close'] - current_price) / current_price * 100).round(2)
                    
                    # Rename columns for display
                    pred_df = pred_df.rename(columns={
                        'Date': 'Date/Time',
                        'Predicted_Close': 'Predicted Price ($)',
                        'Change': 'Change (%)'
                    })
                    
                    st.dataframe(pred_df, use_container_width=True)
                    
                    # Add disclaimer
                    st.info("⚠️ **Disclaimer**: Predictions are based on historical data analysis and should not be used as the sole basis for investment decisions.")
                else:
                    st.warning("Insufficient data to generate predictions")
            
            with tab4:
                # News section
                st.markdown("### Latest News")
                
                try:
                    from data import get_news
                    news = get_news(ticker_input)
                    
                    if news:
                        for item in news[:5]:  # Display top 5 news items
                            news_card(
                                title=item['title'],
                                publisher=item['publisher'],
                                date=item['published'],
                                summary=item['summary'],
                                link=item['link']
                            )
                    else:
                        st.info("No recent news found for this stock.")
                except ImportError:
                    # Fallback if get_news is not available
                    st.info("News functionality not available in the current implementation.")
            
            with tab5:
                # Company information
                st.markdown("### Company Overview")
                st.markdown(stock_info['description'])
                
                # Financial metrics
                st.markdown("### Key Financial Metrics")
                col1, col2 = st.columns(2)
                
                financial_metrics = [
                    ("Market Cap", stock_info['market_cap']),
                    ("P/E Ratio", stock_info['pe_ratio']),
                    ("EPS", stock_info['eps']),
                    ("Dividend Yield", stock_info['dividend_yield']),
                    ("52-Week High", stock_info.get('target_price', 'N/A'))
                ]
                
                with col1:
                    for i in range(0, len(financial_metrics), 2):
                        if i < len(financial_metrics):
                            metric, value = financial_metrics[i]
                            st.markdown(f"**{metric}:** {value}")
                
                with col2:
                    for i in range(1, len(financial_metrics), 2):
                        if i < len(financial_metrics):
                            metric, value = financial_metrics[i]
                            st.markdown(f"**{metric}:** {value}")
                
                # Analyst recommendations
                st.markdown("### Analyst Recommendation")
                rec = stock_info.get('recommendation', 'N/A')
                rec_color = "neutral"
                if rec == "buy" or rec == "strongBuy":
                    rec_color = "positive"
                elif rec == "sell" or rec == "strongSell":
                    rec_color = "negative"
                
                st.markdown(f"<div class='metric-value {rec_color}' style='text-transform:capitalize;'>{rec}</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display watchlist
st.markdown("## Watchlist")
if not st.session_state.watchlist:
    st.info("Your watchlist is empty. Add stocks using the sidebar.")
else:
    # Create a 3-column layout for watchlist items
    cols = st.columns(3)
    for i, ticker in enumerate(st.session_state.watchlist):
        with cols[i % 3]:
            with st.container():
                watchlist_item(ticker)

# Display last update time and next update time
update_time = st.session_state.last_update.strftime("%Y-%m-%d %H:%M:%S")
next_update = (st.session_state.last_update + timedelta(seconds=st.session_state.refresh_interval)).strftime("%Y-%m-%d %H:%M:%S")

st.markdown("---")
col1, col2 = st.columns(2)
with col1:
    st.caption(f"Last updated: {update_time}")
with col2:
    if st.session_state.auto_refresh:
        st.caption(f"Next update: {next_update}")

# Add footer with disclaimer
st.markdown("""
<div style="text-align: center; margin-top: 30px; padding: 10px; background-color: #1a1c24; border-radius: 6px;">
    <p style="font-size: 0.8rem; color: #a3a8b8;">
        Disclaimer: This dashboard is for educational and informational purposes only. 
        Not financial advice. Data may be delayed or inaccurate.
    </p>
</div>
""", unsafe_allow_html=True)