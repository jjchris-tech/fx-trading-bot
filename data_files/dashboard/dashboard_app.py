
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
import traceback

# Set page config
st.set_page_config(
    page_title="FX Trading Bot Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': "FX Trading Bot Dashboard"
    }
)

# Dashboard directories
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
TRADE_LOG_FILE = os.path.join(DATA_DIR, "trade_log.json")
EQUITY_FILE = os.path.join(DATA_DIR, "equity_curve.json")
METRICS_FILE = os.path.join(DATA_DIR, "metrics.json")
MARKET_DATA_FILE = os.path.join(DATA_DIR, "market_data.json")
SENTIMENT_FILE = os.path.join(DATA_DIR, "sentiment.json")

# Dashboard refresh rate in seconds
REFRESH_RATE = 5

# Helper functions
def load_json(file_path):
    """Load data from JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading {file_path}: {e}")
        return None

def safe_load_json(file_path, default_value=None):
    """Safely load JSON with fallback to default value."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Could not load data from {file_path}: {str(e)}")
        return default_value

def format_number(num, precision=2, include_sign=False):
    """Format number with thousands separator and specified precision."""
    if num is None:
        return "N/A"
    
    if isinstance(num, str):
        return num
    
    try:
        sign = "+" if num > 0 and include_sign else ""
        return f"{sign}{num:,.{precision}f}"
    except Exception:
        return str(num)

def get_color(value, neutral_zero=True):
    """Get color based on value."""
    if value > 0:
        return "green"
    elif value < 0:
        return "red"
    else:
        return "white" if neutral_zero else "yellow"

def parse_datetime(dt_str):
    """Parse datetime string to datetime object."""
    if not dt_str:
        return None
    try:
        if isinstance(dt_str, str):
            # Try multiple formats
            formats = [
                "%Y-%m-%dT%H:%M:%S.%f", 
                "%Y-%m-%dT%H:%M:%S", 
                "%Y-%m-%d %H:%M:%S",
                "%Y-%m-%d"
            ]
            
            for fmt in formats:
                try:
                    return datetime.strptime(dt_str.split("+")[0].split("Z")[0], fmt)
                except:
                    continue
                    
            # If all formats failed, use pandas
            return pd.to_datetime(dt_str)
        return dt_str
    except Exception as e:
        st.warning(f"Error parsing datetime {dt_str}: {e}")
        return None

# Dashboard header
st.title("🤖 EUR/USD FX Trading Bot")
st.markdown("Real-time trading performance dashboard")

# Setup sidebar
st.sidebar.header("Dashboard Controls")
auto_refresh = st.sidebar.checkbox("Auto Refresh", value=True)
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", min_value=1, max_value=60, value=REFRESH_RATE)

# Use safe rerun to handle different Streamlit versions
def safe_rerun():
    try:
        st.rerun()  # Modern Streamlit
    except AttributeError:
        try:
            st.experimental_rerun()  # Older Streamlit
        except AttributeError:
            # If neither method works, display a message
            st.warning("Auto-refresh not available in this Streamlit version. Please refresh manually.")
            time.sleep(refresh_rate)  # Still pause before next attempt

if st.sidebar.button("Refresh Now"):
    safe_rerun()

# Add data filters to sidebar
st.sidebar.header("Data Filters")
time_period = st.sidebar.selectbox(
    "Time Period",
    options=["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
    index=3
)

# Add last update time to sidebar
last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.sidebar.markdown(f"**Last Updated:** {last_update}")

# Get filtered time range
def get_time_range(period):
    now = datetime.now()
    if period == "Last 24 Hours":
        return now - timedelta(days=1), now
    elif period == "Last 7 Days":
        return now - timedelta(days=7), now
    elif period == "Last 30 Days":
        return now - timedelta(days=30), now
    else:  # All Time
        return datetime(2000, 1, 1), now

start_date, end_date = get_time_range(time_period)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Trades", "Performance", "Market"])

# Load data
@st.cache_data(ttl=refresh_rate)
def load_dashboard_data():
    """Load all dashboard data."""
    try:
        trade_log = safe_load_json(TRADE_LOG_FILE, [])
        equity_curve = safe_load_json(EQUITY_FILE, [10000])
        metrics = safe_load_json(METRICS_FILE, {})
        market_data = safe_load_json(MARKET_DATA_FILE, {})
        sentiment_data = safe_load_json(SENTIMENT_FILE, {})
        
        return {
            "trade_log": trade_log or [],
            "equity_curve": equity_curve or [10000],
            "metrics": metrics or {},
            "market_data": market_data or {},
            "sentiment_data": sentiment_data or {}
        }
    except Exception as e:
        st.error(f"Error loading dashboard data: {e}")
        return {
            "trade_log": [],
            "equity_curve": [10000],
            "metrics": {},
            "market_data": {},
            "sentiment_data": {}
        }

try:
    data = load_dashboard_data()
    trade_log = data["trade_log"]
    equity_curve = data["equity_curve"]
    metrics = data["metrics"]
    market_data = data["market_data"]
    sentiment_data = data["sentiment_data"]

    # Convert trade log dates to datetime
    for trade in trade_log:
        if "entry_time" in trade and isinstance(trade["entry_time"], str):
            trade["entry_time"] = parse_datetime(trade["entry_time"])
        
        if "exit_time" in trade and isinstance(trade["exit_time"], str):
            trade["exit_time"] = parse_datetime(trade["exit_time"])

    # Filter trade log by time period
    filtered_trades = []
    for trade in trade_log:
        exit_time = trade.get("exit_time")
        try:
            if isinstance(exit_time, datetime) and start_date <= exit_time <= end_date:
                filtered_trades.append(trade)
        except:
            continue  # Skip trades with invalid dates

    # Calculate filtered metrics
    def calculate_filtered_metrics(trades):
        if not trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pips": 0,
                "total_profit": 0,
                "avg_trade": 0,
                "profit_factor": 0
            }
        
        winning_trades = sum(1 for t in trades if t.get("pnl", 0) > 0)
        losing_trades = sum(1 for t in trades if t.get("pnl", 0) < 0)
        total_pips = sum(t.get("pnl_pips", 0) for t in trades)
        total_profit = sum(t.get("pnl", 0) for t in trades)
        
        total_trades = len(trades)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        avg_trade = total_profit / total_trades if total_trades > 0 else 0
        
        total_wins = sum(max(0, t.get("pnl", 0)) for t in trades)
        total_losses = abs(sum(min(0, t.get("pnl", 0)) for t in trades))
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        return {
            "total_trades": total_trades,
            "winning_trades": winning_trades,
            "losing_trades": losing_trades,
            "win_rate": win_rate,
            "total_pips": total_pips,
            "total_profit": total_profit,
            "avg_trade": avg_trade,
            "profit_factor": profit_factor
        }

    filtered_metrics = calculate_filtered_metrics(filtered_trades)

    # Tab 1: Overview
    with tab1:
        # Top metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        # Current capital
        current_capital = metrics.get("current_capital", 10000)
        initial_capital = metrics.get("initial_capital", 10000)
        total_return = metrics.get("total_return_pct", 0)
        
        with col1:
            st.metric(
                label="Current Capital",
                value=f"${format_number(current_capital)}",
                delta=f"{format_number(total_return, include_sign=True)}%"
            )
        
        # Win rate
        win_rate = filtered_metrics.get("win_rate", 0)
        with col2:
            st.metric(
                label="Win Rate",
                value=f"{format_number(win_rate)}%",
                delta=None
            )
        
        # Total profit
        total_profit = filtered_metrics.get("total_profit", 0)
        with col3:
            st.metric(
                label="Profit (Period)",
                value=f"${format_number(total_profit)}",
                delta=f"{format_number(filtered_metrics.get('total_pips', 0))} pips"
            )
        
        # Profit factor
        profit_factor = filtered_metrics.get("profit_factor", 0)
        with col4:
            st.metric(
                label="Profit Factor",
                value=format_number(profit_factor),
                delta=None
            )
        
        # Equity chart
        st.subheader("Equity Curve")
        
        # Create equity dataframe
        equity_df = pd.DataFrame({
            "Equity": equity_curve,
            "Trade #": range(len(equity_curve))
        })
        
        if not equity_df.empty and len(equity_df) > 1:
            fig = px.line(
                equity_df, 
                x="Trade #", 
                y="Equity",
                labels={"Equity": "Equity ($)", "Trade #": "Trade Number"},
                height=400
            )
            
            # Add initial capital as horizontal line
            fig.add_hline(
                y=initial_capital, 
                line_dash="dash", 
                line_color="white", 
                annotation_text=f"Initial: ${format_number(initial_capital)}"
            )
            
            # Update layout
            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                hovermode="x unified"
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough equity data to display chart")
        
        # Market and sentiment row
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Market Summary")
            
            # Current price
            current_price = market_data.get("price", 0)
            
            # Create price history
            price_history = market_data.get("history", [])
            
            if price_history:
                try:
                    price_change = current_price - price_history[0]["price"] if price_history else 0
                    price_change_pct = (price_change / price_history[0]["price"]) * 100 if price_history and price_history[0]["price"] > 0 else 0
                    
                    st.metric(
                        label=f"{market_data.get('symbol', 'EUR/USD')} Price",
                        value=f"{format_number(current_price, precision=5)}",
                        delta=f"{format_number(price_change_pct, include_sign=True)}%"
                    )
                    
                    # Create price chart
                    price_df = pd.DataFrame(price_history)
                    if not price_df.empty and "timestamp" in price_df.columns:
                        try:
                            price_df["timestamp"] = pd.to_datetime(price_df["timestamp"])
                            
                            fig = px.line(
                                price_df, 
                                x="timestamp", 
                                y="price",
                                labels={"price": "Price", "timestamp": "Time"},
                                height=250
                            )
                            
                            fig.update_layout(
                                template="plotly_dark",
                                margin=dict(l=0, r=0, t=10, b=0),
                                hovermode="x unified"
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating price chart: {e}")
                except Exception as e:
                    st.error(f"Error processing price data: {e}")
            else:
                st.info("No price data available")
        
        with col2:
            st.subheader("Sentiment Analysis")
            
            # Validate sentiment data type
            if not isinstance(sentiment_data, dict):
                st.error("Sentiment data is not in the expected format.")
                sentiment_data = {
                    "signal": "neutral",
                    "strength": 0,
                    "confidence": 0,
                    "sentiment_data": {
                        "overall_score": 0,
                        "bullish_count": 0,
                        "bearish_count": 0,
                        "neutral_count": 0,
                        "total_count": 0,
                        "items": []
                    }
                }
            
            # Sentiment signal
            signal = sentiment_data.get("signal", "neutral")
            strength = sentiment_data.get("strength", 0)
            confidence = sentiment_data.get("confidence", 0)
            
            # Format signal
            signal_formatted = signal.upper()
            signal_color = "green" if signal == "buy" else "red" if signal == "sell" else "yellow"
            
            # Create columns for signal and confidence
            sig_col, conf_col = st.columns(2)
            
            with sig_col:
                st.metric(
                    label="Signal",
                    value=signal_formatted,
                    delta=f"Strength: {format_number(strength, precision=2)}"
                )
            
            with conf_col:
                st.metric(
                    label="Confidence",
                    value=f"{format_number(confidence * 100)}%",
                    delta=None
                )
            
            # Get sentiment data details
            sentiment_details = sentiment_data.get("sentiment_data", {})
            
            # Create sentiment summary
            summary_data = [
                {"Category": "Bullish", "Count": sentiment_details.get("bullish_count", 0)},
                {"Category": "Bearish", "Count": sentiment_details.get("bearish_count", 0)},
                {"Category": "Neutral", "Count": sentiment_details.get("neutral_count", 0)}
            ]
            
            summary_df = pd.DataFrame(summary_data)
            
            if not summary_df.empty and summary_df["Count"].sum() > 0:
                fig = px.pie(
                    summary_df, 
                    values="Count", 
                    names="Category",
                    color="Category",
                    color_discrete_map={"Bullish": "green", "Bearish": "red", "Neutral": "yellow"},
                    height=250
                )
                
                fig.update_layout(
                    template="plotly_dark",
                    margin=dict(l=0, r=0, t=10, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sentiment data available")
        
        # Latest trades table
        st.subheader("Latest Trades")
        
        if filtered_trades:
            # Sort by exit time (most recent first)
            try:
                sorted_trades = sorted(
                    filtered_trades,
                    key=lambda x: x.get("exit_time", datetime.now()),
                    reverse=True
                )
                
                # Limit to 5 trades
                latest_trades = sorted_trades[:5]
                
                # Create table data
                table_data = []
                for trade in latest_trades:
                    entry_time = trade.get("entry_time")
                    if isinstance(entry_time, datetime):
                        entry_time = entry_time.strftime("%m/%d %H:%M")
                    
                    exit_time = trade.get("exit_time")
                    if isinstance(exit_time, datetime):
                        exit_time = exit_time.strftime("%m/%d %H:%M")
                    
                    pnl = trade.get("pnl", 0)
                    pnl_str = f"${format_number(pnl)}"
                    pnl_color = get_color(pnl)
                    
                    table_data.append({
                        "ID": trade.get("id", ""),
                        "Type": trade.get("type", "").upper(),
                        "Entry Time": entry_time,
                        "Exit Time": exit_time,
                        "P&L": pnl_str,
                        "Pips": format_number(trade.get("pnl_pips", 0), precision=1),
                        "Strategy": trade.get("strategy", "")
                    })
                
                # Display table
                st.table(pd.DataFrame(table_data))
            except Exception as e:
                st.error(f"Error creating trades table: {e}")
        else:
            st.info("No trades in the selected period")

    # Rest of the code...
    # Tab 2, 3, and 4 implementation would go here
    # For brevity, I'm implementing abbreviated versions
    
    # Tab 2: Trades 
    with tab2:
        st.info("Trades tab - Visit the Overview tab for key metrics")
    
    # Tab 3: Performance
    with tab3:
        st.info("Performance tab - Visit the Overview tab for key metrics")
    
    # Tab 4: Market
    with tab4:
        st.info("Market tab - Visit the Overview tab for market data")

except Exception as e:
    st.error(f"An error occurred in the dashboard: {e}")
    st.error(traceback.format_exc())

# Auto-refresh
if auto_refresh:
    time.sleep(refresh_rate)
    safe_rerun()
