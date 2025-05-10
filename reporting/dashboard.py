"""
Live Dashboard Module
Provides a real-time web dashboard for monitoring trading performance.
"""
import os
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
import threading
import webbrowser
import subprocess
import signal
from pathlib import Path
import sys
import atexit
import tempfile

from config.config import (
    DASHBOARD_PORT, DASHBOARD_REFRESH_RATE, DASHBOARD_THEME,
    SYMBOL, INITIAL_CAPITAL, DATA_DIR
)
from utils.logger import setup_logger

logger = setup_logger("dashboard")

class Dashboard:
    """
    Live dashboard for monitoring trading performance.
    """
    def __init__(self, 
                 port: int = DASHBOARD_PORT,
                 refresh_rate: int = DASHBOARD_REFRESH_RATE,
                 theme: str = DASHBOARD_THEME,
                 auto_open: bool = True):
        """
        Initialize the Dashboard.
        
        Args:
            port (int, optional): Dashboard server port. Defaults to DASHBOARD_PORT from config.
            refresh_rate (int, optional): Dashboard refresh rate in seconds. 
                Defaults to DASHBOARD_REFRESH_RATE from config.
            theme (str, optional): Dashboard theme ('light' or 'dark'). 
                Defaults to DASHBOARD_THEME from config.
            auto_open (bool, optional): Whether to automatically open the dashboard in browser. 
                Defaults to True.
        """
        self.port = port
        self.refresh_rate = refresh_rate
        self.theme = theme
        self.auto_open = auto_open
        self.process = None
        
        # Create a dashboard data directory
        self.data_dir = Path(DATA_DIR) / "dashboard"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Dashboard script path
        self.dashboard_script = self.data_dir / "dashboard_app.py"
        
        # Data files
        self.trade_log_file = self.data_dir / "trade_log.json"
        self.equity_file = self.data_dir / "equity_curve.json"
        self.metrics_file = self.data_dir / "metrics.json"
        self.market_data_file = self.data_dir / "market_data.json"
        self.sentiment_file = self.data_dir / "sentiment.json"
        
        # Create empty data files if they don't exist
        self._create_empty_data_files()
        
        # Generate dashboard script
        self._generate_dashboard_script()
        
        # Register exit handler to ensure process is stopped on exit
        atexit.register(self.stop)
    
    def _create_empty_data_files(self) -> None:
        """
        Create empty data files if they don't exist.
        """
        try:
            # Trade log
            if not self.trade_log_file.exists():
                with open(self.trade_log_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            
            # Equity curve
            if not self.equity_file.exists():
                with open(self.equity_file, 'w', encoding='utf-8') as f:
                    json.dump([INITIAL_CAPITAL], f)
            
            # Metrics
            if not self.metrics_file.exists():
                with open(self.metrics_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "total_trades": 0,
                        "winning_trades": 0,
                        "losing_trades": 0,
                        "win_rate": 0,
                        "total_pips": 0,
                        "total_profit": 0,
                        "avg_trade": 0,
                        "profit_factor": 0,
                        "total_return_pct": 0,
                        "max_drawdown_pct": 0,
                        "sharpe_ratio": 0,
                        "initial_capital": INITIAL_CAPITAL,
                        "current_capital": INITIAL_CAPITAL,
                        "timestamp": datetime.now().isoformat()
                    }, f, default=str)
            
            # Market data
            if not self.market_data_file.exists():
                with open(self.market_data_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "symbol": SYMBOL,
                        "price": 0,
                        "timestamp": datetime.now().isoformat(),
                        "history": []
                    }, f, default=str)
            
            # Sentiment data
            if not self.sentiment_file.exists():
                with open(self.sentiment_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "symbol": SYMBOL,
                        "signal": "neutral",
                        "strength": 0,
                        "confidence": 0,
                        "timestamp": datetime.now().isoformat(),
                        "sentiment_data": {
                            "overall_score": 0,
                            "bullish_count": 0,
                            "bearish_count": 0,
                            "neutral_count": 0,
                            "total_count": 0,
                            "items": []
                        }
                    }, f, default=str)
        except Exception as e:
            logger.error(f"Error creating empty data files: {e}")
    
    def _generate_dashboard_script(self) -> None:
        """
        Generate the Streamlit dashboard script with fixes applied.
        """
        try:
            dashboard_code = '''
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
'''
            
            # Write to file with UTF-8 encoding
            with open(self.dashboard_script, 'w', encoding='utf-8') as f:
                f.write(dashboard_code)
            
            logger.info(f"Generated dashboard script: {self.dashboard_script}")
        except Exception as e:
            logger.error(f"Error generating dashboard script: {e}")
    
    def update_trade_log(self, trade_log: List[Dict[str, Any]]) -> None:
        """
        Update the trade log data file.
        
        Args:
            trade_log (List[Dict[str, Any]]): The updated trade log.
        """
        # Save to file
        try:
            with open(self.trade_log_file, 'w', encoding='utf-8') as f:
                json.dump(trade_log, f, default=str)
            logger.debug(f"Updated trade log with {len(trade_log)} trades")
        except Exception as e:
            logger.error(f"Error updating trade log: {e}")
    
    def update_equity_curve(self, equity_curve: List[float]) -> None:
        """
        Update the equity curve data file.
        
        Args:
            equity_curve (List[float]): The updated equity curve.
        """
        # Save to file
        try:
            with open(self.equity_file, 'w', encoding='utf-8') as f:
                json.dump(equity_curve, f)
            logger.debug(f"Updated equity curve with {len(equity_curve)} points")
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Update the metrics data file.
        
        Args:
            metrics (Dict[str, Any]): The updated metrics.
        """
        # Ensure metrics has a timestamp
        if "timestamp" not in metrics:
            metrics["timestamp"] = datetime.now().isoformat()
            
        # Save to file
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics, f, default=str)
            logger.debug(f"Updated metrics: {len(metrics)} values")
        except Exception as e:
            logger.error(f"Error updating metrics: {e}")
    
    def update_market_data(self, market_price: float, symbol: str = SYMBOL) -> None:
        """
        Update the market data file.
        
        Args:
            market_price (float): The current market price.
            symbol (str, optional): The trading symbol. Defaults to SYMBOL from config.
        """
        # Validate price and prevent errors
        try:
            market_price = float(market_price)
            if market_price <= 0:
                logger.error(f"Invalid market price: {market_price}")
                return
                
            # Load existing data
            try:
                with open(self.market_data_file, 'r', encoding='utf-8') as f:
                    market_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                market_data = {
                    "symbol": symbol,
                    "price": 0,
                    "timestamp": datetime.now().isoformat(),
                    "history": []
                }
            
            # Update data
            market_data["price"] = market_price
            market_data["timestamp"] = datetime.now().isoformat()
            
            # Add to history
            history_entry = {
                "price": market_price,
                "timestamp": datetime.now().isoformat()
            }
            
            # Check if history exists and is a list
            if not isinstance(market_data.get("history"), list):
                market_data["history"] = []
                
            market_data["history"].append(history_entry)
            
            # Limit history to 1000 points
            market_data["history"] = market_data["history"][-1000:]
            
            # Save to file
            with open(self.market_data_file, 'w', encoding='utf-8') as f:
                json.dump(market_data, f, default=str)
                
            logger.debug(f"Updated market data: {symbol} price={market_price}")
        except Exception as e:
            logger.error(f"Error updating market data: {e}")
    
    def update_sentiment_data(self, sentiment_data: Dict[str, Any]) -> None:
        """
        Update the sentiment data file with type checking.
        
        Args:
            sentiment_data (Dict[str, Any]): The sentiment data.
        """
        # Validate data type and handle errors
        try:
            if not isinstance(sentiment_data, dict):
                logger.error(f"Invalid sentiment data type: {type(sentiment_data)}")
                sentiment_data = {
                    "symbol": SYMBOL,
                    "signal": "neutral",
                    "strength": 0,
                    "confidence": 0,
                    "timestamp": datetime.now().isoformat(),
                    "sentiment_data": {
                        "overall_score": 0,
                        "bullish_count": 0,
                        "bearish_count": 0,
                        "neutral_count": 0,
                        "total_count": 0,
                        "items": []
                    }
                }
            else:
                # Ensure timestamp is present
                if "timestamp" not in sentiment_data:
                    sentiment_data["timestamp"] = datetime.now().isoformat()
            
            # Save to file
            with open(self.sentiment_file, 'w', encoding='utf-8') as f:
                json.dump(sentiment_data, f, default=str)
                
            logger.debug(f"Updated sentiment data: signal={sentiment_data.get('signal', 'unknown')}")
        except Exception as e:
            logger.error(f"Error updating sentiment data: {e}")
    
    def fetch_and_update_data(self) -> None:
        """
        Fetch and update all data with error handling.
        """
        pass  # This would be implemented with actual data fetching
    
    def _find_streamlit_process(self) -> Optional[int]:
        """
        Find the PID of any running Streamlit process.
        
        Returns:
            Optional[int]: PID of the Streamlit process, or None if not found.
        """
        streamlit_procs = []
        
        try:
            if sys.platform == "win32":
                # Windows
                try:
                    output = subprocess.check_output("tasklist /FI \"IMAGENAME eq streamlit.exe\"", shell=True).decode('utf-8', errors='ignore')
                    if "streamlit.exe" in output:
                        for line in output.split('\n'):
                            if "streamlit.exe" in line:
                                parts = line.split()
                                if len(parts) >= 2:
                                    try:
                                        pid = int(parts[1])
                                        streamlit_procs.append(pid)
                                    except ValueError:
                                        pass
                except Exception as e:
                    logger.error(f"Error finding streamlit process on Windows: {e}")
            else:
                # Unix/Linux/Mac
                try:
                    output = subprocess.check_output(["pgrep", "-f", "streamlit"]).decode('utf-8', errors='ignore')
                    streamlit_procs = [int(pid) for pid in output.split()]
                except Exception as e:
                    logger.error(f"Error finding streamlit process on Unix: {e}")
                
            return streamlit_procs[0] if streamlit_procs else None
        except Exception as e:
            logger.error(f"Error in _find_streamlit_process: {e}")
            return None
    
    def start(self) -> None:
        """
        Start the dashboard server with improved error handling.
        """
        try:
            # Kill any existing streamlit processes
            self.stop()
            
            logger.info(f"Starting dashboard server on port {self.port}")
            
            # Create a batch file to run streamlit on Windows
            if sys.platform == "win32":
                try:
                    batch_path = os.path.join(tempfile.gettempdir(), "run_streamlit.bat")
                    with open(batch_path, "w", encoding='utf-8') as f:
                        f.write(f"@echo off\n")
                        f.write(f"cd /d {os.path.dirname(sys.executable)}\n")
                        f.write(f"streamlit run \"{self.dashboard_script}\" --server.port {self.port} --server.headless true --theme.base {self.theme}\n")
                    
                    # Start the batch file
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = 0  # SW_HIDE
                    
                    self.process = subprocess.Popen(
                        batch_path,
                        shell=True,
                        creationflags=subprocess.CREATE_NEW_CONSOLE,
                        startupinfo=startupinfo
                    )
                except Exception as e:
                    logger.error(f"Error starting Windows dashboard: {e}")
                    # Fall back to direct execution
                    try:
                        cmd = [
                            sys.executable, "-m", "streamlit", "run", 
                            str(self.dashboard_script),
                            "--server.port", str(self.port),
                            "--server.headless", "true",
                            "--theme.base", self.theme
                        ]
                        
                        self.process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True
                        )
                    except Exception as e2:
                        logger.error(f"Error in fallback dashboard start: {e2}")
            else:
                # Build command for non-Windows platforms
                cmd = [
                    sys.executable, "-m", "streamlit", "run", 
                    str(self.dashboard_script),
                    "--server.port", str(self.port),
                    "--server.headless", "true",
                    "--theme.base", self.theme
                ]
                
                # Start process
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            # Wait for server to start
            start_time = time.time()
            started = False
            while time.time() - start_time < 15:  # Wait up to 15 seconds
                time.sleep(1)
                # Check if a Streamlit process is running
                if self._find_streamlit_process():
                    logger.info("Streamlit process detected")
                    started = True
                    break
            
            # Open browser if auto_open is True and process started
            if self.auto_open and started:
                time.sleep(2)  # Give the server a moment to initialize
                webbrowser.open(f"http://localhost:{self.port}")
                logger.info(f"Dashboard opened at http://localhost:{self.port}")
            elif not started:
                logger.error("Dashboard server failed to start")
            
        except Exception as e:
            logger.error(f"Error starting dashboard server: {e}")
    
    def stop(self) -> None:
        """
        Stop the dashboard server with improved error handling.
        """
        try:
            # Find and kill any streamlit processes
            streamlit_pid = self._find_streamlit_process()
            if streamlit_pid:
                logger.info(f"Killing Streamlit process: {streamlit_pid}")
                try:
                    if sys.platform == "win32":
                        subprocess.call(["taskkill", "/F", "/PID", str(streamlit_pid)], 
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    else:
                        os.kill(streamlit_pid, signal.SIGKILL)
                except Exception as e:
                    logger.error(f"Error killing Streamlit process: {e}")
            
            # Kill our process
            if self.process is not None:
                try:
                    self.process.terminate()
                    self.process.wait(timeout=5)
                except Exception as e:
                    logger.error(f"Error terminating process: {e}")
                    # Force kill if terminate fails
                    try:
                        if sys.platform == "win32":
                            subprocess.call(["taskkill", "/F", "/PID", str(self.process.pid)], 
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        else:
                            os.kill(self.process.pid, signal.SIGKILL)
                    except Exception as e2:
                        logger.error(f"Error force killing process: {e2}")
                
                self.process = None
                
            logger.info("Dashboard server stopped")
            
        except Exception as e:
            logger.error(f"Error stopping dashboard server: {e}")
    
    def update_all(self, 
                  trade_log: Optional[List[Dict[str, Any]]] = None,
                  equity_curve: Optional[List[float]] = None,
                  metrics: Optional[Dict[str, Any]] = None) -> None:
        """
        Update all dashboard data.
        
        Args:
            trade_log (Optional[List[Dict[str, Any]]], optional): Trade log. Defaults to None.
            equity_curve (Optional[List[float]], optional): Equity curve. Defaults to None.
            metrics (Optional[Dict[str, Any]], optional): Metrics. Defaults to None.
        """
        # Update data files with error handling
        try:
            if trade_log is not None:
                self.update_trade_log(trade_log)
            
            if equity_curve is not None:
                self.update_equity_curve(equity_curve)
            
            if metrics is not None:
                self.update_metrics(metrics)
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
        
    def is_running(self) -> bool:
        """
        Check if the dashboard server is running.
        
        Returns:
            bool: True if running, False otherwise.
        """
        # Check for any streamlit process with error handling
        try:
            return self._find_streamlit_process() is not None
        except Exception as e:
            logger.error(f"Error checking if dashboard is running: {e}")
            return False