"""
Discord Alert System
Provides alerts for trades, errors, and optimization via Discord webhooks.
"""
import json
import time
import requests
from datetime import datetime
from typing import Dict, List, Union, Any, Optional
import threading
from collections import deque
import random

from config.api_keys import DISCORD_WEBHOOK_URL
from config.config import SEND_DISCORD_ALERTS, ALERT_ON_TRADE, ALERT_ON_ERROR, ALERT_ON_OPTIMIZATION
from utils.logger import setup_logger

logger = setup_logger("discord_alerts")

# Create a rate limiter and queue for Discord messages
class DiscordRateLimiter:
    """Rate limiter for Discord webhook messages"""
    def __init__(self, webhook_url, max_per_minute=30):
        self.webhook_url = webhook_url
        self.max_per_minute = max_per_minute
        self.message_queue = deque()
        self.last_sent_time = 0
        self.min_interval = 60 / max_per_minute  # Seconds between messages
        self.lock = threading.Lock()
        self.worker_thread = None
        self.running = False
    
    def start(self):
        """Start the message sending worker thread"""
        if self.worker_thread is None or not self.worker_thread.is_alive():
            self.running = True
            self.worker_thread = threading.Thread(target=self._worker, daemon=True)
            self.worker_thread.start()
    
    def stop(self):
        """Stop the message sending worker thread"""
        self.running = False
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=2)
    
    def add_message(self, payload):
        """Add a message to the queue and start worker if needed"""
        with self.lock:
            self.message_queue.append(payload)
        self.start()
    
    def _worker(self):
        """Worker thread that processes the queue with rate limiting"""
        while self.running:
            try:
                # Check if there are messages to send
                if not self.message_queue:
                    time.sleep(0.5)
                    continue
                
                # Check if we need to wait due to rate limiting
                current_time = time.time()
                time_since_last = current_time - self.last_sent_time
                
                if time_since_last < self.min_interval:
                    # Add a small random delay to prevent bursts
                    sleep_time = self.min_interval - time_since_last + random.uniform(0.1, 0.5)
                    time.sleep(sleep_time)
                
                # Get the next message from the queue
                with self.lock:
                    if not self.message_queue:
                        continue
                    payload = self.message_queue.popleft()
                
                # Send the message
                try:
                    response = requests.post(
                        self.webhook_url,
                        data=json.dumps(payload),
                        headers={"Content-Type": "application/json"},
                        timeout=10
                    )
                    
                    if response.status_code == 204:
                        # Success
                        self.last_sent_time = time.time()
                    elif response.status_code == 429:
                        # Rate limited - parse retry_after and wait
                        try:
                            retry_after = response.json().get("retry_after", 1)
                            logger.warning(f"Discord rate limited, waiting {retry_after} seconds")
                            time.sleep(retry_after)
                        except:
                            # Default to 5 seconds if can't parse response
                            time.sleep(5)
                        
                        # Put the message back in the queue
                        with self.lock:
                            self.message_queue.appendleft(payload)
                    else:
                        logger.warning(f"Discord API error: {response.status_code} - {response.text}")
                except requests.exceptions.RequestException as e:
                    logger.error(f"Discord webhook request failed: {e}")
                    
                    # Put the message back on the queue for retry
                    with self.lock:
                        self.message_queue.appendleft(payload)
                    time.sleep(2)
            
            except Exception as e:
                logger.error(f"Error in Discord rate limiter worker: {e}")
                time.sleep(1)
        
        logger.info("Discord rate limiter worker stopped")

# Create a global rate limiter instance
rate_limiter = DiscordRateLimiter(DISCORD_WEBHOOK_URL)

def send_discord_alert(
    content: str,
    webhook_url: str = DISCORD_WEBHOOK_URL,
    username: str = "FX Trading Bot",
    embed_title: Optional[str] = None,
    embed_fields: Optional[List[Dict[str, str]]] = None,
    embed_color: int = 3447003,  # Discord blue
    retry_count: int = 3
) -> bool:
    """
    Send an alert to Discord via webhook.
    
    Args:
        content (str): The message content.
        webhook_url (str, optional): The Discord webhook URL. Defaults to DISCORD_WEBHOOK_URL from config.
        username (str, optional): The bot username. Defaults to "FX Trading Bot".
        embed_title (Optional[str], optional): The embed title. Defaults to None.
        embed_fields (Optional[List[Dict[str, str]]], optional): List of embed fields. Defaults to None.
        embed_color (int, optional): The embed color. Defaults to Discord blue.
        retry_count (int, optional): Number of retries on failure. Defaults to 3.
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    # Check if Discord alerts are enabled
    if not SEND_DISCORD_ALERTS:
        logger.debug("Discord alerts are disabled")
        return False
    
    # Validate webhook URL
    if not webhook_url or not webhook_url.startswith("https://discord.com/api/webhooks/"):
        logger.error(f"Invalid Discord webhook URL: {webhook_url}")
        return False
    
    # Validate content
    if not content or not isinstance(content, str):
        logger.error(f"Invalid Discord alert content: {content}")
        return False
    
    # Truncate content if too long (Discord has a 2000 character limit)
    if len(content) > 1900:
        content = content[:1900] + "... [truncated]"
    
    # Prepare payload
    payload = {
        "content": content,
        "username": username[:80]  # Discord has a username length limit
    }
    
    # Add embed if specified
    if embed_title or embed_fields:
        embed = {
            "title": embed_title[:256] if embed_title else "",  # Discord has a title length limit
            "color": embed_color,
            "timestamp": datetime.now().isoformat()
        }
        
        if embed_fields:
            # Validate and process embed fields
            processed_fields = []
            for field in embed_fields:
                if isinstance(field, dict) and "name" in field and "value" in field:
                    # Discord has field name and value length limits
                    processed_field = {
                        "name": str(field["name"])[:256],
                        "value": str(field["value"])[:1024],
                        "inline": field.get("inline", False)
                    }
                    processed_fields.append(processed_field)
            
            # Discord has a limit of 25 fields
            embed["fields"] = processed_fields[:25]
        
        payload["embeds"] = [embed]
    
    # Add to rate limiter queue instead of direct sending
    rate_limiter.add_message(payload)
    return True

def send_trade_alert(
    trade_type: str,
    position_type: str,
    symbol: str,
    entry_price: Optional[float] = None,
    exit_price: Optional[float] = None,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    position_size: Optional[float] = None,
    pnl: Optional[float] = None,
    pnl_pips: Optional[float] = None,
    strategy: Optional[str] = None,
    exit_reason: Optional[str] = None
) -> bool:
    """
    Send a trade alert to Discord.
    
    Args:
        trade_type (str): The trade type ("ENTRY" or "EXIT").
        position_type (str): The position type ("LONG" or "SHORT").
        symbol (str): The trading symbol.
        entry_price (Optional[float], optional): The entry price. Defaults to None.
        exit_price (Optional[float], optional): The exit price. Defaults to None.
        stop_loss (Optional[float], optional): The stop loss price. Defaults to None.
        take_profit (Optional[float], optional): The take profit price. Defaults to None.
        position_size (Optional[float], optional): The position size. Defaults to None.
        pnl (Optional[float], optional): The profit/loss amount. Defaults to None.
        pnl_pips (Optional[float], optional): The profit/loss in pips. Defaults to None.
        strategy (Optional[str], optional): The strategy name. Defaults to None.
        exit_reason (Optional[str], optional): The exit reason. Defaults to None.
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    # Check if trade alerts are enabled
    if not ALERT_ON_TRADE:
        return False
    
    # Create content based on trade type
    if trade_type == "ENTRY":
        content = f"🔔 **{position_type} {symbol} ENTRY**"
        color = 5814783  # Green
        
        fields = [
            {"name": "Entry Price", "value": f"{entry_price:.5f}", "inline": True}
        ]
        
        if stop_loss:
            fields.append({"name": "Stop Loss", "value": f"{stop_loss:.5f}", "inline": True})
        
        if take_profit:
            fields.append({"name": "Take Profit", "value": f"{take_profit:.5f}", "inline": True})
        
        if position_size:
            fields.append({"name": "Position Size", "value": f"{position_size:.2f}", "inline": True})
        
        if strategy:
            fields.append({"name": "Strategy", "value": strategy, "inline": True})
        
    elif trade_type == "EXIT":
        # Determine if profitable
        profit = pnl is not None and pnl > 0
        content = f"🔔 **{position_type} {symbol} EXIT {('✅ PROFIT' if profit else '❌ LOSS') if pnl is not None else ''}**"
        color = 5814783 if profit else 15158332  # Green or red
        
        fields = []
        
        if entry_price:
            fields.append({"name": "Entry Price", "value": f"{entry_price:.5f}", "inline": True})
        
        if exit_price:
            fields.append({"name": "Exit Price", "value": f"{exit_price:.5f}", "inline": True})
        
        if pnl is not None:
            fields.append({"name": "P&L", "value": f"${pnl:.2f}", "inline": True})
        
        if pnl_pips is not None:
            fields.append({"name": "Pips", "value": f"{pnl_pips:.1f}", "inline": True})
        
        if position_size:
            fields.append({"name": "Position Size", "value": f"{position_size:.2f}", "inline": True})
        
        if exit_reason:
            fields.append({"name": "Exit Reason", "value": exit_reason, "inline": True})
        
        if strategy:
            fields.append({"name": "Strategy", "value": strategy, "inline": True})
    
    else:
        logger.warning(f"Unknown trade type: {trade_type}")
        return False
    
    # Send alert
    return send_discord_alert(
        content=content,
        embed_title=f"{symbol} {trade_type}",
        embed_fields=fields,
        embed_color=color
    )

def send_error_alert(error_message: str, error_type: str = "Error", details: Optional[Dict[str, Any]] = None) -> bool:
    """
    Send an error alert to Discord.
    
    Args:
        error_message (str): The error message.
        error_type (str, optional): The type of error. Defaults to "Error".
        details (Optional[Dict[str, Any]], optional): Additional error details. Defaults to None.
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    # Check if error alerts are enabled
    if not ALERT_ON_ERROR:
        return False
    
    content = f"⚠️ **{error_type}**"
    
    fields = [
        {"name": "Message", "value": error_message, "inline": False}
    ]
    
    if details:
        for key, value in details.items():
            fields.append({"name": key, "value": str(value), "inline": True})
    
    # Add timestamp
    fields.append({"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": False})
    
    # Send alert
    return send_discord_alert(
        content=content,
        embed_title=f"Error: {error_type}",
        embed_fields=fields,
        embed_color=15158332  # Red
    )

def send_optimization_alert(
    strategy: str,
    best_params: Dict[str, Any],
    performance: Dict[str, Any],
    optimization_type: str = "Standard"
) -> bool:
    """
    Send an optimization result alert to Discord.
    
    Args:
        strategy (str): The strategy name.
        best_params (Dict[str, Any]): The best parameters found.
        performance (Dict[str, Any]): The performance metrics.
        optimization_type (str, optional): The type of optimization. Defaults to "Standard".
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    # Check if optimization alerts are enabled
    if not ALERT_ON_OPTIMIZATION:
        return False
    
    content = f"🔧 **{strategy} Optimization Completed**"
    
    # Create fields for performance metrics
    performance_fields = []
    
    for metric, value in performance.items():
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.4f}" if isinstance(value, float) else str(value)
            performance_fields.append({"name": metric, "value": formatted_value, "inline": True})
    
    # Create fields for best parameters
    param_fields = []
    
    for param, value in best_params.items():
        param_fields.append({"name": param, "value": str(value), "inline": True})
    
    # Combine fields (up to 25 total)
    fields = performance_fields[:10]  # First 10 performance metrics
    
    # Add separator
    fields.append({"name": "Best Parameters", "value": "---", "inline": False})
    
    # Add parameters (up to 14)
    fields.extend(param_fields[:14])
    
    # Send alert
    return send_discord_alert(
        content=content,
        embed_title=f"{optimization_type} Optimization for {strategy}",
        embed_fields=fields,
        embed_color=10181046  # Purple
    )

def send_status_alert(
    status: str,
    details: Optional[Dict[str, Any]] = None,
    color: int = 3447003  # Discord blue
) -> bool:
    """
    Send a status update alert to Discord.
    
    Args:
        status (str): The status message.
        details (Optional[Dict[str, Any]], optional): Additional status details. Defaults to None.
        color (int, optional): The embed color. Defaults to Discord blue.
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    content = f"ℹ️ **Status Update**"
    
    fields = [
        {"name": "Status", "value": status, "inline": False}
    ]
    
    if details:
        for key, value in details.items():
            fields.append({"name": key, "value": str(value), "inline": True})
    
    # Add timestamp
    fields.append({"name": "Time", "value": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "inline": False})
    
    # Send alert
    return send_discord_alert(
        content=content,
        embed_title="Bot Status Update",
        embed_fields=fields,
        embed_color=color
    )

def send_performance_alert(performance: Dict[str, Any]) -> bool:
    """
    Send a performance summary alert to Discord.
    
    Args:
        performance (Dict[str, Any]): The performance metrics.
        
    Returns:
        bool: True if the alert was sent successfully, False otherwise.
    """
    content = "📊 **Performance Summary**"
    
    # Determine color based on overall performance
    total_return = performance.get("total_return_pct", 0)
    
    if total_return > 0:
        color = 5814783  # Green
    elif total_return < 0:
        color = 15158332  # Red
    else:
        color = 3447003  # Blue
    
    # Create fields
    fields = []
    
    # Add key metrics
    key_metrics = [
        "total_trades", "winning_trades", "losing_trades", "win_rate",
        "total_pips", "total_profit", "total_return_pct", "max_drawdown_pct",
        "profit_factor", "sharpe_ratio"
    ]
    
    for metric in key_metrics:
        if metric in performance:
            value = performance[metric]
            
            # Format value based on type
            if isinstance(value, float):
                if metric.endswith("_pct"):
                    formatted_value = f"{value:.2f}%"
                elif metric in ["profit_factor", "sharpe_ratio"]:
                    formatted_value = f"{value:.2f}"
                else:
                    formatted_value = f"{value:.2f}"
            else:
                formatted_value = str(value)
            
            # Format metric name
            formatted_name = metric.replace("_", " ").title()
            
            fields.append({"name": formatted_name, "value": formatted_value, "inline": True})
    
    # Add strategy performance if available
    if "strategy_performance" in performance and performance["strategy_performance"]:
        fields.append({"name": "Strategy Performance", "value": "---", "inline": False})
        
        for strategy, stats in performance["strategy_performance"].items():
            if "win_rate" in stats and "pnl" in stats:
                fields.append({
                    "name": strategy,
                    "value": f"Win Rate: {stats['win_rate']:.2f}%, P&L: ${stats['pnl']:.2f}",
                    "inline": True
                })
    
    # Send alert
    return send_discord_alert(
        content=content,
        embed_title="Trading Performance Summary",
        embed_fields=fields,
        embed_color=color
    )