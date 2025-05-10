"""
Sentiment Analysis Module
Fetches and analyzes news sentiment for forex pairs using FMP API.
"""
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any, Optional
import re

from config.api_keys import FMP_API_KEY
from config.config import (
    SYMBOL, API_REQUEST_TIMEOUT, API_MAX_RETRIES, 
    API_RETRY_DELAY, DATA_DIR
)
from utils.logger import setup_logger
from utils.helpers import retry_function

# Set up logger
logger = setup_logger("sentiment")

class SentimentAnalyzer:
    """
    Class for analyzing sentiment from forex news.
    """
    def __init__(self, symbol: str = SYMBOL, api_key: str = FMP_API_KEY):
        """
        Initialize the SentimentAnalyzer class.
        
        Args:
            symbol (str, optional): The forex symbol to analyze. Defaults to SYMBOL from config.
            api_key (str, optional): The FMP API key. Defaults to FMP_API_KEY from config.
        """
        self.symbol = symbol.replace("/", "")  # Remove slash for API
        self.api_key = api_key
        self.base_url = "https://financialmodelingprep.com/api/v3"
        self.cache_dir = os.path.join(DATA_DIR, "sentiment_cache")
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Define positive and negative keywords for manual sentiment analysis
        self.positive_keywords = [
            "rally", "surge", "jump", "rise", "gain", "bullish", "strong", 
            "positive", "uptrend", "recovery", "outperform", "beat", "exceed",
            "upside", "optimism", "confident", "robust", "strength", "growth",
            "momentum", "attractive", "opportunity", "upgrade", "buy", "long"
        ]
        
        self.negative_keywords = [
            "decline", "drop", "fall", "slump", "bearish", "weak", "negative",
            "downtrend", "downturn", "recession", "underperform", "miss", "below",
            "downside", "pessimism", "worried", "fragile", "weakness", "contraction",
            "slowdown", "risk", "concern", "downgrade", "sell", "short"
        ]
        
        # Currency-specific keywords
        self.currency_keywords = {
            "EUR": ["euro", "eurozone", "ecb", "european central bank", "draghi", "lagarde"],
            "USD": ["dollar", "usd", "fed", "federal reserve", "powell", "yellen"],
            "GBP": ["pound", "sterling", "boe", "bank of england", "bailey", "carney"],
            "JPY": ["yen", "boj", "bank of japan", "kuroda"],
            "CHF": ["franc", "swiss", "snb", "swiss national bank", "jordan"],
            "AUD": ["aussie", "australian dollar", "rba", "reserve bank of australia", "lowe"],
            "CAD": ["loonie", "canadian dollar", "boc", "bank of canada", "macklem"],
            "NZD": ["kiwi", "new zealand dollar", "rbnz", "reserve bank of new zealand", "orr"]
        }
        
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a request to the FMP API with improved error handling.
        
        Args:
            endpoint (str): The API endpoint.
            params (Dict[str, Any]): The request parameters.
            
        Returns:
            Dict[str, Any]: The API response.
        """
        url = f"{self.base_url}/{endpoint}"
        params["apikey"] = self.api_key
        
        try:
            response = retry_function(
                requests.get,
                max_retries=API_MAX_RETRIES,
                retry_delay=API_RETRY_DELAY,
                url=url,
                params=params,
                timeout=API_REQUEST_TIMEOUT
            )
            
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError as e:
                    logger.error(f"Invalid JSON response: {e}")
                    return []
            else:
                logger.error(f"API request failed with status code {response.status_code}: {response.text}")
                return []
                
        except Exception as e:
            logger.error(f"Error making API request: {e}")
            return []
    
    def get_forex_news(self, limit: int = 50, from_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get forex news from FMP API with improved error handling.
        
        Args:
            limit (int, optional): The maximum number of news items to retrieve. Defaults to 50.
            from_date (Optional[str], optional): Get news from this date onwards in format YYYY-MM-DD.
                Defaults to None (get the most recent news).
                
        Returns:
            List[Dict[str, Any]]: The forex news items.
        """
        logger.info(f"Fetching forex news for {self.symbol}")
        
        # Parse currency pair
        try:
            base_currency = self.symbol[:3]
            quote_currency = self.symbol[3:]
        except IndexError:
            logger.error(f"Invalid symbol format: {self.symbol}")
            base_currency = "EUR"
            quote_currency = "USD"
        
        # Create cache key
        if from_date:
            cache_key = f"forex_news_{base_currency}_{quote_currency}_{from_date}_{limit}.json"
        else:
            cache_key = f"forex_news_{base_currency}_{quote_currency}_{limit}.json"
        
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        # Check if we have cached data that's recent enough (less than 6 hours old)
        if os.path.exists(cache_path):
            file_mod_time = os.path.getmtime(cache_path)
            if time.time() - file_mod_time < 21600:  # 6 hours in seconds
                logger.info(f"Loading cached news from {cache_path}")
                try:
                    with open(cache_path, "r", encoding='utf-8') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading cached news: {e}")
                    # Continue to fetch new data if cache load fails
        
        # Prepare parameters
        params = {"limit": limit}
        if from_date:
            params["from"] = from_date
        
        # Make API request for forex news
        news = self._make_request("fx_news", params)
        
        if not news:
            logger.warning("Failed to get forex news, retrying with general FX search")
            # If no specific news found, try general financial news with currency filters
            params = {"limit": limit * 2}  # Get more to filter
            if from_date:
                params["from"] = from_date
            
            general_news = self._make_request("fmp/articles", params)
            
            # Filter for relevant currency news
            news = []
            
            # Get currency keywords - handle errors
            base_keywords = self.currency_keywords.get(base_currency, [])
            quote_keywords = self.currency_keywords.get(quote_currency, [])
            combined_keywords = base_keywords + quote_keywords
            
            if not combined_keywords:
                logger.warning(f"No keywords found for {base_currency} or {quote_currency}")
                # Use default keywords if none found for the currencies
                combined_keywords = ["forex", "currency", "exchange rate", "fx", "foreign exchange"]
            
            # Safely compile patterns, skipping any that cause errors
            currency_patterns = []
            for kw in combined_keywords:
                try:
                    pattern = re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE)
                    currency_patterns.append(pattern)
                except Exception as e:
                    logger.error(f"Error compiling regex for keyword {kw}: {e}")
            
            # Filter news with error handling
            for item in general_news:
                if not isinstance(item, dict):
                    continue
                
                title = item.get("title", "")
                text = item.get("text", "")
                
                if not isinstance(title, str) or not isinstance(text, str):
                    continue
                    
                content_text = f"{title} {text}"
                
                # Check if any pattern matches
                matches = False
                for pattern in currency_patterns:
                    try:
                        if pattern.search(content_text):
                            matches = True
                            break
                    except Exception as e:
                        logger.error(f"Error matching pattern: {e}")
                
                if matches:
                    news.append(item)
            
            # Limit to requested number
            news = news[:limit]
        
        # Save to cache with error handling
        try:
            with open(cache_path, "w", encoding='utf-8') as f:
                json.dump(news, f)
            logger.info(f"Saved {len(news)} news items to cache")
        except Exception as e:
            logger.error(f"Error saving news to cache: {e}")
        
        return news
    
    def analyze_news_sentiment(self, news: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze sentiment of news items with improved error handling.
        
        Args:
            news (List[Dict[str, Any]]): The news items to analyze.
            
        Returns:
            Dict[str, Any]: The sentiment analysis results.
        """
        logger.info(f"Analyzing sentiment for {len(news)} news items")
        
        if not news:
            return {
                "overall_score": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "total_count": 0,
                "weighted_score": 0,
                "base_currency_score": 0,
                "quote_currency_score": 0,
                "pair_impact": 0,
                "recent_trend": "neutral",
                "items": []
            }
        
        # Parse currency pair safely
        try:
            base_currency = self.symbol[:3]
            quote_currency = self.symbol[3:]
        except:
            logger.error(f"Invalid symbol format: {self.symbol}")
            base_currency = "EUR"
            quote_currency = "USD"
        
        # Process each news item
        analyzed_items = []
        total_score = 0
        base_currency_score = 0
        quote_currency_score = 0
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        for item in news:
            if not isinstance(item, dict):
                continue
                
            # Get text content safely
            title = item.get("title", "")
            content = item.get("content", item.get("text", ""))
            
            if not isinstance(title, str) or not isinstance(content, str):
                continue
                
            full_text = f"{title} {content}"
            
            # Calculate sentiment score using keyword analysis with error handling
            try:
                positive_count = sum(1 for kw in self.positive_keywords if re.search(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE))
                negative_count = sum(1 for kw in self.negative_keywords if re.search(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE))
            except Exception as e:
                logger.error(f"Error in keyword matching: {e}")
                positive_count = 0
                negative_count = 0
            
            # Currency specific sentiment with error handling
            base_kws = self.currency_keywords.get(base_currency, [])
            quote_kws = self.currency_keywords.get(quote_currency, [])
            
            try:
                base_mentions = sum(1 for kw in base_kws if re.search(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE))
                quote_mentions = sum(1 for kw in quote_kws if re.search(r'\b' + re.escape(kw) + r'\b', full_text, re.IGNORECASE))
            except Exception as e:
                logger.error(f"Error in currency keyword matching: {e}")
                base_mentions = 0
                quote_mentions = 0
            
            # Calculate raw sentiment score with division by zero protection
            raw_score = positive_count - negative_count
            denominator = positive_count + negative_count + 1  # +1 to avoid division by zero
            normalized_score = raw_score / denominator if denominator > 1 else 0
            
            # Determine sentiment label
            if normalized_score > 0.2:
                sentiment = "bullish"
                bullish_count += 1
            elif normalized_score < -0.2:
                sentiment = "bearish"
                bearish_count += 1
            else:
                sentiment = "neutral"
                neutral_count += 1
            
            # Calculate currency-specific impact
            if base_mentions > 0 and sentiment != "neutral":
                base_impact = normalized_score * base_mentions
                base_currency_score += base_impact
            
            if quote_mentions > 0 and sentiment != "neutral":
                # Invert for quote currency (positive for base is negative for quote)
                quote_impact = -normalized_score * quote_mentions
                quote_currency_score += quote_impact
            
            # Add timestamp in standard format
            timestamp = item.get("date", item.get("publishedDate", ""))
            if isinstance(timestamp, str):
                try:
                    # Try to parse various date formats
                    timestamp = pd.to_datetime(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                except:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            else:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Add to analyzed items
            analyzed_items.append({
                "title": title,
                "timestamp": timestamp,
                "url": item.get("url", ""),
                "source": item.get("site", item.get("source", "")),
                "sentiment": sentiment,
                "score": normalized_score,
                "base_currency_mentions": base_mentions,
                "quote_currency_mentions": quote_mentions
            })
            
            # Add to total score
            total_score += normalized_score
        
        # Calculate overall metrics with protection against empty lists
        total_count = len(news)
        overall_score = total_score / total_count if total_count > 0 else 0
        
        # Calculate weighted score (more recent news has higher weight)
        weighted_score = 0
        if analyzed_items:
            try:
                # Sort by timestamp (most recent first)
                # Use error handling to ensure we can sort
                sorted_items = sorted(
                    analyzed_items, 
                    key=lambda x: pd.to_datetime(x.get("timestamp", "1970-01-01 00:00:00")), 
                    reverse=True
                )
                
                # Apply exponential decay weights
                decay_factor = 0.9
                weight_sum = 0
                
                for i, item in enumerate(sorted_items):
                    weight = decay_factor ** i
                    score = item.get("score", 0)
                    weighted_score += score * weight
                    weight_sum += weight
                
                weighted_score /= weight_sum if weight_sum > 0 else 1
            except Exception as e:
                logger.error(f"Error calculating weighted score: {e}")
                weighted_score = overall_score
        
        # Determine recent trend with error handling
        recent_trend = "neutral"
        try:
            recent_items = analyzed_items[:min(5, len(analyzed_items))]
            recent_bullish = sum(1 for item in recent_items if item.get("sentiment") == "bullish")
            recent_bearish = sum(1 for item in recent_items if item.get("sentiment") == "bearish")
            
            if recent_bullish > recent_bearish:
                recent_trend = "bullish"
            elif recent_bearish > recent_bullish:
                recent_trend = "bearish"
        except Exception as e:
            logger.error(f"Error determining recent trend: {e}")
        
        # Calculate pair impact (positive means bullish for the pair)
        # Base currency positive = bullish for pair
        # Quote currency positive = bearish for pair
        pair_impact = base_currency_score - quote_currency_score
        
        return {
            "overall_score": overall_score,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
            "neutral_count": neutral_count,
            "total_count": total_count,
            "weighted_score": weighted_score,
            "base_currency_score": base_currency_score,
            "quote_currency_score": quote_currency_score,
            "pair_impact": pair_impact,
            "recent_trend": recent_trend,
            "items": analyzed_items
        }
    
    def get_sentiment(self, limit: int = 50, from_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get and analyze sentiment for the forex pair.
        
        Args:
            limit (int, optional): The maximum number of news items to retrieve. Defaults to 50.
            from_date (Optional[str], optional): Get news from this date onwards in format YYYY-MM-DD.
                Defaults to None (get the most recent news).
                
        Returns:
            Dict[str, Any]: The sentiment analysis results.
        """
        # Get forex news
        news = self.get_forex_news(limit=limit, from_date=from_date)
        
        # Analyze sentiment
        sentiment = self.analyze_news_sentiment(news)
        
        return sentiment
    
    def get_sentiment_signal(self) -> Dict[str, Any]:
        """
        Get a trading signal based on sentiment analysis with improved error handling.
        
        Returns:
            Dict[str, Any]: The sentiment signal.
        """
        # Get sentiment for the last 3 days
        from_date = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        sentiment = self.get_sentiment(limit=30, from_date=from_date)
        
        # Calculate signal strength
        signal_strength = 0.0
        
        # Use weighted score for base signal
        base_signal = sentiment.get("weighted_score", 0)
        
        # Boost signal if recent trend is strong
        recent_trend = sentiment.get("recent_trend", "neutral") 
        if recent_trend == "bullish":
            base_signal += 0.2
        elif recent_trend == "bearish":
            base_signal -= 0.2
        
        # Add currency pair impact with safe accessor
        pair_impact = sentiment.get("pair_impact", 0)
        signal_strength = base_signal + (pair_impact * 0.3)
        
        # Normalize signal strength to [-1, 1]
        signal_strength = max(min(signal_strength, 1.0), -1.0)
        
        # Determine signal type
        if signal_strength > 0.3:
            signal = "buy"
        elif signal_strength < -0.3:
            signal = "sell"
        else:
            signal = "neutral"
        
        # Calculate confidence level (0 to 1)
        confidence = abs(signal_strength)
        
        return {
            "symbol": self.symbol,
            "signal": signal,
            "strength": signal_strength,
            "confidence": confidence,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sentiment_data": sentiment
        }