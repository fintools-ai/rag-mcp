"""
Trading-specific pattern matching for query enhancement
"""

import re
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, time

logger = logging.getLogger(__name__)


class TradingPatternMatcher:
    """
    Recognizes trading-specific patterns in queries and suggests enhancements
    """
    
    def __init__(self):
        """Initialize pattern matcher with trading-specific patterns"""
        self.patterns = self._initialize_patterns()
        self.time_contexts = self._initialize_time_contexts()
        
    def _initialize_patterns(self) -> Dict[str, Dict]:
        """Initialize query pattern definitions"""
        return {
            # Market overview patterns - prioritize options flow
            "market_overview": {
                "patterns": [
                    r"\b(market|trend|direction|bias)\b",
                    r"\b(overall|general|current)\s+(market|view|picture)\b",
                    r"\b(what.*look.*like|how.*market)\b",
                    r"\b(market.*now|current.*state)\b",
                    r"\b(market.*conditions|sentiment)\b"
                ],
                "enhancements": [
                    "prioritize options flow for sentiment analysis",
                    "analyze current order flow patterns with ORB context",
                    "examine technical indicators for ORB setups", 
                    "check institutional vs retail activity during range formation",
                    "review volume profile and ORB levels",
                    "check memory for context and previous ORB analysis"
                ],
                "suggested_tools": ["options_order_flow_tool", "equity_order_flow_tool", "technical_analysis", "volume_profile", "memory_tools"],
                "priority_tools": ["options_order_flow_tool"]
            },

            # ORB-specific patterns
            "orb_analysis": {
                "patterns": [
                    r"\b(orb|opening.*range|range.*break)\b",
                    r"\b(first.*30|opening.*30|30.*minute)\b",
                    r"\b(breakout|breakdown|range.*high|range.*low)\b",
                    r"\b(morning.*range|opening.*levels)\b"
                ],
                "enhancements": [
                    "focus on ORB range establishment and breakout potential",
                    "analyze options flow during range formation",
                    "monitor institutional positioning for ORB setups",
                    "examine volume confirmation on potential breakouts",
                    "track key ORB levels and support/resistance"
                ],
                "suggested_tools": ["options_order_flow_tool", "financial_technical_zones", "equity_order_flow_tool", "financial_volume_profile"],
                "priority_tools": ["options_order_flow_tool"]
            },
            
            # Options-specific patterns
            "options_analysis": {
                "patterns": [
                    r"\b(greeks|delta|gamma|theta|vega|options)\b",
                    r"\b(calls?|puts?|strikes?)\b",
                    r"\b(implied.*volatility|iv|skew)\b",
                    r"\b(0dte|dte|expiration)\b",
                    r"\b(options.*flow|flow.*options)\b"
                ],
                "enhancements": [
                    "prioritize real-time options flow analysis",
                    "analyze options Greeks positioning for ORB context",
                    "examine implied volatility patterns during market sessions",
                    "check unusual options activity for directional bias",
                    "review options order flow for institutional sentiment"
                ],
                "suggested_tools": ["options_order_flow_tool"],
                "priority_tools": ["options_order_flow_tool"]
            },
            
            # Volume analysis patterns  
            "volume_analysis": {
                "patterns": [
                    r"\b(volume|vol|flow|absorption)\b",
                    r"\b(buying|selling|pressure)\b",
                    r"\b(institutional|retail|smart.*money)\b",
                    r"\b(poc|vah|val|profile)\b"
                ],
                "enhancements": [
                    "examine volume distribution",
                    "analyze order flow imbalances",
                    "check institutional vs retail activity",
                    "review volume profile key levels"
                ],
                "suggested_tools": ["equity_order_flow", "volume_profile", "technical_zones"]
            },
            
            # Status/validation patterns
            "status_check": {
                "patterns": [
                    r"\b(still.*valid|remains.*valid|thesis.*hold)\b",
                    r"\b(any.*chang|what.*chang|update|status)\b",
                    r"\b(still.*good|recommendation.*valid)\b",
                    r"\b(holding|maintain|keep)\b"
                ],
                "enhancements": [
                    "compare to original recommendation from memory",
                    "check current order flow vs original thesis",
                    "verify key technical levels still hold",
                    "examine recent market developments",
                    "analyze options flow for sentiment shifts"
                ],
                "suggested_tools": ["memory_tools", "equity_order_flow", "options_order_flow", "technical_analysis"]
            },
            
            # Monitoring patterns
            "monitoring_request": {
                "patterns": [
                    r"\b(monitor|watch|track|alert)\b",
                    r"\b(specific.*strike|strike.*\d+)\b",
                    r"\b(set.*up|configure|start.*tracking)\b"
                ],
                "enhancements": [
                    "configure monitoring for specific levels",
                    "set up alerts for unusual activity", 
                    "track key strike prices",
                    "monitor both calls and puts"
                ],
                "suggested_tools": ["options_monitoring_tool"]
            }
        }
    
    def _initialize_time_contexts(self) -> Dict[str, Dict]:
        """Initialize time-based context enhancements with ORB focus"""
        return {
            "pre_market": {
                "time_range": (time(4, 0), time(9, 30)),
                "enhancements": [
                    "analyze pre-market options flow for sentiment",
                    "check overnight positioning changes",
                    "examine gap setup potential",
                    "monitor unusual pre-market options activity"
                ],
                "orb_context": "pre_market_setup"
            },
            "market_open": {
                "time_range": (time(9, 30), time(10, 0)),
                "enhancements": [
                    "focus on opening range breakout (ORB) analysis",
                    "monitor first 30-minute range establishment",
                    "analyze options flow during range formation",
                    "track institutional opening positioning",
                    "examine overnight gaps and ORB potential"
                ],
                "orb_context": "range_formation"
            },
            "orb_breakout": {
                "time_range": (time(10, 0), time(10, 15)),
                "enhancements": [
                    "monitor ORB breakout attempts and confirmations",
                    "analyze options flow during breakout",
                    "check volume confirmation on breakouts", 
                    "track institutional vs retail flow during moves"
                ],
                "orb_context": "breakout_monitoring"
            },
            "mid_morning": {
                "time_range": (time(10, 30), time(11, 30)), 
                "enhancements": [
                    "check for institutional positioning post-ORB",
                    "analyze momentum sustainability with options flow",
                    "monitor for pullback opportunities"
                ],
                "orb_context": "trend_continuation"
            },
            "lunch_time": {
                "time_range": (time(11, 30), time(14, 0)),
                "enhancements": [
                    "focus on range-bound activity and consolidation",
                    "check for breakout setups from lunch consolidation",
                    "analyze options flow during low volume periods",
                    "monitor for afternoon breakout potential"
                ],
                "orb_context": "consolidation_phase"
            },
            "power_hour": {
                "time_range": (time(15, 0), time(16, 0)),
                "enhancements": [
                    "focus on closing positioning and EOD flows",
                    "examine end-of-day options flow patterns",
                    "check for late-day ORB extensions or reversals",
                    "analyze institutional closing activity"
                ],
                "orb_context": "closing_phase"
            }
        }
    
    def match_patterns(self, query: str) -> Dict[str, Any]:
        """
        Match query against trading patterns
        
        Args:
            query: User query to analyze
            
        Returns:
            Dictionary with matched patterns and suggestions
        """
        query_lower = query.lower()
        matches = {
            "matched_patterns": [],
            "suggested_enhancements": [],
            "suggested_tools": [],
            "time_context": None
        }
        
        # Check pattern matches
        for pattern_name, pattern_info in self.patterns.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, query_lower):
                    matches["matched_patterns"].append(pattern_name)
                    matches["suggested_enhancements"].extend(pattern_info["enhancements"])
                    matches["suggested_tools"].extend(pattern_info["suggested_tools"])
                    break
        
        # Add time context
        current_time = datetime.now().time()
        time_context = self._get_time_context(current_time)
        if time_context:
            matches["time_context"] = time_context
            matches["suggested_enhancements"].extend(time_context["enhancements"])
        
        # Remove duplicates
        matches["suggested_enhancements"] = list(set(matches["suggested_enhancements"]))
        matches["suggested_tools"] = list(set(matches["suggested_tools"]))
        
        return matches
    
    def _get_time_context(self, current_time: time) -> Optional[Dict]:
        """Get time-based context for the current time"""
        for context_name, context_info in self.time_contexts.items():
            start_time, end_time = context_info["time_range"]
            if start_time <= current_time <= end_time:
                return {
                    "name": context_name,
                    "enhancements": context_info["enhancements"]
                }
        return None
    
    def suggest_tool_categories(self, query: str) -> List[str]:
        """
        Suggest tool categories based on query content
        
        Args:
            query: User query
            
        Returns:
            List of suggested tool categories
        """
        matches = self.match_patterns(query)
        return matches["suggested_tools"]
    
    def get_orb_context(self, current_time: Optional[time] = None) -> Optional[str]:
        """
        Get current ORB context based on market time
        
        Args:
            current_time: Time to check, defaults to now
            
        Returns:
            ORB context string or None
        """
        if current_time is None:
            current_time = datetime.now().time()
        
        time_context = self._get_time_context(current_time)
        if time_context and "orb_context" in self.time_contexts.get(time_context["name"], {}):
            return self.time_contexts[time_context["name"]]["orb_context"]
        return None
    
    def get_priority_tools(self, query: str) -> List[str]:
        """
        Get priority tools for a query, ALWAYS including both order flow tools
        
        Args:
            query: User query
            
        Returns:
            List of priority tool names
        """
        # ALWAYS start with core trading tools as priority
        priority_tools = ["options_order_flow_tool", "equity_order_flow_tool", "financial_technical_analysis"]
        
        # Add any additional priority tools from pattern matches
        query_lower = query.lower()
        for pattern_name, pattern_info in self.patterns.items():
            for pattern in pattern_info["patterns"]:
                if re.search(pattern, query_lower):
                    additional_tools = pattern_info.get("priority_tools", [])
                    priority_tools.extend(additional_tools)
                    break
        
        # Return unique list with order flow tools always first
        unique_tools = []
        seen = set()
        for tool in priority_tools:
            if tool not in seen:
                unique_tools.append(tool)
                seen.add(tool)
        
        return unique_tools
    
    def enhance_query_with_tool_context(self, query: str, ticker: str, current_time: Optional[datetime] = None) -> str:
        """
        Enhance query with specific context for available tools
        
        Args:
            query: Original user query
            ticker: Stock ticker symbol
            current_time: Current market time
            
        Returns:
            Enhanced query with tool-specific context
        """
        if current_time is None:
            current_time = datetime.now()
        
        market_time = current_time.time()
        
        # Tool-specific context based on what's available
        tool_context = []
        
        # For options_order_flow_tool - always prioritized
        tool_context.append(f"For options flow: Analyze {ticker} PUT vs CALL sentiment and positioning")
        
        # For equity_order_flow_tool - always prioritized  
        tool_context.append(f"For equity flow: Check {ticker} institutional vs retail activity, recent {10}-minute window")
        
        # For financial_technical_analysis - if technical query
        if any(word in query.lower() for word in ["technical", "indicators", "rsi", "macd", "trend"]):
            tool_context.append(f"For technical analysis: Current {ticker} momentum, trend direction indicators")
        
        # For market_structure_tool - exclude for market overview
        if not any(pattern in query.lower() for pattern in ["what.*look.*like", "market.*now", "current.*state"]):
            tool_context.append(f"For market structure: {ticker} support/resistance levels, bias analysis")
        
        # For financial_volume_profile - if volume/levels query
        if any(word in query.lower() for word in ["volume", "profile", "poc", "levels", "support", "resistance"]):
            tool_context.append(f"For volume profile: {ticker} key price levels (POC, VAH, VAL) on granular timeframes")
        
        # For greeks tools - if options query
        if any(word in query.lower() for word in ["greeks", "delta", "gamma", "options", "iv"]):
            tool_context.append(f"For Greeks: Initialize {ticker} monitoring first, then retrieve metrics and insights")
        
        # For memory tools - if status/validation query
        if any(word in query.lower() for word in ["status", "valid", "recommendation", "previous", "update"]):
            tool_context.append(f"For memory: Retrieve {ticker} previous recommendations and context")
        
        # Time-based context for tools
        session_context = self._get_session_tool_context(market_time)
        if session_context:
            tool_context.append(session_context)
        
        # Combine with original query
        enhanced_query = f"{query}\n\nTool Context:\n" + "\n".join(f"- {ctx}" for ctx in tool_context)
        
        return enhanced_query
    
    def _get_session_tool_context(self, market_time: time) -> Optional[str]:
        """Get session-specific tool context"""
        if time(9, 30) <= market_time < time(10, 0):
            return "Session focus: Opening range data, initial directional bias from flows"
        elif time(15, 30) <= market_time <= time(16, 0):
            return "Session focus: Closing positioning, end-of-day flow patterns"
        elif time(11, 30) <= market_time < time(14, 0):
            return "Session focus: Range-bound activity, consolidation patterns"
        return None
    
    def _get_market_session(self, market_time: time) -> Optional[str]:
        """Get current market session"""
        if time(4, 0) <= market_time < time(9, 30):
            return "Pre-market (gap analysis, overnight positioning)"
        elif time(9, 30) <= market_time < time(10, 0):
            return "Market open (ORB formation, initial directional bias)"
        elif time(10, 0) <= market_time < time(11, 30):
            return "Morning trend (ORB breakout confirmation, momentum)"
        elif time(11, 30) <= market_time < time(14, 0):
            return "Lunch consolidation (range trading, setup formation)"
        elif time(14, 0) <= market_time < time(15, 30):
            return "Afternoon session (trend continuation, reversal signals)"
        elif time(15, 30) <= market_time <= time(16, 0):
            return "Power hour (closing positioning, EOD flows)"
        return None
    
    def _get_volatility_context(self, market_time: time) -> Optional[str]:
        """Get volatility expectations based on time"""
        if time(9, 30) <= market_time < time(10, 30):
            return "High volatility expected (market open)"
        elif time(11, 30) <= market_time < time(14, 0):
            return "Low volatility expected (lunch period)"
        elif time(15, 30) <= market_time <= time(16, 0):
            return "High volatility expected (power hour)"
        return "Normal volatility expected"
    
    def _get_risk_context(self, query: str, market_time: time) -> List[str]:
        """Get risk management context"""
        risk_context = []
        
        # Position sizing context
        if any(word in query.lower() for word in ["buy", "sell", "trade", "position"]):
            risk_context.append("Risk: Check current portfolio exposure")
            risk_context.append("Risk: Verify position sizing rules")
        
        # Options risk context
        if any(word in query.lower() for word in ["options", "calls", "puts", "greeks"]):
            risk_context.append("Risk: Monitor Greeks exposure (delta, gamma, theta)")
            risk_context.append("Risk: Check implied volatility levels")
        
        # Time-based risk
        if time(15, 45) <= market_time <= time(16, 0):
            risk_context.append("Risk: Close to market close - consider overnight exposure")
        
        return risk_context
    
    def _get_execution_context(self, query: str, market_time: time) -> List[str]:
        """Get execution-focused context"""
        execution_context = []
        
        # Liquidity context
        if time(9, 30) <= market_time < time(10, 0):
            execution_context.append("Execution: High liquidity at open - watch for slippage")
        elif time(11, 30) <= market_time < time(14, 0):
            execution_context.append("Execution: Lower liquidity during lunch")
        
        # Market structure context
        if any(word in query.lower() for word in ["entry", "exit", "stop"]):
            execution_context.append("Execution: Check volume profile for support/resistance")
            execution_context.append("Execution: Verify order flow direction")
        
        # Options execution context
        if any(word in query.lower() for word in ["options", "strikes", "expiration"]):
            execution_context.append("Execution: Check bid/ask spreads and open interest")
        
        return execution_context
    
