"""
Context analyzer for understanding trading state and history
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ContextAnalyzer:
    """
    Analyzes trading context to provide relevant enhancements
    """
    
    def __init__(self):
        """Initialize context analyzer"""
        logger.debug("Initialized ContextAnalyzer")
    
    def analyze_context(self, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze context and extract relevant insights
        
        Args:
            context: Context dictionary with trading state
            
        Returns:
            Dictionary with context insights
        """
        if not context:
            return {}
        
        insights = {
            "ticker": context.get("ticker"),
            "has_recommendation": False,
            "has_conversation_history": False,
            "context_enhancements": [],
            "priority_tools": []
        }
        
        try:
            # Analyze recommendation context
            recommendation = context.get("recommendation")
            if recommendation:
                insights["has_recommendation"] = True
                rec_insights = self._analyze_recommendation_context(recommendation)
                insights["context_enhancements"].extend(rec_insights["enhancements"])
                insights["priority_tools"].extend(rec_insights["tools"])
            
            # Analyze conversation history
            history = context.get("conversation_history", [])
            if history:
                insights["has_conversation_history"] = True
                hist_insights = self._analyze_conversation_history(history)
                insights["context_enhancements"].extend(hist_insights["enhancements"])
                insights["priority_tools"].extend(hist_insights["tools"])
            
            # Analyze time context
            current_time = context.get("current_time")
            if current_time:
                time_insights = self._analyze_time_context(current_time)
                insights["context_enhancements"].extend(time_insights["enhancements"])
            
            # Remove duplicates
            insights["context_enhancements"] = list(set(insights["context_enhancements"]))
            insights["priority_tools"] = list(set(insights["priority_tools"]))
            
        except Exception as e:
            logger.error(f"Context analysis failed: {e}")
        
        return insights
    
    def _analyze_recommendation_context(self, recommendation: Dict[str, Any]) -> Dict[str, List[str]]:
        """Analyze recommendation context"""
        enhancements = []
        tools = []
        
        # Check recommendation type and add relevant context
        rec_type = recommendation.get("recommendation_type", "").upper()
        if "CALL" in rec_type:
            enhancements.extend([
                "verify bullish thesis conditions",
                "check support levels holding",
                "examine call option activity"
            ])
            tools.extend(["greeks_tool", "options_order_flow", "market_structure"])
        elif "PUT" in rec_type:
            enhancements.extend([
                "verify bearish thesis conditions", 
                "check resistance levels holding",
                "examine put option activity"
            ])
            tools.extend(["greeks_tool", "options_order_flow", "market_structure"])
        
        # Check conviction level
        conviction = recommendation.get("conviction", "").upper()
        if conviction in ["HIGH", "MEDIUM"]:
            enhancements.append("validate high-conviction recommendation")
            tools.append("market_structure")
        
        return {"enhancements": enhancements, "tools": tools}
    
    def _analyze_conversation_history(self, history: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze conversation history for context"""
        enhancements = []
        tools = []
        
        if not history:
            return {"enhancements": enhancements, "tools": tools}
        
        # Look at recent conversation topics
        recent_topics = set()
        for turn in history[-3:]:  # Last 3 turns
            user_msg = turn.get("user_message", "").lower()
            assistant_msg = turn.get("assistant_message", "").lower()
            
            # Extract topics from messages
            combined_text = f"{user_msg} {assistant_msg}"
            
            if any(word in combined_text for word in ["greeks", "options", "calls", "puts"]):
                recent_topics.add("options")
            if any(word in combined_text for word in ["volume", "flow", "buying", "selling"]):
                recent_topics.add("volume")
            if any(word in combined_text for word in ["support", "resistance", "levels"]):
                recent_topics.add("levels")
        
        # Add context based on recent topics
        if "options" in recent_topics:
            enhancements.append("continue options analysis thread")
            tools.append("greeks_insights")
        if "volume" in recent_topics:
            enhancements.append("follow up on volume analysis")
            tools.append("equity_order_flow")
        if "levels" in recent_topics:
            enhancements.append("update key level analysis")
            tools.append("technical_zones")
        
        if history:
            enhancements.append("maintain conversation context")
        
        return {"enhancements": enhancements, "tools": tools}
    
    def _analyze_time_context(self, timestamp: float) -> Dict[str, List[str]]:
        """Analyze time-based context"""
        enhancements = []
        
        try:
            dt = datetime.fromtimestamp(timestamp)
            current_time = dt.time()
            
            # Market hours context
            if current_time.hour == 9 and current_time.minute >= 30:
                enhancements.append("focus on market opening dynamics")
            elif 15 <= current_time.hour < 16:
                enhancements.append("consider closing hour positioning")
            elif current_time.hour >= 16:
                enhancements.append("analyze after-hours implications")
            
        except Exception as e:
            logger.debug(f"Time context analysis failed: {e}")
        
        return {"enhancements": enhancements}