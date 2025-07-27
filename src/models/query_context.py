"""
Query context model for trading applications
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional


@dataclass
class QueryContext:
    """
    Context information for query processing in trading applications
    """
    ticker: str
    current_time: float
    recommendation: Optional[Dict[str, Any]] = None
    conversation_history: Optional[List[Dict[str, Any]]] = None
    bot_state: Optional[Dict[str, Any]] = None
    user_preferences: Optional[Dict[str, Any]] = None
    market_session: Optional[str] = None  # "pre_market", "regular", "after_hours"
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.conversation_history is None:
            self.conversation_history = []
        if self.bot_state is None:
            self.bot_state = {}
        if self.user_preferences is None:
            self.user_preferences = {}
    
    def has_active_recommendation(self) -> bool:
        """Check if there's an active recommendation"""
        return self.recommendation is not None
    
    def get_recent_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        if not self.conversation_history:
            return []
        return self.conversation_history[-limit:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "ticker": self.ticker,
            "current_time": self.current_time,
            "recommendation": self.recommendation,
            "conversation_history": self.conversation_history,
            "bot_state": self.bot_state,
            "user_preferences": self.user_preferences,
            "market_session": self.market_session,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueryContext":
        """Create from dictionary"""
        return cls(
            ticker=data["ticker"],
            current_time=data["current_time"],
            recommendation=data.get("recommendation"),
            conversation_history=data.get("conversation_history"),
            bot_state=data.get("bot_state"),
            user_preferences=data.get("user_preferences"),
            market_session=data.get("market_session"),
        )