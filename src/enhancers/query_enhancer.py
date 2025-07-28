"""
Query enhancer that adds context and improves tool selection
"""

import logging
from typing import Dict, Any, Optional

from ..enhancers.trading_patterns import TradingPatternMatcher
from ..enhancers.context_analyzer import ContextAnalyzer

logger = logging.getLogger(__name__)


class QueryEnhancer:
    """
    Enhances user queries with context to improve tool selection
    """
    
    def __init__(self):
        """Initialize query enhancer"""
        self.pattern_matcher = TradingPatternMatcher()
        self.context_analyzer = ContextAnalyzer()
        
        logger.info("Initialized QueryEnhancer")
    
    def enhance_query(
        self, 
        query: str, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Enhance a query with context and trading-specific information
        
        Args:
            query: Original user query
            context: Optional context dictionary
            
        Returns:
            Enhanced query string
        """
        if not query.strip():
            return query
        
        try:
            # Analyze patterns in the query
            pattern_matches = self.pattern_matcher.match_patterns(query)
            
            # Analyze context if provided
            context_insights = {}
            if context:
                context_insights = self.context_analyzer.analyze_context(context)
            
            # Build enhanced query
            enhanced_query = self._build_enhanced_query(
                original_query=query,
                pattern_matches=pattern_matches,
                context_insights=context_insights
            )
            
            logger.debug(f"Enhanced query: '{query}' -> '{enhanced_query}'")
            return enhanced_query
            
        except Exception as e:
            logger.error(f"Query enhancement failed: {e}")
            return query  # Return original query on failure
    
    def _build_enhanced_query(
        self,
        original_query: str,
        pattern_matches: Dict[str, Any],
        context_insights: Dict[str, Any]
    ) -> str:
        """
        Build enhanced query from analysis results
        
        Args:
            original_query: Original user query
            pattern_matches: Results from pattern matching
            context_insights: Results from context analysis
            
        Returns:
            Enhanced query string
        """
        parts = [original_query]
        
        # Add pattern-based enhancements
        if pattern_matches.get("suggested_enhancements"):
            enhancement_text = ", ".join(pattern_matches["suggested_enhancements"][:3])
            parts.append(f"- {enhancement_text}")
        
        # Add time context
        if pattern_matches.get("time_context"):
            time_name = pattern_matches["time_context"]["name"]
            parts.append(f"considering {time_name} market conditions")
        
        # Add context-based enhancements
        if context_insights.get("context_enhancements"):
            context_text = ", ".join(context_insights["context_enhancements"][:2])
            parts.append(f"with focus on {context_text}")
        
        # Add ticker context if available
        if context_insights.get("ticker"):
            ticker = context_insights["ticker"]
            parts.append(f"for {ticker}")
        
        return " ".join(parts)
    
    def get_enhancement_info(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Get detailed information about query enhancement
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Dictionary with enhancement details
        """
        pattern_matches = self.pattern_matcher.match_patterns(query)
        context_insights = self.context_analyzer.analyze_context(context) if context else {}
        
        return {
            "original_query": query,
            "enhanced_query": self.enhance_query(query, context),
            "matched_patterns": pattern_matches.get("matched_patterns", []),
            "suggested_tools": pattern_matches.get("suggested_tools", []),
            "time_context": pattern_matches.get("time_context"),
            "context_insights": context_insights,
        }