"""
Specialized RAG-MCP agent for trading applications
"""

import logging
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from src.core.retriever import RAGMCPRetriever
from src.enhancers.query_enhancer import QueryEnhancer
from src.models.query_context import QueryContext
from src.storage.base_store import BaseStore
from src.enhancers.trading_patterns import TradingPatternMatcher

logger = logging.getLogger(__name__)


class TradingAgentRAGMCP:
    """
    Specialized RAG-MCP agent optimized for trading applications
    """
    
    def __init__(
        self,
        storage: Optional[BaseStore] = None,
        ticker: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        similarity_threshold: float = 0.3,
        max_tools: int = 4
    ):
        """
        Initialize trading-focused RAG-MCP agent
        
        Args:
            storage: Storage backend for persistence
            ticker: Default ticker symbol
            tools: List of available tools
            similarity_threshold: Minimum similarity for tool selection
            max_tools: Maximum tools to return
        """
        self.ticker = ticker
        self.storage = storage
        
        # Initialize core components
        self.retriever = RAGMCPRetriever(
            similarity_threshold=similarity_threshold,
            max_tools=max_tools
        )
        self.query_enhancer = QueryEnhancer()
        
        # Add tools if provided
        if tools:
            self.add_tools(tools)
        
        # Load from storage if available
        if storage:
            self._load_from_storage()
        
        logger.info(f"Initialized TradingAgentRAGMCP for ticker: {ticker}")
    
    def add_tools(self, tools: List[Dict[str, Any]]) -> None:
        """
        Add tools to the agent
        
        Args:
            tools: List of MCP tool specifications
        """
        self.retriever.add_tools(tools)
        
        # Save to storage if available
        if self.storage:
            self._save_to_storage()
        
        logger.info(f"Added {len(tools)} tools to trading agent")
    
    def get_tools_for_query(
        self, 
        query: str, 
        context: Optional[QueryContext] = None
    ) -> Tuple[List[Dict[str, Any]], str]:
        """
        Get relevant tools and enhanced query for a trading query
        
        Args:
            query: User query
            context: Optional trading context
            
        Returns:
            Tuple of (filtered_tools, enhanced_query)
        """
        try:
            # Get current time for market context
            current_time = datetime.fromtimestamp(context.current_time) if context and context.current_time else datetime.now()
            
            # Enhance query with basic trading context first
            context_dict = context.to_dict() if context else None
            enhanced_query = self.query_enhancer.enhance_query(query, context_dict)
            
            # Further enhance with tool-specific context
            pattern_matcher = TradingPatternMatcher()
            tool_enhanced_query = pattern_matcher.enhance_query_with_tool_context(
                enhanced_query, 
                self.ticker, 
                current_time
            )
            
            # Get priority tools for this query
            priority_tools = pattern_matcher.get_priority_tools(query)
            
            # Get relevant tools using tool-enhanced query with priorities
            retrieved_tools = self.retriever.retrieve_tools(
                tool_enhanced_query, 
                include_reasoning=False,
                priority_tools=priority_tools
            )
            relevant_tools = [rt.tool_spec for rt in retrieved_tools]
            
            # Add memory/context tools if relevant
            if context and context.has_active_recommendation():
                relevant_tools = self._maybe_add_context_tools(relevant_tools, context)
            
            logger.info(
                f"Selected {len(relevant_tools)} tools for query: '{query[:50]}...'"
            )
            logger.info(f"Priority tools used: {priority_tools}")
            
            return relevant_tools, tool_enhanced_query
            
        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return [], query
    
    def get_detailed_analysis(
        self, 
        query: str, 
        context: Optional[QueryContext] = None
    ) -> Dict[str, Any]:
        """
        Get detailed analysis of tool selection process
        
        Args:
            query: User query
            context: Optional context
            
        Returns:
            Detailed analysis dictionary
        """
        context_dict = context.to_dict() if context else None
        
        # Get enhancement details
        enhancement_info = self.query_enhancer.get_enhancement_info(query, context_dict)
        
        # Get priority tools and ORB context
        pattern_matcher = TradingPatternMatcher()
        priority_tools = pattern_matcher.get_priority_tools(query)
        orb_context = pattern_matcher.get_orb_context()
        
        # Get retrieval details with priorities
        retrieved_tools = self.retriever.retrieve_tools(
            enhancement_info["enhanced_query"], 
            include_reasoning=True,
            priority_tools=priority_tools
        )
        
        return {
            "original_query": query,
            "enhanced_query": enhancement_info["enhanced_query"],
            "enhancement_info": enhancement_info,
            "priority_tools": priority_tools,
            "orb_context": orb_context,
            "retrieved_tools": [
                {
                    "name": rt.tool_name,
                    "similarity_score": rt.similarity_score,
                    "reasoning": rt.reasoning,
                    "is_priority": rt.tool_name in priority_tools if priority_tools else False
                }
                for rt in retrieved_tools
            ],
            "total_tools_available": len(self.retriever.tool_index),
            "context_summary": self._summarize_context(context) if context else None
        }
    
    def _maybe_add_context_tools(
        self, 
        tools: List[Dict[str, Any]], 
        context: QueryContext
    ) -> List[Dict[str, Any]]:
        """
        Add context-aware tools if appropriate
        
        Args:
            tools: Currently selected tools
            context: Query context
            
        Returns:
            Updated tools list
        """
        # Extract tool names for checking
        tool_names = [
            tool.get("toolSpec", {}).get("name", "") 
            for tool in tools
        ]
        
        # Add recommendation history tool if asking about status/validation
        status_keywords = ["valid", "still", "status", "change", "update"]
        query_lower = context.bot_state.get("last_query", "").lower()
        
        if any(keyword in query_lower for keyword in status_keywords):
            # Try to add memory/history tools if not already present
            memory_tools = ["recommendation_history_tool", "get_trading_bias"]
            for memory_tool in memory_tools:
                if memory_tool not in tool_names:
                    # Find this tool in our index and add it
                    for indexed_tool in self.retriever.tool_index.tools:
                        if indexed_tool.get("toolSpec", {}).get("name") == memory_tool:
                            tools.append(indexed_tool)
                            break
        
        return tools
    
    def _summarize_context(self, context: QueryContext) -> Dict[str, Any]:
        """Summarize context for analysis"""
        return {
            "ticker": context.ticker,
            "has_recommendation": context.has_active_recommendation(),
            "conversation_turns": len(context.conversation_history),
            "market_session": context.market_session,
        }
    
    def _save_to_storage(self) -> None:
        """Save agent state to storage"""
        if not self.storage:
            return
        
        try:
            # Save tool index data
            data = {
                "tools": self.retriever.tool_index.tools,
                "tool_texts": self.retriever.tool_index.tool_texts,
                "config": {
                    "similarity_threshold": self.retriever.similarity_threshold,
                    "max_tools": self.retriever.max_tools,
                    "ticker": self.ticker
                }
            }
            
            self.storage.save_agent_data("trading_agent", data)
            logger.debug("Saved agent data to storage")
            
        except Exception as e:
            logger.error(f"Failed to save to storage: {e}")
    
    def _load_from_storage(self) -> None:
        """Load agent state from storage"""
        if not self.storage:
            return
        
        try:
            data = self.storage.load_agent_data("trading_agent")
            if data and "tools" in data:
                # Reload tools
                tools = data["tools"]
                if tools:
                    self.retriever.add_tools(tools)
                    logger.info(f"Loaded {len(tools)} tools from storage")
                
                # Restore config
                config = data.get("config", {})
                if "similarity_threshold" in config:
                    self.retriever.similarity_threshold = config["similarity_threshold"]
                
        except Exception as e:
            logger.error(f"Failed to load from storage: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get agent statistics"""
        stats = self.retriever.get_index_stats()
        stats.update({
            "ticker": self.ticker,
            "storage_enabled": self.storage is not None,
        })
        return stats
    
    def __repr__(self) -> str:
        return (
            f"TradingAgentRAGMCP("
            f"ticker='{self.ticker}', "
            f"tools={len(self.retriever.tool_index)}"
            f")"
        )