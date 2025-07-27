"""
Main RAG-MCP retriever for filtering tools based on semantic similarity
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.core.embeddings import EmbeddingModel
from src.core.tool_index import ToolIndex

logger = logging.getLogger(__name__)


@dataclass
class RetrievedTool:
    """Represents a retrieved tool with metadata"""
    tool_spec: Dict[str, Any]
    similarity_score: float
    tool_name: str
    reasoning: Optional[str] = None


class RAGMCPRetriever:
    """
    Main RAG-MCP retriever that filters tools based on semantic similarity
    """
    
    def __init__(
        self, 
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.3,
        max_tools: int = 5
    ):
        """
        Initialize RAG-MCP Retriever
        
        Args:
            embedding_model: Name of the sentence transformer model
            similarity_threshold: Minimum similarity score for tool selection
            max_tools: Maximum number of tools to return
        """
        self.similarity_threshold = similarity_threshold
        self.max_tools = max_tools
        
        # Initialize embedding model and tool index
        self.embedding_model = EmbeddingModel(embedding_model)
        self.tool_index = ToolIndex(self.embedding_model)
        
        logger.info(f"Initialized RAGMCPRetriever with threshold={similarity_threshold}")
    
    def add_tool(self, tool_spec: Dict[str, Any]) -> None:
        """Add a tool to the index"""
        self.tool_index.add_tool(tool_spec)
    
    def add_tools(self, tool_specs: List[Dict[str, Any]]) -> None:
        """Add multiple tools to the index"""
        self.tool_index.add_tools(tool_specs)
        logger.info(f"Added {len(tool_specs)} tools to RAG-MCP index")
    
    def _should_exclude_tool(self, query: str, tool_name: str) -> bool:
        """
        Check if a tool should be excluded based on query patterns
        
        Args:
            query: User query string
            tool_name: Name of the tool to check
            
        Returns:
            True if tool should be excluded
        """
        import re
        
        query_lower = query.lower()
        
        # Market overview queries should exclude market_structure_tool
        market_overview_patterns = [
            r"\b(what.*look.*like|how.*market)\b",
            r"\b(market.*now|current.*state|market.*conditions)\b",
            r"\b(overall|general|current)\s+(market|view|picture)\b"
        ]
        
        # Check if query matches market overview patterns
        is_market_overview = any(re.search(pattern, query_lower) for pattern in market_overview_patterns)
        
        if is_market_overview and tool_name == "market_structure_tool":
            return True
        
            
        return False

    def retrieve_tools(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        include_reasoning: bool = False,
        priority_tools: Optional[List[str]] = None
    ) -> List[RetrievedTool]:
        """
        Retrieve the most relevant tools for a query
        
        Args:
            query: User query string
            top_k: Number of tools to retrieve (uses max_tools if None)
            include_reasoning: Whether to generate reasoning for selections
            priority_tools: List of tool names to prioritize
            
        Returns:
            List of RetrievedTool objects sorted by relevance
        """
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        if top_k is None:
            top_k = self.max_tools
        
        try:
            # Encode query
            query_embedding = self.embedding_model.encode_single(query)
            
            # Search for similar tools (get more to allow for prioritization)
            raw_results = self.tool_index.search(query_embedding, top_k * 3)
            
            # Separate priority tools and regular tools
            priority_results = []
            regular_results = []
            
            for tool_spec, score in raw_results:
                if score >= self.similarity_threshold:
                    tool_name = tool_spec.get("toolSpec", {}).get("name", "Unknown")
                    
                    # Check if tool should be excluded
                    if self._should_exclude_tool(query, tool_name):
                        logger.debug(f"Excluding tool {tool_name} for query: {query[:50]}...")
                        continue
                    
                    reasoning = None
                    if include_reasoning:
                        reasoning = self._generate_reasoning(query, tool_name, score)
                    
                    retrieved_tool = RetrievedTool(
                        tool_spec=tool_spec,
                        similarity_score=score,
                        tool_name=tool_name,
                        reasoning=reasoning
                    )
                    
                    # Check if this is a priority tool
                    if priority_tools and tool_name in priority_tools:
                        # Boost priority tool scores for better ranking
                        retrieved_tool.similarity_score += 0.1
                        priority_results.append(retrieved_tool)
                    else:
                        regular_results.append(retrieved_tool)
            
            # Sort both lists by score
            priority_results.sort(key=lambda x: x.similarity_score, reverse=True)
            regular_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Combine results: priority tools first, then regular tools
            retrieved_tools = priority_results + regular_results
            
            # Limit to top_k
            retrieved_tools = retrieved_tools[:top_k]
            
            logger.info(f"Retrieved {len(retrieved_tools)} tools for query: '{query[:50]}...'")
            
            # Log tool names for debugging
            tool_names = [rt.tool_name for rt in retrieved_tools]
            priority_tool_names = [rt.tool_name for rt in priority_results[:top_k]]
            logger.debug(f"Selected tools: {tool_names}")
            logger.debug(f"Priority tools: {priority_tool_names}")
            
            return retrieved_tools
            
        except Exception as e:
            logger.error(f"Tool retrieval failed: {e}")
            return []
    
    def get_filtered_tools(
        self, 
        query: str, 
        top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Convenience method that returns just the tool specs
        
        Args:
            query: User query string
            top_k: Number of tools to retrieve
            
        Returns:
            List of tool specifications ready for Bedrock
        """
        retrieved = self.retrieve_tools(query, top_k, include_reasoning=False)
        return [rt.tool_spec for rt in retrieved]
    
    def _generate_reasoning(self, query: str, tool_name: str, score: float) -> str:
        """Generate reasoning for why a tool was selected"""
        return (
            f"Selected '{tool_name}' (similarity: {score:.3f}) "
            f"based on semantic relevance to query"
        )
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get statistics about the tool index"""
        stats = self.tool_index.get_stats()
        stats.update({
            "similarity_threshold": self.similarity_threshold,
            "max_tools": self.max_tools,
            "embedding_model": self.embedding_model.model_name,
        })
        return stats
    
    def clear_index(self) -> None:
        """Clear all tools from the index"""
        self.tool_index = ToolIndex(self.embedding_model)
        logger.info("Cleared tool index")
    
    def __repr__(self) -> str:
        return (
            f"RAGMCPRetriever("
            f"tools={len(self.tool_index)}, "
            f"threshold={self.similarity_threshold}, "
            f"model='{self.embedding_model.model_name}'"
            f")"
        )