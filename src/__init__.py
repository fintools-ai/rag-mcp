"""
RAG-MCP Package - Retrieval-Augmented Generation for Model Context Protocol
"""

__version__ = "0.1.0"
__author__ = "RAG-MCP Team"

# Main exports
from .core.retriever import RAGMCPRetriever
from .specialized.trading_agent import TradingAgentRAGMCP
from .models.query_context import QueryContext

__all__ = [
    "RAGMCPRetriever",
    "TradingAgentRAGMCP", 
    "QueryContext",
]