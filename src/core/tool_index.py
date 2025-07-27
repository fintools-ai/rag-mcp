"""
Tool index for storing and searching tool vectors
"""

import numpy as np
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity

from src.core.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class ToolIndex:
    """
    Manages the vector index of MCP tools for efficient similarity search
    """
    
    def __init__(self, embedding_model: EmbeddingModel):
        """
        Initialize tool index
        
        Args:
            embedding_model: Embedding model for encoding tool descriptions
        """
        self.embedding_model = embedding_model
        self.tools: List[Dict[str, Any]] = []
        self.embeddings: List[np.ndarray] = []
        self.tool_texts: List[str] = []
        
        logger.info("Initialized ToolIndex")
    
    def add_tool(self, tool_spec: Dict[str, Any]) -> None:
        """
        Add a tool to the index
        
        Args:
            tool_spec: Tool specification in MCP format
        """
        try:
            # Extract searchable text from tool
            tool_text = self._extract_tool_text(tool_spec)
            
            # Generate embedding
            embedding = self.embedding_model.encode_single(tool_text)
            
            # Store everything
            self.tools.append(tool_spec)
            self.embeddings.append(embedding)
            self.tool_texts.append(tool_text)
            
            tool_name = tool_spec.get("toolSpec", {}).get("name", "Unknown")
            logger.debug(f"Added tool to index: {tool_name}")
            
        except Exception as e:
            logger.error(f"Failed to add tool to index: {e}")
            raise
    
    def add_tools(self, tool_specs: List[Dict[str, Any]]) -> None:
        """
        Add multiple tools to the index
        
        Args:
            tool_specs: List of tool specifications
        """
        for tool_spec in tool_specs:
            self.add_tool(tool_spec)
        
        logger.info(f"Added {len(tool_specs)} tools to index")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Dict[str, Any], float]]:
        """
        Search for similar tools
        
        Args:
            query_embedding: Query vector to search for
            top_k: Number of top results to return
            
        Returns:
            List of tuples (tool_spec, similarity_score) sorted by relevance
        """
        if not self.embeddings:
            logger.warning("No tools in index")
            return []
        
        try:
            # Convert embeddings to matrix
            embeddings_matrix = np.array(self.embeddings)
            
            # Ensure query is 2D for sklearn
            query_2d = query_embedding.reshape(1, -1)
            
            # Calculate cosine similarities
            similarities = cosine_similarity(query_2d, embeddings_matrix)[0]
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            # Build results
            results = []
            for idx in top_indices:
                results.append((self.tools[idx], float(similarities[idx])))
            
            logger.debug(f"Found {len(results)} similar tools")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _extract_tool_text(self, tool_spec: Dict[str, Any]) -> str:
        """
        Extract searchable text from tool specification
        
        Args:
            tool_spec: Tool specification
            
        Returns:
            Combined text for embedding
        """
        tool_info = tool_spec.get("toolSpec", {})
        
        parts = []
        
        # Tool name and description
        name = tool_info.get("name", "")
        description = tool_info.get("description", "")
        
        if name:
            parts.append(f"Tool: {name}")
        if description:
            parts.append(f"Description: {description}")
        
        # Parameter descriptions
        input_schema = tool_info.get("inputSchema", {}).get("json", {})
        properties = input_schema.get("properties", {})
        
        for param_name, param_info in properties.items():
            param_desc = param_info.get("description", "")
            if param_desc:
                parts.append(f"Parameter {param_name}: {param_desc}")
        
        # Combine all parts
        combined_text = " ".join(parts)
        
        if not combined_text.strip():
            logger.warning(f"No extractable text for tool: {name}")
            combined_text = name or "unknown_tool"
        
        return combined_text
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_tools": len(self.tools),
            "embedding_dimension": self.embedding_model.dimension if self.embeddings else 0,
            "index_size_mb": self._estimate_size_mb(),
        }
    
    def _estimate_size_mb(self) -> float:
        """Estimate index size in MB"""
        if not self.embeddings:
            return 0.0
        
        # Size of embeddings (float32 = 4 bytes)
        embeddings_size = len(self.embeddings) * self.embedding_model.dimension * 4
        
        # Size of tool specs (rough estimate)
        tools_size = len(json.dumps(self.tools).encode('utf-8'))
        
        total_bytes = embeddings_size + tools_size
        return total_bytes / (1024 * 1024)
    
    def __len__(self) -> int:
        """Return number of tools in index"""
        return len(self.tools)
    
    def __repr__(self) -> str:
        return f"ToolIndex(tools={len(self.tools)}, dim={self.embedding_model.dimension})"