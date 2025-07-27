"""
Base storage interface for RAG-MCP
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseStore(ABC):
    """
    Abstract base class for RAG-MCP storage backends
    """
    
    @abstractmethod
    def save_agent_data(self, key: str, data: Dict[str, Any]) -> bool:
        """
        Save agent data
        
        Args:
            key: Storage key
            data: Data to save
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def load_agent_data(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Load agent data
        
        Args:
            key: Storage key
            
        Returns:
            Loaded data or None if not found
        """
        pass
    
    @abstractmethod
    def delete_agent_data(self, key: str) -> bool:
        """
        Delete agent data
        
        Args:
            key: Storage key
            
        Returns:
            True if successful
        """
        pass
    
    @abstractmethod
    def exists(self, key: str) -> bool:
        """
        Check if key exists
        
        Args:
            key: Storage key
            
        Returns:
            True if key exists
        """
        pass
    
    @abstractmethod
    def clear_all(self) -> bool:
        """
        Clear all stored data
        
        Returns:
            True if successful
        """
        pass