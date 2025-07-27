"""
In-memory storage backend for RAG-MCP (for testing/development)
"""

import logging
from typing import Dict, Any, Optional

from src.storage.base_store import BaseStore

logger = logging.getLogger(__name__)


class MemoryStore(BaseStore):
    """
    In-memory storage backend for RAG-MCP (non-persistent)
    """
    
    def __init__(self):
        """Initialize memory store"""
        self.data: Dict[str, Dict[str, Any]] = {}
        logger.info("Memory store initialized")
    
    def save_agent_data(self, key: str, data: Dict[str, Any]) -> bool:
        """Save agent data to memory"""
        try:
            # Deep copy to avoid reference issues
            import copy
            self.data[key] = copy.deepcopy(data)
            
            logger.debug(f"Saved data to memory key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to memory: {e}")
            return False
    
    def load_agent_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Load agent data from memory"""
        try:
            if key not in self.data:
                logger.debug(f"No data found for memory key: {key}")
                return None
            
            # Deep copy to avoid reference issues
            import copy
            result = copy.deepcopy(self.data[key])
            
            logger.debug(f"Loaded data from memory key: {key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load from memory: {e}")
            return None
    
    def delete_agent_data(self, key: str) -> bool:
        """Delete agent data from memory"""
        try:
            if key in self.data:
                del self.data[key]
                logger.debug(f"Deleted memory key: {key}")
                return True
            else:
                logger.debug(f"Key not found for deletion: {key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete from memory: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in memory"""
        return key in self.data
    
    def clear_all(self) -> bool:
        """Clear all data from memory"""
        try:
            cleared_count = len(self.data)
            self.data.clear()
            
            logger.info(f"Cleared {cleared_count} keys from memory")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear memory: {e}")
            return False
    
    def get_all_keys(self) -> list:
        """Get all stored keys"""
        return list(self.data.keys())
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            import sys
            
            # Estimate memory usage
            total_size = sys.getsizeof(self.data)
            for key, value in self.data.items():
                total_size += sys.getsizeof(key) + sys.getsizeof(value)
            
            return {
                "backend": "memory",
                "total_keys": len(self.data),
                "estimated_size_bytes": total_size,
                "keys": list(self.data.keys())
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {"backend": "memory", "error": str(e)}