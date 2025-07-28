"""
Redis storage backend for RAG-MCP
"""

import json
import logging
from typing import Dict, Any, Optional

from ..storage.base_store import BaseStore

logger = logging.getLogger(__name__)


class RedisStore(BaseStore):
    """
    Redis-based storage backend for RAG-MCP
    """
    
    def __init__(self, redis_client, key_prefix: str = "rag_mcp"):
        """
        Initialize Redis store
        
        Args:
            redis_client: Redis client instance
            key_prefix: Prefix for all keys
        """
        self.redis = redis_client
        self.key_prefix = key_prefix
        
        # Test connection
        try:
            self.redis.ping()
            logger.info("Redis store initialized successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix"""
        return f"{self.key_prefix}:{key}"
    
    def save_agent_data(self, key: str, data: Dict[str, Any]) -> bool:
        """Save agent data to Redis"""
        try:
            redis_key = self._make_key(key)
            serialized_data = json.dumps(data)
            
            self.redis.set(redis_key, serialized_data)
            logger.debug(f"Saved data to Redis key: {redis_key}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to Redis: {e}")
            return False
    
    def load_agent_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Load agent data from Redis"""
        try:
            redis_key = self._make_key(key)
            data = self.redis.get(redis_key)
            
            if data is None:
                logger.debug(f"No data found for key: {redis_key}")
                return None
            
            # Deserialize JSON data
            if isinstance(data, bytes):
                data = data.decode('utf-8')
            
            result = json.loads(data)
            logger.debug(f"Loaded data from Redis key: {redis_key}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to load from Redis: {e}")
            return None
    
    def delete_agent_data(self, key: str) -> bool:
        """Delete agent data from Redis"""
        try:
            redis_key = self._make_key(key)
            result = self.redis.delete(redis_key)
            
            if result > 0:
                logger.debug(f"Deleted Redis key: {redis_key}")
                return True
            else:
                logger.debug(f"Key not found for deletion: {redis_key}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete from Redis: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in Redis"""
        try:
            redis_key = self._make_key(key)
            return bool(self.redis.exists(redis_key))
            
        except Exception as e:
            logger.error(f"Failed to check existence in Redis: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all RAG-MCP data from Redis"""
        try:
            # Find all keys with our prefix
            pattern = f"{self.key_prefix}:*"
            keys = self.redis.keys(pattern)
            
            if keys:
                deleted_count = self.redis.delete(*keys)
                logger.info(f"Cleared {deleted_count} RAG-MCP keys from Redis")
            else:
                logger.info("No RAG-MCP keys found to clear")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear Redis data: {e}")
            return False
    
    def get_all_keys(self) -> list:
        """Get all RAG-MCP keys"""
        try:
            pattern = f"{self.key_prefix}:*"
            keys = self.redis.keys(pattern)
            
            # Remove prefix from key names
            clean_keys = [
                key.decode('utf-8').replace(f"{self.key_prefix}:", "") 
                if isinstance(key, bytes) else key.replace(f"{self.key_prefix}:", "")
                for key in keys
            ]
            
            return clean_keys
            
        except Exception as e:
            logger.error(f"Failed to get keys from Redis: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            keys = self.get_all_keys()
            
            # Calculate total size estimate
            total_size = 0
            for key in keys:
                redis_key = self._make_key(key)
                try:
                    size = self.redis.strlen(redis_key)
                    total_size += size
                except:
                    pass
            
            return {
                "backend": "redis",
                "total_keys": len(keys),
                "estimated_size_bytes": total_size,
                "key_prefix": self.key_prefix,
                "keys": keys
            }
            
        except Exception as e:
            logger.error(f"Failed to get Redis stats: {e}")
            return {"backend": "redis", "error": str(e)}