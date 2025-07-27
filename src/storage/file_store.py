"""
File-based storage backend for RAG-MCP
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from src.storage.base_store import BaseStore

logger = logging.getLogger(__name__)


class FileStore(BaseStore):
    """
    File-based storage backend for RAG-MCP
    """
    
    def __init__(self, storage_dir: str = "~/.rag_mcp"):
        """
        Initialize file store
        
        Args:
            storage_dir: Directory to store files
        """
        self.storage_dir = Path(storage_dir).expanduser()
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"File store initialized at: {self.storage_dir}")
    
    def _make_path(self, key: str) -> Path:
        """Create file path for key"""
        # Replace invalid filename characters
        safe_key = key.replace("/", "_").replace("\\", "_").replace(":", "_")
        return self.storage_dir / f"{safe_key}.json"
    
    def save_agent_data(self, key: str, data: Dict[str, Any]) -> bool:
        """Save agent data to file"""
        try:
            file_path = self._make_path(key)
            
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.debug(f"Saved data to file: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save to file: {e}")
            return False
    
    def load_agent_data(self, key: str) -> Optional[Dict[str, Any]]:
        """Load agent data from file"""
        try:
            file_path = self._make_path(key)
            
            if not file_path.exists():
                logger.debug(f"File not found: {file_path}")
                return None
            
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            logger.debug(f"Loaded data from file: {file_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load from file: {e}")
            return None
    
    def delete_agent_data(self, key: str) -> bool:
        """Delete agent data file"""
        try:
            file_path = self._make_path(key)
            
            if file_path.exists():
                file_path.unlink()
                logger.debug(f"Deleted file: {file_path}")
                return True
            else:
                logger.debug(f"File not found for deletion: {file_path}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to delete file: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if file exists"""
        try:
            file_path = self._make_path(key)
            return file_path.exists()
            
        except Exception as e:
            logger.error(f"Failed to check file existence: {e}")
            return False
    
    def clear_all(self) -> bool:
        """Clear all RAG-MCP files"""
        try:
            deleted_count = 0
            
            # Delete all .json files in storage directory
            for file_path in self.storage_dir.glob("*.json"):
                file_path.unlink()
                deleted_count += 1
            
            logger.info(f"Cleared {deleted_count} files from storage")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear files: {e}")
            return False
    
    def get_all_keys(self) -> list:
        """Get all stored keys"""
        try:
            keys = []
            
            for file_path in self.storage_dir.glob("*.json"):
                # Remove .json extension to get key
                key = file_path.stem
                keys.append(key)
            
            return keys
            
        except Exception as e:
            logger.error(f"Failed to get file keys: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics"""
        try:
            keys = self.get_all_keys()
            
            # Calculate total size
            total_size = 0
            for key in keys:
                file_path = self._make_path(key)
                if file_path.exists():
                    total_size += file_path.stat().st_size
            
            return {
                "backend": "file",
                "storage_dir": str(self.storage_dir),
                "total_keys": len(keys),
                "total_size_bytes": total_size,
                "keys": keys
            }
            
        except Exception as e:
            logger.error(f"Failed to get file stats: {e}")
            return {"backend": "file", "error": str(e)}