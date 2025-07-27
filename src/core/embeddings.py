"""
Embedding model wrapper for converting text to vectors
"""

import numpy as np
import logging
from typing import Union, List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Wrapper for sentence transformer embedding models
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence transformer model
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        
        try:
            self.model = SentenceTransformer(model_name)
            logger.info(f"Successfully loaded model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def encode(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Encode text into embeddings
        
        Args:
            text: Single string or list of strings
            
        Returns:
            Embedding vector(s) as numpy array
        """
        try:
            embeddings = self.model.encode(
                text, 
                convert_to_numpy=True, 
                show_progress_bar=False,
                normalize_embeddings=True
            )
            
            # Ensure we always return 2D array for consistency
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
                
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to encode text: {e}")
            raise
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Convenience method to encode single text and return 1D array
        
        Args:
            text: Text to encode
            
        Returns:
            1D embedding vector
        """
        result = self.encode(text)
        return result[0] if result.ndim == 2 else result
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def __repr__(self) -> str:
        return f"EmbeddingModel(model='{self.model_name}', dim={self.dimension})"