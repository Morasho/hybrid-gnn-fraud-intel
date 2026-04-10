"""
Configuration utility for GNN models
Auto-detects embedding dimensions and other configuration parameters
"""

import pandas as pd
import os
from typing import Tuple

def get_embedding_dimensions(embeddings_file: str = 'data/processed/user_embeddings.csv') -> int:
    """
    Auto-detects the embedding dimensions from the embeddings CSV file.
    
    Args:
        embeddings_file: Path to the embeddings CSV file
        
    Returns:
        Number of embedding dimensions (columns - 1 for user_id column)
        
    Raises:
        FileNotFoundError: If embeddings file doesn't exist
        ValueError: If embeddings file is empty or malformed
    """
    if not os.path.exists(embeddings_file):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
    
    try:
        embeddings_df = pd.read_csv(embeddings_file, nrows=1)
        
        if embeddings_df.empty:
            raise ValueError("Embeddings file is empty")
        
        # Assuming first column is user_id, subtract 1
        embedding_dim = len(embeddings_df.columns) - 1
        
        if embedding_dim <= 0:
            raise ValueError("No embedding dimensions detected (file has only 1 column)")
        
        return embedding_dim
    
    except Exception as e:
        raise ValueError(f"Error reading embeddings file: {str(e)}")


def load_embeddings_with_prefix(
    embeddings_file: str = 'data/processed/user_embeddings.csv',
    prefix: str = 'gnn_'
) -> Tuple[pd.DataFrame, int]:
    """
    Loads embeddings CSV and adds a prefix to column names.
    
    Args:
        embeddings_file: Path to the embeddings CSV file
        prefix: Prefix to add to embedding columns (default: 'gnn_')
        
    Returns:
        Tuple of (prefixed_embeddings_df, embedding_dimensions)
    """
    embeddings_df = pd.read_csv(embeddings_file)
    embedding_dim = len(embeddings_df.columns) - 1
    
    # Keep original user_id column name in mind, it will become '{prefix}user_id'
    embeddings_df = embeddings_df.add_prefix(prefix)
    
    return embeddings_df, embedding_dim


def get_model_config() -> dict:
    """
    Returns the configuration for GNN models based on detected embeddings.
    
    Returns:
        Dictionary with model configuration including auto-detected dimensions
    """
    try:
        embedding_dim = get_embedding_dimensions()
        
        return {
            'embedding_dim': embedding_dim,
            'hidden_channels': embedding_dim,  # Use detected dimension as hidden channels
            'edge_classifier_hidden': embedding_dim,
            'edge_classifier_output': 1,
            'learning_rate': 0.01,
            'random_state': 42,
            'num_epochs': 100
        }
    except FileNotFoundError:
        # If embeddings don't exist yet, use default but don't warn
        return {
            'embedding_dim': 64,  # Default fallback
            'hidden_channels': 64,
            'edge_classifier_hidden': 64,
            'edge_classifier_output': 1,
            'learning_rate': 0.01,
            'random_state': 42,
            'num_epochs': 100
        }
    except Exception as e:
        print(f"Warning: Could not auto-detect config: {str(e)}")
        return {
            'embedding_dim': 64,  # Default fallback
            'hidden_channels': 64,
            'edge_classifier_hidden': 64,
            'edge_classifier_output': 1,
            'learning_rate': 0.01,
            'random_state': 42,
            'num_epochs': 100
        }


if __name__ == "__main__":
    # Test the utility functions
    try:
        dim = get_embedding_dimensions()
        print(f"✓ Auto-detected embedding dimensions: {dim}")
        
        config = get_model_config()
        print(f"✓ Model config: {config}")
    except Exception as e:
        print(f"✗ Error: {str(e)}")
