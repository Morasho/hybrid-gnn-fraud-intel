import os
import pandas as pd
import pytest

print("--- Week 4 Validation: Structural Embedding Tests ---")

def test_embedding_dimensions():
    """Tests if the GNN successfully converted topological shapes into 64-D tabular features."""
    embed_file = 'data/processed/user_embeddings.csv'
    
    # 1. Check if the file exists
    assert os.path.exists(embed_file), "Embeddings file missing! Training failed."
    
    df = pd.read_csv(embed_file)
    
    # 2. Verify all 10,000 users made it out of the neural network
    assert len(df) == 10000, f"Data loss detected! Expected 10,000 users, got {len(df)}"
    
    # 3. Verify the architecture shape (1 ID column + 64 dimension columns = 65)
    assert df.shape[1] == 65, f"Architecture Error: Expected 65 columns, got {df.shape[1]}"

def test_embedding_numerical_stability():
    """Tests if the neural network output valid math (no NaNs or infinite loops)."""
    df = pd.read_csv('data/processed/user_embeddings.csv')
    
    # Drop the ID column so we are only looking at pure PyTorch math
    math_only = df.drop(columns=['user_id'])
    
    # Check for NaNs
    assert not math_only.isnull().values.any(), "Mathematical Failure: NaNs detected in embeddings."