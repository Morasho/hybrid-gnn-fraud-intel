"""
Edge Weight Mapping Module for GNN
Calculates meaningful weights for edges based on transaction properties
"""

import pandas as pd
import numpy as np
from typing import Tuple
import torch

def calculate_edge_weights(
    transactions_df: pd.DataFrame,
    weight_scheme: str = 'normalized_amount'
) -> np.ndarray:
    """
    Calculate edge weights for the graph based on transaction properties.
    
    Args:
        transactions_df: DataFrame with transaction data
        weight_scheme: Type of weight calculation
            - 'normalized_amount': Weight based on transaction amount (0-1)
            - 'frequency':  Weight based on edge frequency (number of transactions per edge)
            - 'combined': Combined weight from amount + frequency
            - 'inverse_amount': Inverse of amount (emphasize small transactions)
            
    Returns:
        np.ndarray: Edge weights array with same length as transactions_df
    """
    if weight_scheme == 'normalized_amount':
        # Normalize amounts to 0-1 range
        amounts = transactions_df['amount'].values
        min_amount = amounts.min()
        max_amount = amounts.max()
        
        if max_amount == min_amount:
            weights = np.ones_like(amounts, dtype=np.float32) * 0.5
        else:
            weights = (amounts - min_amount) / (max_amount - min_amount)
            weights = weights.astype(np.float32)
    
    elif weight_scheme == 'frequency':
        # Weight based on how many transactions between each sender-receiver pair
        pair_counts = transactions_df.groupby(['sender_id', 'receiver_id']).size()
        
        # Map each transaction to its pair's frequency
        weights = transactions_df.apply(
            lambda row: pair_counts.get((row['sender_id'], row['receiver_id']), 1),
            axis=1
        ).values
        
        # Normalize to 0-1
        weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
        weights = weights.astype(np.float32)
    
    elif weight_scheme == 'combined':
        # Combine amount and frequency
        amounts = transactions_df['amount'].values
        min_amount, max_amount = amounts.min(), amounts.max()
        amount_weights = (amounts - min_amount) / (max_amount - min_amount + 1e-8)
        
        # Frequency component
        pair_counts = transactions_df.groupby(['sender_id', 'receiver_id']).size()
        freq_weights = transactions_df.apply(
            lambda row: pair_counts.get((row['sender_id'], row['receiver_id']), 1),
            axis=1
        ).values
        freq_weights = (freq_weights - freq_weights.min()) / (freq_weights.max() - freq_weights.min() + 1e-8)
        
        # Combined weight (60% amount, 40% frequency)
        weights = (0.6 * amount_weights + 0.4 * freq_weights).astype(np.float32)
    
    elif weight_scheme == 'inverse_amount':
        # Lower amounts get higher weights (micro-fraud detection)
        amounts = transactions_df['amount'].values
        # Use log to compress the range
        log_amounts = np.log1p(amounts)
        min_log = log_amounts.min()
        max_log = log_amounts.max()
        
        # Inverse: high amounts get low weights
        weights = 1.0 - (log_amounts - min_log) / (max_log - min_log + 1e-8)
        weights = weights.astype(np.float32)
    
    else:
        raise ValueError(f"Unknown weight scheme: {weight_scheme}")
    
    # Ensure all weights are valid and in reasonable range
    weights = np.clip(weights, 0.01, 1.0)
    
    return weights


def calculate_fraud_risk_weights(
    transactions_df: pd.DataFrame,
    fraud_threshold: float = 0.5
) -> np.ndarray:
    """
    Calculate weights emphasizing potential fraud patterns.
    Transactions with fraud indicators get higher weights.
    
    Args:
        transactions_df: DataFrame with transaction data
        fraud_threshold: Use fraud probability/score if available
        
    Returns:
        np.ndarray: Fraud-weighted values (0-1)
    """
    weights = np.ones(len(transactions_df), dtype=np.float32) * 0.5
    
    # If we have fraud labels, use them
    if 'is_fraud' in transactions_df.columns:
        # Fraud transactions get high weight
        weights[transactions_df['is_fraud'] == 1] = 0.9
        weights[transactions_df['is_fraud'] == 0] = 0.3
    
    # Apply modifiers for suspicious patterns
    if 'night_activity_flag' in transactions_df.columns:
        weights[transactions_df['night_activity_flag'] == 1] *= 1.2
    
    if 'round_amount_flag' in transactions_df.columns:
        weights[transactions_df['round_amount_flag'] == 1] *= 0.8  # Round amounts are often legit
    
    # Normalize again
    weights = np.clip(weights, 0.01, 1.0)
    
    return weights


def create_edge_weight_tensor(
    transactions_df: pd.DataFrame,
    weight_scheme: str = 'normalized_amount'
) -> torch.Tensor:
    """
    Create a PyTorch tensor of edge weights ready for GNN.
    
    Args:
        transactions_df: DataFrame with transaction data
        weight_scheme: Edge weight calculation method
        
    Returns:
        torch.Tensor: Edge weights as 1D tensor
    """
    weights = calculate_edge_weights(transactions_df, weight_scheme)
    return torch.tensor(weights, dtype=torch.float)


def print_weight_statistics(weights: np.ndarray):
    """Print summary statistics about edge weights."""
    print(f"Edge Weight Statistics:")
    print(f"  Min: {weights.min():.4f}")
    print(f"  Max: {weights.max():.4f}")
    print(f"  Mean: {weights.mean():.4f}")
    print(f"  Std: {weights.std():.4f}")
    print(f"  Median: {np.median(weights):.4f}")
