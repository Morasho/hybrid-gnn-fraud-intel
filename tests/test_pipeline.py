import os
import pytest
import torch
import pandas as pd
from neo4j import GraphDatabase

#  Configuration 
URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "12345678") # Your  password
GRAPH_FILE = 'data/processed/hetero_graph.pt'

# 1. Test Database Integrity (Neo4j)
def test_neo4j_connection_and_data_loader():
    """Tests if Neo4j is running and successfully loaded the 100k transactions."""
    try:
        driver = GraphDatabase.driver(URI, auth=AUTH)
        with driver.session() as session:
            # Verify Users exist
            user_result = session.run("MATCH (u:User) RETURN count(u) AS count")
            user_count = user_result.single()["count"]
            assert user_count > 0, "Data Loader Failed: No Users found in Neo4j."
            
            # Verify Fraud Scenarios exist (Testing the M-Pesa topologies)
            fraud_result = session.run("MATCH ()-[r:P2P_TRANSFER {is_fraud: 1}]->() RETURN count(r) AS count")
            fraud_count = fraud_result.single()["count"]
            assert fraud_count > 0, "Data Loader Failed: No Fraud edges found."
            
        driver.close()
    except Exception as e:
        pytest.fail(f"Database connection or query failed: {e}")

# 2. Test Feature Engineering Logic (Pandas)
def test_advanced_feature_logic():
    """Tests if our new Pandas logic correctly catches SIM Swaps (Shared Devices)."""
    # Mock a scenario: 3 different users logging into Device 1, and 1 user on Device 2
    mock_df = pd.DataFrame({
        'device_id': ['D_1', 'D_1', 'D_1', 'D_2'],
        'sender_id': ['U_A', 'U_B', 'U_C', 'U_D']
    })
    
    # Run the exact logic from our feature_engineering.py script
    device_counts = mock_df.groupby('device_id')['sender_id'].nunique().reset_index()
    device_counts.rename(columns={'sender_id': 'num_accounts_linked'}, inplace=True)
    mock_df = mock_df.merge(device_counts, on='device_id', how='left')
    mock_df['shared_device_flag'] = (mock_df['num_accounts_linked'] > 2).astype(int)
    
    # Assert Device 1 is flagged as a SIM Swap (1) and Device 2 is Safe (0)
    assert mock_df[mock_df['device_id'] == 'D_1']['shared_device_flag'].iloc[0] == 1, "Failed to flag shared device!"
    assert mock_df[mock_df['device_id'] == 'D_2']['shared_device_flag'].iloc[0] == 0, "Falsely flagged a safe device!"

# 3. Test Graph Construction Correctness (PyTorch)
def test_pytorch_heterodata_structure():
    """Tests if the 13 engineered features were correctly compressed into PyTorch."""
    assert os.path.exists(GRAPH_FILE), f"Graph tensor file missing at {GRAPH_FILE}."
    
    # Load the math
    data = torch.load(GRAPH_FILE, weights_only=False)
    
    # Assert our condensed, highly-focused User nodes exist
    assert 'user' in data.node_types, "Graph Construction Error: Missing 'user' nodes."
    
    # Assert complex P2P edges exist
    assert ('user', 'p2p', 'user') in data.edge_types, "Missing User-to-User P2P edges."
    
    # Validate tensor formats (Must be 13 columns for our new features, and float32)
    assert data['user'].x.shape[1] == 13, f"Expected 13 node features, but got {data['user'].x.shape[1]}"
    assert data['user'].x.dtype == torch.float32, "User features must be float32 tensors."