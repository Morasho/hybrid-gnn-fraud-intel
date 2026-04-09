import torch

# Load the ultimate graph
data = torch.load('data/processed/hetero_graph.pt', weights_only=False)

# 1. Check User Features (The 13 Engineered Columns)
user_features = data['user'].x
print(f"User Matrix Shape: {user_features.shape}")
print(f"Any NaNs in Users?: {torch.isnan(user_features).any().item()}")

# 2. Check Edges (The Connections)
p2p_edges = data['user', 'p2p', 'user'].edge_index
print(f"Total P2P Connections: {p2p_edges.shape[1]}")

# 3. Check Labels (The Fraud Target)
labels = data['user', 'p2p', 'user'].y
print(f"Total Fraud Labels: {labels.sum().item()}")
print(f"Any NaNs in Labels?: {torch.isnan(labels.float()).any().item()}")