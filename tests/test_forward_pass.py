"""
Forward Pass Test Script
========================
Test if the HybridGNN model can successfully process data from left to right
(Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output) without crashes.

Usage:
    python test_forward_pass.py
"""

import torch
import sys
import traceback

# =====================================================================
# 1. LOAD THE GRAPH DATA
# =====================================================================
print("=" * 70)
print("STEP 1: Loading Hetero Graph Data")
print("=" * 70)

try:
    GRAPH_PATH = 'data/processed/hetero_graph.pt'
    data = torch.load(GRAPH_PATH, weights_only=False)
    print(f"✓ Successfully loaded graph from {GRAPH_PATH}")
except FileNotFoundError:
    print(f"✗ ERROR: Could not find {GRAPH_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"✗ ERROR loading graph: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 2. CHECK DATA STRUCTURE
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: Inspecting Data Structure")
print("=" * 70)

try:
    print(f"  Data keys: {data.keys()}")
    print(f"  Metadata: {data.metadata()}")
    print(f"  User nodes: {data['user'].num_nodes}")
    print(f"  User features shape: {data['user'].x.shape}")
    print(f"  P2P edges: {data['user', 'p2p', 'user'].num_edges}")
    print(f"  Edge index shape: {data['user', 'p2p', 'user'].edge_index.shape}")
    print(f"  Labels shape: {data['user', 'p2p', 'user'].y.shape}")
    print("✓ Data structure verified")
except Exception as e:
    print(f"✗ ERROR inspecting data: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 3. DEFINE THE MODEL (copied from gnn_embeddings.py)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: Defining HybridGNN Model Architecture")
print("=" * 70)

try:
    from torch_geometric.nn import SAGEConv, to_hetero
    import torch.nn as nn

    class GNNEncoder(nn.Module):
        def __init__(self, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv((-1, -1), hidden_channels)
            self.conv2 = SAGEConv((-1, -1), out_channels)

        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index).relu()
            x = self.conv2(x, edge_index)
            return x

    class EdgeClassifier(nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.lin1 = nn.Linear(2 * hidden_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, 1)

        def forward(self, z_dict, edge_index, sender_type, receiver_type):
            row, col = edge_index
            sender_z = z_dict[sender_type][row]
            receiver_z = z_dict[receiver_type][col]
            
            # Concatenate sender and receiver to evaluate the relationship
            z = torch.cat([sender_z, receiver_z], dim=-1)
            z = self.lin1(z).relu()
            z = self.lin2(z)
            return z.view(-1)

    class HybridGNN(nn.Module):
        def __init__(self, hidden_channels):
            super().__init__()
            self.encoder = GNNEncoder(hidden_channels, hidden_channels)
            self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
            self.classifier = EdgeClassifier(hidden_channels)

        def forward(self, x_dict, edge_index_dict, target_edge_index, sender_type, receiver_type):
            z_dict = self.encoder(x_dict, edge_index_dict)
            return self.classifier(z_dict, target_edge_index, sender_type, receiver_type)

    print("✓ Model classes defined successfully")
except Exception as e:
    print(f"✗ ERROR defining model: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 4. INITIALIZE THE MODEL
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: Initializing HybridGNN Model")
print("=" * 70)

try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Using device: {device}")
    
    data = data.to(device)
    model = HybridGNN(hidden_channels=64).to(device)
    model.eval()  # Set to evaluation mode (no training)
    print(f"✓ Model initialized successfully with 64-dim embeddings")
except Exception as e:
    print(f"✗ ERROR initializing model: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 5. PREPARE TEST DATA
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: Preparing Test Batch (subset of edges)")
print("=" * 70)

try:
    # Use first 100 edges for quick testing
    num_test_edges = min(100, data['user', 'p2p', 'user'].edge_index.shape[1])
    test_edge_index = data['user', 'p2p', 'user'].edge_index[:, :num_test_edges]
    
    print(f"  Test edges: {num_test_edges} out of {data['user', 'p2p', 'user'].num_edges} total")
    print(f"  Test edge index shape: {test_edge_index.shape}")
except Exception as e:
    print(f"✗ ERROR preparing test data: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 6. RUN THE FORWARD PASS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: FORWARD PASS TEST")
print("=" * 70)

try:
    with torch.no_grad():
        output = model(
            data.x_dict,
            data.edge_index_dict,
            test_edge_index,
            'user',
            'user'
        )
    
    print("✓ FORWARD PASS SUCCESSFUL!")
    print(f"  Output shape: {output.shape}")
    print(f"  Output dtype: {output.dtype}")
    print(f"  Output device: {output.device}")
    
except RuntimeError as e:
    if "CUDA out of memory" in str(e):
        print(f"✗ CUDA OUT OF MEMORY ERROR: {e}")
        print("  Solution: Reduce num_test_edges or use CPU")
    elif "shape mismatch" in str(e).lower():
        print(f"✗ SHAPE MISMATCH ERROR: {e}")
        print("  Solution: Check data dimensions and model input expectations")
    else:
        print(f"✗ RUNTIME ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)
except Exception as e:
    print(f"✗ UNEXPECTED ERROR: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# 7. VALIDATE OUTPUT
# =====================================================================
print("\n" + "=" * 70)
print("STEP 7: Validating Output Tensor")
print("=" * 70)

try:
    # Check if output is valid numbers (not NaN or Inf)
    has_nan = torch.isnan(output).any().item()
    has_inf = torch.isinf(output).any().item()
    
    if has_nan:
        print("⚠ WARNING: Output contains NaN values")
    elif has_inf:
        print("⚠ WARNING: Output contains Inf values")
    else:
        print("✓ Output contains only valid numbers")
    
    # Print sample outputs
    print(f"\n  First 10 output values:")
    print(f"  {output[:10]}")
    print(f"\n  Output statistics:")
    print(f"    Min: {output.min().item():.6f}")
    print(f"    Max: {output.max().item():.6f}")
    print(f"    Mean: {output.mean().item():.6f}")
    print(f"    Std: {output.std().item():.6f}")
    
except Exception as e:
    print(f"✗ ERROR validating output: {e}")
    traceback.print_exc()
    sys.exit(1)

# =====================================================================
# FINAL RESULT
# =====================================================================
print("\n" + "=" * 70)
print("✓ ALL TESTS PASSED - FORWARD PASS IS WORKING!")
print("=" * 70)
print("\nYour model is ready for training. Next steps:")
print("  1. Set up training loop with full edge batches")
print("  2. Add loss function (BCEWithLogitsLoss for imbalance)")
print("  3. Implement validation and early stopping")
print("=" * 70)
