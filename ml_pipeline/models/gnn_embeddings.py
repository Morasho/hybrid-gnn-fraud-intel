import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero
import torch_geometric.transforms as T
import pandas as pd
import os
from config import get_model_config

print(" Phase 2, Week 4: Graph Neural Network Training ")

# Get auto-detected configuration
config = get_model_config()
embedding_dim = config['embedding_dim']
print(f"Using auto-detected embedding dimension: {embedding_dim}")


#  1. Load the Upgraded Tensors 
GRAPH_PATH = 'data/processed/hetero_graph.pt'
data = torch.load(GRAPH_PATH, weights_only=False)

# Make the graph undirected so information flows both ways!
data = T.ToUndirected()(data)

#  2. Define the Neural Network Architecture 
class GNNEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

class EdgeClassifier(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(2 * hidden_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, z_dict, edge_index, sender_type, receiver_type):
        row, col = edge_index
        sender_z = z_dict[sender_type][row]
        receiver_z = z_dict[receiver_type][col]
        
        # Concatenate sender and receiver to evaluate the relationship
        z = torch.cat([sender_z, receiver_z], dim=-1)
        z = self.lin1(z).relu()
        z = self.lin2(z)
        return z.view(-1)

class HybridGNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.encoder = GNNEncoder(hidden_channels, hidden_channels)
        self.encoder = to_hetero(self.encoder, data.metadata(), aggr='mean')
        self.classifier = EdgeClassifier(hidden_channels)

    def forward(self, x_dict, edge_index_dict, target_edge_index, sender_type, receiver_type):
        z_dict = self.encoder(x_dict, edge_index_dict)
        return self.classifier(z_dict, target_edge_index, sender_type, receiver_type)

# 3. Training Setup 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = data.to(device)

# Initialize model with auto-detected embedding dimensions
print(f"Initializing GNN model with {embedding_dim}-dimensional embeddings...")
model = HybridGNN(hidden_channels=embedding_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Handle the 2.8% Fraud Imbalance (CRUCIAL for real-world detection)
num_neg = data['user', 'p2p', 'user'].y.size(0) - data['user', 'p2p', 'user'].y.sum()
pos_weight = num_neg / data['user', 'p2p', 'user'].y.sum()
criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

train_edges = data['user', 'p2p', 'user'].edge_index
train_labels = data['user', 'p2p', 'user'].y.float()

#  4. Training Loop 
print("Training GNN on 100,000 transactions to learn network shapes...")
model.train()
for epoch in range(1, 101):
    optimizer.zero_grad()
    
    out = model(data.x_dict, data.edge_index_dict, train_edges, 'user', 'user')
    loss = criterion(out, train_labels)
    loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.4f}')

# 5. Extract and Save Embeddings
print(f"\nExtracting {embedding_dim}-dimensional structural embeddings...")
model.eval()
with torch.no_grad():
    final_embeddings = model.encoder(data.x_dict, data.edge_index_dict)
    user_embeddings = final_embeddings['user'].cpu().numpy()

# Map the math back to the actual User IDs
df = pd.read_csv('data/processed/final_model_data.csv')
all_users = pd.concat([df['sender_id'], df['receiver_id']]).unique()

os.makedirs('data/processed', exist_ok=True)
embed_df = pd.DataFrame(user_embeddings)
embed_df.insert(0, 'user_id', all_users)
embed_df.to_csv('data/processed/user_embeddings.csv', index=False)

print("Saved User embeddings to data/processed/user_embeddings.csv!")
print("Week 4 Complete: The graph structure is now a tabular feature!")