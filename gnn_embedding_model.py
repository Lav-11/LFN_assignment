
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

# Define the GraphSAGE model with a GNN encoder and an edge-level head.
class HeteroPolarityGNN(nn.Module):
    def __init__(self, metadata, num_users, emb_dim=64, hidden_dim=128, dropout=0.3):
        super().__init__()

        self.node_emb = nn.Embedding(num_users, emb_dim)
        
        # Base Homogeneous GNN
        class BaseGNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(p=dropout) 
                self.conv1 = SAGEConv(emb_dim, hidden_dim)
                self.conv2 = SAGEConv(hidden_dim, hidden_dim)
            def forward(self, x, edge_index):
                x = self.dropout(x)
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = self.dropout(x)
                x = self.conv2(x, edge_index)
                x = F.relu(x)
                return x
        
        # Convert to Hetero GNN
        self.hetero_gnn = to_hetero(BaseGNN(), metadata, aggr='sum')
        
        edge_in = hidden_dim * 4
        
        # Head for polarity
        self.pol_mlp = nn.Sequential(
            nn.Linear(edge_in, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 3)
        )

    def forward(self, batch):
        # Dict for node embeddings
        n_id = batch['user'].n_id
        x = self.node_emb(n_id) # [batch_nodes, emb_dim]

        x_dict = {'user': x}
        
        # Learn GNN embeddings
        z_dict = self.hetero_gnn(x_dict, batch.edge_index_dict)
        
        src, tgt = batch[('user', 'votes', 'user')].edge_label_index
        
        # Get user embeddings
        z = z_dict['user']
        h_src = z[src]
        h_tgt = z[tgt]
        
        # Edge features
        e = torch.cat([h_src, h_tgt, torch.abs(h_src - h_tgt), h_src * h_tgt], dim=1)
        
        # Predict
        return self.pol_mlp(e)