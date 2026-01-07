
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, to_hetero

class HeteroPolarityGNN(torch.nn.Module):
    def __init__(self, num_nodes, metadata, emb_dim=64, hidden_dim=256, out_dim=3, dropout=0.2):
        """
        GNN for Vote Polarity Prediction using ONLY Learnable Embeddings (No manual features).
        
        Args:
            num_nodes (int): Total number of unique users (for Embedding layer).
            metadata (tuple): Graph metadata (node_types, edge_types).
            emb_dim (int): Dimension of learnable node embeddings.
            hidden_dim (int): Hidden dimension size for GNN layers.
            out_dim (int): Number of output classes (3 for Oppose/Neutral/Support).
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.dropout = dropout
        
        # 1. Learnable Node Embeddings (The only source of node features)
        self.node_emb = nn.Embedding(num_nodes, emb_dim)
        
        # 2. GNN Backbone (Encoder)
        class GraphSAGE_Encoder(torch.nn.Module):
            def __init__(self, in_dim, hidden_dim, dropout):
                super().__init__()
                # Layer 1: emb_dim -> hidden_dim
                self.conv1 = SAGEConv((-1, -1), hidden_dim)
                # Layer 2: hidden_dim -> hidden_dim
                self.conv2 = SAGEConv((-1, -1), hidden_dim)
                self.dropout = dropout

            def forward(self, x, edge_index):
                x = self.conv1(x, edge_index)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x = self.conv2(x, edge_index)
                return x

        # Transform homogeneous GNN to heterogeneous
        # We pass 'emb_dim' as input dimension to the encoder
        self.gnn = to_hetero(GraphSAGE_Encoder(emb_dim, hidden_dim, dropout), metadata, aggr='mean')
        
        # 3. Edge Classifier
        # Takes concatenated embeddings of source and target nodes (2 * hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, batch):
        # 1. Retrieve Original Node IDs (mapped by LinkNeighborLoader)
        n_id = batch['user'].n_id
        
        # 2. Lookup Embeddings (This is our initial 'x')
        x = self.node_emb(n_id) # [batch_nodes, emb_dim]
        
        # Note: We completely IGNORE batch['user'].x (manual features) here.
        
        # 3. Run GNN to get updated Node Embeddings (Message Passing)
        x_dict = {'user': x}
        x_out_dict = self.gnn(x_dict, batch.edge_index_dict)
        x_user = x_out_dict['user'] # [num_nodes_in_batch, hidden_dim]
        
        # 4. Decode Edges for Classification
        # Get source and target indices for the edges in the current batch
        edge_label_index = batch['user', 'votes', 'user'].edge_label_index
        src_idx = edge_label_index[0]
        tgt_idx = edge_label_index[1]
        
        x_src = x_user[src_idx]
        x_tgt = x_user[tgt_idx]
        
        # Concatenate source and target embeddings
        edge_feat = torch.cat([x_src, x_tgt], dim=-1)
        
        # Predict Class Logits
        out = self.classifier(edge_feat)
        return out
