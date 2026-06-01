
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(nn.Module):
    """
    Graph Neural Network of molecular property prediction.

    Architecture:
        Input: atom feature matrix (node features) + bond conntectivity
        Hidden: 4 rounds of message passing between atoms
        Output: single scalar (predicted HOLO-LUMO gap)
    How message passing works:
        Each atom collects feature vectors from its neighbors,
        average them, each atom 'knows' about atom upto 4 bonds away.
        Then we average all atom representations to get a single
        molecule-level vector.
    """

    def __init__ (
            self,
            node_features: int = 11, # size of each atom feature vector
            hidden_dim: int = 128,  # size of internal representations
            num_layers: int  = 4, # rounds of message passing
            dropout: float = 0.1
    ):
        super().__init__()

        self.num_layers = num_layers

        # Message passing layers
        # ------------------------
        # GCNConv = Graph Convolutional Network layer
        # Each layer: takes atom features -> passes messages
        # along bonds -> produces new atom features

        self.convs = nn.ModuleList()

        # fist layer: input features -> hidden_dim
        self.convs.append(GCNConv(node_features, hidden_dim))

        # remaining layers: hidden_dim -> hidden_dim
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))

        # Batch normalization
        # -------------------------
        # Normalizes activations after each layer.
        # One BN layer per GCN layer.
        self.bns = nn.ModuleList([
            nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)
        ])

        # Dropout: Randomly zeros 10%  of neurons during training.
        # ------------------------
        self.dropout = nn.Dropout(p=dropout)

        # Prediction head
        # ------------------------
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1)
        )

    def forward(self, data):
        """
        Forward pass through the GNN.

        Args:
            data: PyG Data object containing:
                data.x - atom features [num_atoms, 11]
                data.edge_index - bond connectivity [2, num_bonds*2]
                data.batch - maps atoms to their molecule index
        Returns:
            Tensor of shape [batch_size] - one prediction per molecule
        """

        x = data.x.float()          # atom features
        edge_index = data.edge_index # bond connectivity
        batch = data.batch           # which molecule each atom belongs to

        # Message passing rounds
        # ------------------------
        for conv, bn in zip(self.convs, self.bns):

            # 1. Message passing: atoms collect from neighbors
            x = conv(x, edge_index)

            # 2. Batch normalization
            x = bn(x)

            # 3. ReLU
            x = torch.relu(x)

            # 4. Dropout
            x = self.dropout(x)

        # Global pooling
        # ---------------------
        # now x shape: [total_atoms_in_batch, 128]
        # we need -> one vector per molecule, not one per atom.
        # global_mean_pool averages all atom vectors belongs to same molecule.
        # result shape: [batch_size, 128]
        x = global_mean_pool(x, batch)

        # Prediction
        return self.predictor(x).squeeze(-1)
