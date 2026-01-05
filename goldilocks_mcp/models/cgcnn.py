from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool, MessagePassing
from torch.nn import Linear, BatchNorm1d
from torch_geometric.data import Data


class Standardize(nn.Module):
    """Standardize node features: subtract mean and divide by std."""

    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: Data) -> Data:
        """Apply standardization to data.x (node features)."""
        data = data.clone()  # avoid modifying in-place
        data.x = (data.x - self.mean) / self.std
        return data

class RBFExpansion(nn.Module):
    """Radial basis function expansion for continuous features.
    
    Expands scalar features (e.g., distances) into a vector representation
    using Gaussian radial basis functions.
    
    Args:
        vmin (float, optional): Minimum value for the RBF centers. Defaults to 0.
        vmax (float, optional): Maximum value for the RBF centers. Defaults to 8.
        bins (int, optional): Number of RBF centers. Defaults to 40.
        lengthscale (float, optional): Length scale for the Gaussian kernels.
            If None, computed as the spacing between centers. Defaults to None.
    """
    def __init__(self, vmin=0, vmax=8, bins=40, lengthscale=None):
        super().__init__()
        centers = torch.linspace(vmin, vmax, bins)
        self.register_buffer("centers", centers)
        self.gamma = 1 / ((lengthscale or (centers[1] - centers[0]).item()) ** 2)

    def forward(self, distance):
        """Expand scalar distances into RBF features.
        
        Args:
            distance (torch.Tensor): Scalar distance values of shape [num_edges]
                or [num_edges, 1].
        
        Returns:
            torch.Tensor: RBF-expanded features of shape [num_edges, bins].
        """
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)

class CGCNNConv(MessagePassing):
    """CGCNN gated convolution layer.
    
    Implements the gated convolution mechanism from CGCNN:
    h_i^(t+1) = h_i^(t) + sum_over_neighbors[sigmoid(z_ij W_gate + b) * softplus(z_ij W_conv + d)]
    where z_ij = concat(v_i, v_j, e_ij) is the concatenated features of source node,
    destination node, and edge between them.
    
    Args:
        node_dim (int): Dimension of node features.
        edge_dim (int): Dimension of edge features (after RBF expansion).
        out_dim (int): Dimension of output node features.
    """
    def __init__(self, node_dim, edge_dim, out_dim):
        super().__init__(aggr='add')  # sum aggregation
        
        self.lin_f = Linear(2 * node_dim + edge_dim, out_dim)
        self.lin_s = Linear(2 * node_dim + edge_dim, out_dim)
        self.batch_norm = BatchNorm1d(out_dim)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass through the CGCNN convolution layer.
        
        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, node_dim].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix of shape [num_edges, edge_dim].
        
        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_dim].
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Compute gated messages from source nodes to destination nodes.
        
        Args:
            x_i (torch.Tensor): Source node features of shape [num_edges, node_dim].
            x_j (torch.Tensor): Destination node features of shape [num_edges, node_dim].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, edge_dim].
        
        Returns:
            torch.Tensor: Gated messages of shape [num_edges, out_dim].
        """
        z = torch.cat([x_i, x_j, edge_attr], dim=1)
        gate = torch.sigmoid(self.lin_f(z))
        msg = F.softplus(self.lin_s(z))
        return gate * msg

    def update(self, aggr_out, x):
        """Update node features with aggregated messages.
        
        Args:
            aggr_out (torch.Tensor): Aggregated messages of shape [num_nodes, out_dim].
            x (torch.Tensor): Original node features of shape [num_nodes, node_dim].
        
        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_dim].
        """
        return self.batch_norm(aggr_out+x)


class CGCNN_PyG(nn.Module):
    """CGCNN (Crystal Graph Convolutional Neural Network) model for materials property prediction.
    
    This model uses graph convolutional layers to learn representations of crystal
    structures. It supports regression, classification, robust regression, and
    quantile regression tasks.
    
    Args:
        orig_atom_fea_len (int): Number of input atomic features.
        edge_feat_dim (int, optional): Number of RBF bins for edge features. Defaults to 64.
        name (str, optional): Model name identifier. Defaults to 'cgcnn'.
        h_fea_len (int, optional): Number of hidden features after pooling. Defaults to 128.
        atom_fea_len (int, optional): Number of hidden atom features in convolutional layers.
            Defaults to 64.
        n_conv (int, optional): Number of convolutional layers. Defaults to 3.
        n_h (int, optional): Number of hidden layers after pooling. Defaults to 3.
        robust_regression (bool, optional): If True, output mean and std for robust regression.
            Defaults to False.
        classification (bool, optional): If True, use classification output. Defaults to False.
        quantile_regression (bool, optional): If True, output quantiles. Defaults to False.
        num_quantiles (int, optional): Number of quantiles to predict. Defaults to 1.
        pooling_type (str, optional): Type of pooling ('mean_pool'). Defaults to 'mean_pool'.
        num_classes (int, optional): Number of classes for classification. Defaults to 2.
        additional_compound_features (bool, optional): If True, include additional
            compound-level features. Defaults to False.
        add_feat_len (int, optional): Length of additional compound features. Defaults to None.
    """
    def __init__(self, orig_atom_fea_len, edge_feat_dim=64, name ='cgcnn',
                 h_fea_len=128, atom_fea_len=64, n_conv=3, n_h=3,
                 robust_regression=False, classification=False,
                 quantile_regression=False, num_quantiles = 1,
                 pooling_type = 'mean_pool', num_classes=2,
                 additional_compound_features=False, add_feat_len=None):
        super().__init__()
        self.name=name
        self.classification = classification
        self.robust_regression = robust_regression
        self.quantile_regression = quantile_regression

        if classification:
            self.num_classes = num_classes
        elif quantile_regression:
            self.num_quantiles = num_quantiles

        self.global_pooling = pooling_type
        self.additional_compound_features = additional_compound_features
        if self. additional_compound_features:
            self.add_feat_len=add_feat_len
        
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)

        # Add RBFExpander inside the model
        self.rbf = RBFExpansion(vmin=0, vmax=8.0, bins=edge_feat_dim)

        # Your CGCNNConv would now take RBF-expanded edge_attr
        self.convs = nn.ModuleList([
            CGCNNConv(atom_fea_len, edge_feat_dim, out_dim=atom_fea_len) 
            for _ in range(n_conv)
        ])
        self.conv_to_fc_softplus = nn.Softplus()

        if(self.additional_compound_features):
            self.add_feat_norm =  nn.BatchNorm1d(add_feat_len)
            self.proj_add_feat = nn.Linear(add_feat_len,atom_fea_len)
            self.conv_to_fc = nn.Linear(2*atom_fea_len, h_fea_len)
            self.softplus = nn.Softplus()
        else:
            self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])
        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, self.num_classes)
        elif self.robust_regression:
            self.fc_out = nn.Linear(h_fea_len, 2)
        elif self.quantile_regression:
            self.fc_out = nn.Linear(h_fea_len, self.num_quantiles)
        else:
            self.fc_out = nn.Linear(h_fea_len, 1)
        
    def forward(self, data):
        """Forward pass through the CGCNN model.
        
        Args:
            data (torch_geometric.data.Data): Graph data object with:
                - x: Node features of shape [num_nodes, orig_atom_fea_len]
                - edge_index: Edge connectivity of shape [2, num_edges]
                - edge_attr: Edge distances of shape [num_edges] (scalar)
                - batch: Batch assignment of shape [num_nodes]
                - additional_compound_features: Additional features of shape [batch_size, add_feat_len]
                  (optional, only if additional_compound_features=True)
        
        Returns:
            torch.Tensor: Model predictions of shape [batch_size] for regression,
                or [batch_size, num_classes] for classification, or [batch_size, num_quantiles]
                for quantile regression.
        """
        if self.additional_compound_features:
            x, edge_index, edge_attr, batch, add_feat, _ = data.x, data.edge_index, data.edge_attr, data.batch, data.additional_compound_features, data.y
            add_feat = add_feat.view(-1, self.add_feat_len)
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.embedding(x)
        # Apply RBFExpansion to scalar edge_attr (e.g. bond lengths)
        edge_attr = self.rbf(edge_attr.view(-1))  # ensure it's [num_edges]

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        if(self.global_pooling =='mean_pool'):    
            x = global_mean_pool(x, batch)
        
        x = self.conv_to_fc_softplus(x)

        if self.additional_compound_features:
            add_feat = self.add_feat_norm(add_feat)
            add_feat = self.proj_add_feat(add_feat)
            add_feat = self.softplus(add_feat)
            combined = torch.cat([x, add_feat], dim=1)
            x = self.conv_to_fc(combined)
        else:
            x = self.conv_to_fc(x)

        x = self.conv_to_fc_softplus(x)

        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                x = softplus(fc(x))
        out = self.fc_out(x)
        return out
    
    def extract_crystal_repr(self, data):
        """Extract crystal representations after graph convolution and pooling.
        
        This method returns the learned representation of the crystal structure
        before the final output layer, which can be used for transfer learning
        or feature analysis.
        
        Args:
            data (torch_geometric.data.Data): Graph data object with:
                - x: Node features of shape [num_nodes, orig_atom_fea_len]
                - edge_index: Edge connectivity of shape [2, num_edges]
                - edge_attr: Edge distances of shape [num_edges] (scalar)
                - batch: Batch assignment of shape [num_nodes]
                - additional_compound_features: Additional features (optional)
        
        Returns:
            torch.Tensor: Crystal representation of shape [batch_size, atom_fea_len].
        """
        if self.additional_compound_features:
            x, edge_index, edge_attr, batch, add_feat, _ = data.x, data.edge_index, data.edge_attr, data.batch, data.additional_compound_features, data.y
            add_feat = add_feat.view(-1, self.add_feat_len)
        else:
            x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.embedding(x)
        edge_attr = self.rbf(edge_attr.view(-1))

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
        if(self.global_pooling =='mean_pool'):    
            x = global_mean_pool(x, batch)
        return x


