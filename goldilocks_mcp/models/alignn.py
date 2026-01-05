import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool
from torch_geometric.data import Data


class EdgeGatedGraphConvPyG(MessagePassing):
    """Edge-gated graph convolution layer with corrected message normalization.
    
    This layer implements an edge-gated message passing mechanism where messages
    are gated by a combination of source node, destination node, and edge features.
    The gating mechanism uses a SiLU activation function, and the layer supports
    residual connections and normalization (LayerNorm or BatchNorm).
    
    Args:
        in_channels (int): Number of input node features.
        out_channels (int): Number of output node features.
        residual (bool, optional): Whether to use residual connections. Defaults to True.
        use_layer_norm (bool, optional): If True, use LayerNorm; otherwise use BatchNorm1d.
            Defaults to True.
    """
    
    def __init__(self, in_channels, out_channels, residual=True, use_layer_norm=True):
        super().__init__(aggr='add')
        self.residual = residual
        self.use_layer_norm = use_layer_norm

        self.src_gate = nn.Linear(in_channels, out_channels)
        self.dst_gate = nn.Linear(in_channels, out_channels)
        self.edge_gate = nn.Linear(in_channels, out_channels)

        self.src_update = nn.Linear(in_channels, out_channels)
        self.dst_update = nn.Linear(in_channels, out_channels)

        if use_layer_norm:
            self.norm = nn.LayerNorm(out_channels)
        else:
            self.norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edge_index, edge_attr):
        """Forward pass through the edge-gated graph convolution layer.
        
        Args:
            x (torch.Tensor): Node feature matrix of shape [num_nodes, in_channels].
            edge_index (torch.Tensor): Graph connectivity in COO format with shape [2, num_edges].
            edge_attr (torch.Tensor): Edge feature matrix of shape [num_edges, in_channels].
        
        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        """Compute gated messages from source nodes to destination nodes.
        
        Args:
            x_i (torch.Tensor): Source node features of shape [num_edges, in_channels].
            x_j (torch.Tensor): Destination node features of shape [num_edges, in_channels].
            edge_attr (torch.Tensor): Edge features of shape [num_edges, in_channels].
        
        Returns:
            torch.Tensor: Gated messages of shape [num_edges, out_channels].
        """
        # Compute gate: element-wise combination of source, dest, and edge features
        gate = self.src_gate(x_i) + self.dst_gate(x_j) + self.edge_gate(edge_attr)
        gate = F.silu(gate)
        
        # Message: gated update from destination node
        msg = gate * self.dst_update(x_j)
        return msg
  
    def update(self, aggr_out, x):
        """Update node features with aggregated messages.
        
        Args:
            aggr_out (torch.Tensor): Aggregated messages of shape [num_nodes, out_channels].
            x (torch.Tensor): Original node features of shape [num_nodes, in_channels].
        
        Returns:
            torch.Tensor: Updated node features of shape [num_nodes, out_channels].
        """
        # Update: source node feature + aggregated messages
        out = self.src_update(x) + aggr_out
        
        # Normalize
        out = self.norm(out)
        out = F.silu(out)
        
        # Residual connection
        if self.residual and x.shape == out.shape:
            out = out + x
        
        return out


class ALIGNNConvPyG(nn.Module):
    """One ALIGNN layer: edge updates on line graph, then node updates on bond graph.
    
    ALIGNN (Atomistic Line Graph Neural Network) layers update edge features
    using a line graph representation (where edges become nodes), then update
    node features using the updated edge features. This allows the model to
    capture both bond and angle information.
    
    Args:
        hidden_dim (int): Hidden feature dimension for nodes and edges.
        use_layer_norm (bool, optional): If True, use LayerNorm in sub-layers;
            otherwise use BatchNorm1d. Defaults to True.
    """
    
    def __init__(self, hidden_dim, use_layer_norm=True):
        super().__init__()
        self.node_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim, use_layer_norm=use_layer_norm)
        self.edge_update = EdgeGatedGraphConvPyG(hidden_dim, hidden_dim, use_layer_norm=use_layer_norm)

    def forward(self, data_g, data_lg):
        """Forward pass through the ALIGNN layer.
        
        Args:
            data_g (torch_geometric.data.Data): Atomic graph with node features (x),
                edge indices (edge_index), and edge attributes (edge_attr).
            data_lg (torch_geometric.data.Data): Line graph where nodes correspond
                to edges in the atomic graph. Should have node features (x),
                edge indices (edge_index), and edge attributes (edge_attr) representing angles.
        
        Returns:
            tuple: A tuple containing:
                - x (torch.Tensor): Updated node features of shape [num_nodes, hidden_dim].
                - edge_attr (torch.Tensor): Updated edge features of shape [num_edges, hidden_dim].
        """
        # Update edges using line graph
        edge_attr = self.edge_update(data_lg.x, data_lg.edge_index, data_lg.edge_attr)
        
        # Update nodes using updated edges
        x = self.node_update(data_g.x, data_g.edge_index, edge_attr)
        
        return x, edge_attr


class RBFExpansion(nn.Module):
    """Radial basis function expansion for continuous features.
    
    Expands scalar features (e.g., distances, angles) into a vector representation
    using Gaussian radial basis functions. This is useful for encoding continuous
    geometric information like bond distances or angles.
    
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
        if lengthscale is None:
            lengthscale = (centers[1] - centers[0]).item()
        self.gamma = 1.0 / (lengthscale ** 2)

    def forward(self, distance):
        """Expand scalar distances into RBF features.
        
        Args:
            distance (torch.Tensor): Scalar distance values of shape [num_edges]
                or [num_edges, 1].
        
        Returns:
            torch.Tensor: RBF-expanded features of shape [num_edges, bins].
        """
        # RBF: exp(-gamma * (distance - center)^2)
        return torch.exp(-self.gamma * (distance.unsqueeze(1) - self.centers) ** 2)


class Standardize(nn.Module):
    """Standardize node features by subtracting mean and dividing by standard deviation.
    
    This module applies z-score normalization to node features, which is useful
    for preprocessing input features to have zero mean and unit variance.
    
    Args:
        mean (torch.Tensor): Mean values for each feature dimension.
        std (torch.Tensor): Standard deviation values for each feature dimension.
    """
    
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, data: Data) -> Data:
        """Apply standardization to node features.
        
        Args:
            data (torch_geometric.data.Data): Graph data object with node features (x).
        
        Returns:
            torch_geometric.data.Data: New data object with standardized node features.
        """
        data = data.clone()
        data.x = (data.x - self.mean) / (self.std + 1e-8)
        return data


class ALIGNN_PyG(nn.Module):
    """ALIGNN (Atomistic Line Graph Neural Network) model for materials property prediction.
    
    This model uses both atomic graphs and line graphs to capture bond and angle
    information in crystal structures. It supports regression, classification,
    robust regression, and quantile regression tasks.
    
    Args:
        atom_input_features (int): Number of input atomic features.
        hidden_features (int, optional): Hidden feature dimension. Defaults to 64.
        radius (float, optional): Cutoff radius for neighbor search. Defaults to 10.0.
        edge_input_features (int, optional): Number of RBF bins for edge features.
            Defaults to 40.
        triplet_input_features (int, optional): Number of RBF bins for angle features.
            Defaults to 20.
        alignn_layers (int, optional): Number of ALIGNN layers. Defaults to 4.
        gcn_layers (int, optional): Number of final GCN layers. Defaults to 4.
        classification (bool, optional): If True, use classification output.
            Defaults to False.
        num_classes (int, optional): Number of classes for classification. Defaults to 2.
        robust_regression (bool, optional): If True, output mean and std for robust
            regression. Defaults to False.
        quantile_regression (bool, optional): If True, output quantiles. Defaults to False.
        num_quantiles (int, optional): Number of quantiles to predict. Defaults to 1.
        name (str, optional): Model name identifier. Defaults to 'alignn'.
        use_layer_norm (bool, optional): If True, use LayerNorm; otherwise BatchNorm1d.
            Defaults to True.
        additional_compound_features (bool, optional): If True, include additional
            compound-level features. Defaults to False.
        add_feat_len (int, optional): Length of additional compound features.
            Defaults to 231.
    """

    def __init__(self, 
                 atom_input_features,
                 hidden_features=64,
                 radius=10.0,
                 edge_input_features=40,
                 triplet_input_features=20,
                 alignn_layers=4,
                 gcn_layers=4,
                 classification=False,
                 num_classes=2,
                 robust_regression=False,
                 quantile_regression=False,
                 num_quantiles=1,
                 name='alignn',
                 use_layer_norm=True,
                 additional_compound_features=False,
                 add_feat_len=231):
        super().__init__()
        self.name = name
        self.hidden_features = hidden_features
        self.additional_compound_features = additional_compound_features
        
        if self.additional_compound_features:
            self.add_feat_len = add_feat_len

        # Atom embedding
        self.atom_embedding = nn.Sequential(
            nn.Linear(atom_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # Edge (bond distance) embedding
        self.edge_rbf = RBFExpansion(0, radius, edge_input_features)
        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # Angle (triplet) embedding
        self.angle_rbf = RBFExpansion(-1.0, 1.0, triplet_input_features)
        self.angle_embedding = nn.Sequential(
            nn.Linear(triplet_input_features, hidden_features),
            nn.LayerNorm(hidden_features) if use_layer_norm else nn.BatchNorm1d(hidden_features),
            nn.SiLU(),
        )

        # ALIGNN layers
        self.alignn_layers = nn.ModuleList([
            ALIGNNConvPyG(hidden_features, use_layer_norm=use_layer_norm) 
            for _ in range(alignn_layers)
        ])

        # Final GCN layers
        self.gcn_layers = nn.ModuleList([
            EdgeGatedGraphConvPyG(hidden_features, hidden_features, use_layer_norm=use_layer_norm) 
            for _ in range(gcn_layers)
        ])

        # Readout
        self.readout = global_mean_pool
        
        # Additional compound features processing
        if self.additional_compound_features:
            norm_layer = nn.LayerNorm(add_feat_len) if use_layer_norm else nn.BatchNorm1d(add_feat_len)
            self.add_feat_norm = norm_layer
            self.proj_add_feat = nn.Linear(add_feat_len, hidden_features)
            self.add_feat_activation = nn.SiLU()
            self.conv_to_fc = nn.Linear(2 * hidden_features, hidden_features)
        else:
            self.conv_to_fc = nn.Linear(hidden_features, hidden_features)

        self.fc_activation = nn.SiLU()

        # Output layer
        if classification:
            self.output_layer = nn.Linear(hidden_features, num_classes)
        elif robust_regression:
            self.output_layer = nn.Linear(hidden_features, 2)
        elif quantile_regression:
            self.output_layer = nn.Linear(hidden_features, num_quantiles)
        else:
            self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, data_g, data_lg):
        """Forward pass through the ALIGNN model.
        
        Args:
            data_g (torch_geometric.data.Data): Atomic graph with:
                - x: Node features of shape [num_nodes, atom_input_features]
                - edge_index: Edge connectivity of shape [2, num_edges]
                - edge_attr: Edge distances of shape [num_edges] (scalar)
                - batch: Batch assignment of shape [num_nodes] (optional)
                - additional_compound_features: Additional features of shape [batch_size, add_feat_len]
                  (optional, only if additional_compound_features=True)
            data_lg (torch_geometric.data.Data): Line graph with:
                - x: Edge features from atomic graph (will be set internally)
                - edge_index: Line graph connectivity of shape [2, num_line_edges]
                - edge_attr: Angle cosines of shape [num_line_edges] (scalar)
        
        Returns:
            torch.Tensor: Model predictions of shape [batch_size] for regression,
                or [batch_size, num_classes] for classification, or [batch_size, num_quantiles]
                for quantile regression.
        """
        # Embed features
        x = self.atom_embedding(data_g.x)
        
        # Apply RBF expansion first, then embedding
        edge_attr = self.edge_rbf(data_g.edge_attr)
        edge_attr = self.edge_embedding(edge_attr)
        
        angle_attr = self.angle_rbf(data_lg.edge_attr)
        angle_attr = self.angle_embedding(angle_attr)

        # Set graph data
        data_g.x = x
        data_g.edge_attr = edge_attr
        data_lg.x = edge_attr  # nodes in line graph = edges in original graph
        data_lg.edge_attr = angle_attr

        has_line_graph = data_lg.edge_index.shape[1] > 0
        if has_line_graph:
            # ALIGNN layers (normal path)
            for layer in self.alignn_layers:
                x, edge_attr = layer(data_g, data_lg)
                data_g.x = x
                data_g.edge_attr = edge_attr
                data_lg.x = edge_attr
        else:
            # Skip ALIGNN layers, just use embedded features
            x = data_g.x
            edge_attr = data_g.edge_attr

        # Final GCN layers
        for gcn in self.gcn_layers:
            x = gcn(x, data_g.edge_index, edge_attr)

        # Readout and output
        x_pool = self.readout(x, data_g.batch)
        
        if self.additional_compound_features:
            # Reshape additional features to [batch_size, add_feat_len]
            add_feat = data_g.additional_compound_features.view(-1, self.add_feat_len)
            add_feat = self.add_feat_norm(add_feat)
            add_feat = self.proj_add_feat(add_feat)
            add_feat = self.add_feat_activation(add_feat)
            combined = torch.cat([x_pool, add_feat], dim=1)
            x_pool = self.conv_to_fc(combined)
        else:
            x_pool = self.conv_to_fc(x_pool)
        
        x_pool = self.fc_activation(x_pool)
        output = self.output_layer(x_pool).squeeze(-1)
        
        return output