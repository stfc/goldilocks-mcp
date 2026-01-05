from pymatgen.core.structure import Structure
from pymatgen.analysis.local_env import CrystalNN
from typing import List
import torch
from torch_geometric.data import Data
import warnings


def build_radius_cgcnn_graph_from_structure(structure: Structure, 
                                         atom_features: List, 
                                         radius: float=10.0, 
                                         max_neighbors: int=12) -> Data:
    """Convert a pymatgen Structure to a PyTorch Geometric Data object using radius-based neighbor search.
    
    Constructs a graph where nodes are atoms and edges connect atoms within
    a specified radius. Edge attributes store the interatomic distances.
    
    Args:
        structure (pymatgen.core.structure.Structure): Crystal structure to convert.
        atom_features (List): List of atomic feature vectors, one per atom.
            Each element should be a list or array of features.
        radius (float, optional): Cutoff radius for neighbor search in Angstroms.
            Defaults to 10.0.
        max_neighbors (int, optional): Maximum number of neighbors per atom.
            Defaults to 12.
    
    Returns:
        torch_geometric.data.Data: Graph data object with:
            - x: Node features of shape [num_atoms, feature_dim]
            - edge_index: Edge connectivity of shape [2, num_edges]
            - edge_attr: Edge distances of shape [num_edges, 1]
    
    Warns:
        UserWarning: If some atoms have no neighbors within the radius.
    """
    
    x = torch.tensor(atom_features, dtype=torch.float32) 
    # Edge features: collect neighbors
    edge_index = []
    edge_attr = []
    
    all_neighbors = structure.get_all_neighbors(radius, include_index=True)
    disconnected_atoms=[]

    for i, neighbors in enumerate(all_neighbors):
        neighbors = sorted(neighbors, key=lambda x: x[1])[:max_neighbors]  # take closest max_neighbors
        if len(neighbors) == 0:
            disconnected_atoms.append(i)
        for neighbor in neighbors:
            j = neighbor[2]  # neighbor atom index
            dist = neighbor[1]
            edge_index.append([i, j])
            edge_attr.append([dist])
    
    if disconnected_atoms:
        warnings.warn(
            f"{len(disconnected_atoms)} atoms had no neighbors within radius {radius}. "
            f"Disconnected atom indices: {disconnected_atoms}"
        )

    # Convert to tensors
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
    
    # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def build_crystalnn_cgcnn_graph_from_structure(structure: Structure, 
                                         atom_features: List, 
                                         radius: float=10.0) -> Data:
    """Convert a pymatgen Structure to a PyTorch Geometric Data object using CrystalNN neighbor search.
    
    Constructs a graph where nodes are atoms and edges connect atoms identified
    by the CrystalNN algorithm, which uses coordination environments rather than
    simple distance cutoffs. Edge attributes store the interatomic distances.
    
    Args:
        structure (pymatgen.core.structure.Structure): Crystal structure to convert.
        atom_features (List): List of atomic feature vectors, one per atom.
            Each element should be a list or array of features.
        radius (float, optional): Maximum search radius for CrystalNN in Angstroms.
            Defaults to 10.0.
    
    Returns:
        torch_geometric.data.Data: Graph data object with:
            - x: Node features of shape [num_atoms, feature_dim]
            - edge_index: Edge connectivity of shape [2, num_edges]
            - edge_attr: Edge distances of shape [num_edges, 1]
    
    Warns:
        UserWarning: If some atoms have no neighbors found by CrystalNN.
    """
    x = torch.tensor(atom_features, dtype=torch.float32) 
    # Edge features: collect neighbors
    edge_index = []
    edge_attr = []

    local_env = CrystalNN(distance_cutoffs=[0.5,radius],search_cutoff=radius)
    disconnected_atoms=[]
    
    for i in range(len(structure)):
        nn=local_env.get_nn_info(structure, i)
        if len(nn) == 0:
            disconnected_atoms.append(i)
        for neighbor in nn:
            j=neighbor['site_index']
            site = neighbor['site']
            dist = structure[0].distance(site)
            edge_index.append([i, j])
            edge_attr.append([dist])

    if disconnected_atoms:
        warnings.warn(
                f"{len(disconnected_atoms)} atoms had no neighbors within radius {radius}. "
                f"Disconnected atom indices: {disconnected_atoms}"
            )

    # Convert to tensors
    if edge_index:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float32)
        
        # Create PyG Data object
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data




