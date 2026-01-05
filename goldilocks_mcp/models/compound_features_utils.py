import numpy as np
import pandas as pd
from typing import List
import matminer
from matminer.featurizers import composition as composition_featurizers
from matminer.featurizers import structure as structure_featurizers
from matminer.featurizers.base import MultipleFeaturizer
from dscribe.descriptors import SOAP
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core.composition import Composition
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

import sys
from pathlib import Path
# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))


def normalize_formulas(df: pd.DataFrame, formula_column: str = 'formula') -> pd.DataFrame:
    """Normalize chemical formulas to IUPAC format.
    
    Converts chemical formulas to IUPAC standard format, removing duplicates
    that arise from different structural representations of the same compound.
    
    Args:
        df (pd.DataFrame): DataFrame containing at least a formula column.
        formula_column (str, optional): Name of the column containing formulas.
            Defaults to 'formula'.
    
    Returns:
        pd.DataFrame: DataFrame with normalized formulas in the specified column.
    """
    formula=[]
    for form in df[formula_column].values:
        formula.append(Composition(Composition(form).get_integer_formula_and_factor()[0]).iupac_formula)
    df[formula_column]=formula
    return df 

def matminer_composition_features(df: pd.DataFrame, 
                                  list_of_features: List, 
                                  formula_column = 'formula'):
    """Calculate composition-based features using matminer featurizers.
    
    Computes various composition descriptors (e.g., elemental properties,
    stoichiometric features) for each compound in the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing chemical formulas.
        list_of_features (List[str]): List of matminer featurizer names to use.
            Examples: ['ElementProperty', 'Stoichiometry', 'ValenceOrbital'].
        formula_column (str, optional): Name of the column containing formulas.
            Defaults to 'formula'.
    
    Returns:
        numpy.ndarray: Feature matrix of shape [num_compounds, feature_dim].
            NaN values are replaced with 0.0.
    """
    df = normalize_formulas(df, formula_column)
    df['composition'] = [Composition(form) for form in df[formula_column]]
    
    list_of_feat_meth=[]
    for feat in list_of_features:
        if hasattr(composition_featurizers, feat):
            if(feat=='ElementProperty'):
                method = getattr(matminer.featurizers.composition , feat).from_preset('magpie', impute_nan=True)
            else:
                try:
                    method = getattr(matminer.featurizers.composition , feat)(impute_nan=True)
                except Exception as _:
                    method = getattr(matminer.featurizers.composition , feat)()
            list_of_feat_meth.append(method)
            
            # Use individual featurizers instead of MultipleFeaturizer to avoid argument passing issues
    composition_featurizer = MultipleFeaturizer(list_of_feat_meth)
    
    comp_feat_len = len(composition_featurizer.featurize(df.iloc[0]['composition']))
    features=np.zeros((len(df),comp_feat_len))     
    for i,comp in enumerate(df['composition'].values):
        features[i,:]=composition_featurizer.featurize(comp)
    
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def matminer_structure_features(df: pd.DataFrame,
                                list_of_features: List,
                                structure_column = 'structure'):
    """Calculate structure-based features using matminer featurizers.
    
    Computes various structural descriptors (e.g., symmetry features, density)
    for each crystal structure in the dataframe.
    
    Args:
        df (pd.DataFrame): DataFrame containing pymatgen Structure objects.
        list_of_features (List[str]): List of matminer structure featurizer names.
            Examples: ['GlobalSymmetryFeatures', 'DensityFeatures'].
        structure_column (str, optional): Name of the column containing structures.
            Defaults to 'structure'.
    
    Returns:
        numpy.ndarray: Feature matrix of shape [num_compounds, feature_dim].
            NaN values are replaced with 0.0. Failed featurizations result in zero vectors.
    
    Note:
        Prints warnings for structures that fail featurization.
    """
    list_of_feat_meth=[]
    for feat in list_of_features:
        if(feat=='GlobalSymmetryFeatures'):
            props=["spacegroup_num", "crystal_system_int", "is_centrosymmetric"]
            method = getattr(structure_featurizers, feat)(props)
        elif(feat=='DensityFeatures'):
            props=["density", "vpa", "packing fraction"]
            method = getattr(structure_featurizers, feat)(props)
        list_of_feat_meth.append(method)
    
    structure_featurizer = MultipleFeaturizer(list_of_feat_meth)
    struct_feat_len = len(structure_featurizer.featurize(df.iloc[0][structure_column]))
    features=np.zeros((len(df),struct_feat_len))
    for i, struct in enumerate(df[structure_column].values):
        try:
            features[i, :] = structure_featurizer.featurize(struct)
        except Exception as e:
            print(f"Warning: structure {struct.formula} at index {i} failed featurization: {e}")
            features[i, :] = 0.0  # fallback: zeros

    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features


def lattice_features(df: pd.DataFrame, structure_column: str = 'structure'):
    """Calculate lattice-related features for crystal structures.
    
    Extracts geometric and symmetry information from crystal lattices:
    - Lattice constants (a, b, c)
    - Lattice angles (alpha, beta, gamma)
    - Reciprocal lattice constants and angles
    - Space group number
    - Crystal system (encoded as integer)
    - Bravais lattice type (encoded as integer)
    
    Args:
        df (pd.DataFrame): DataFrame containing pymatgen Structure objects.
        structure_column (str, optional): Name of the column containing structures.
            Defaults to 'structure'.
    
    Returns:
        numpy.ndarray: Feature matrix of shape [num_compounds, 15].
            Features are: [a, b, c, alpha, beta, gamma, a*, b*, c*, alpha*, beta*, gamma*,
            crystal_system_id, bravais_id, space_group_number].
            NaN values are replaced with 0.0.
    
    Note:
        Prints warnings for structures that fail feature calculation.
    """
    # 7 crystal systems
    crystal_system_map = {
        "triclinic": 0,
        "monoclinic": 1,
        "orthorhombic": 2,
        "tetragonal": 3,
        "trigonal": 4,
        "hexagonal": 5,
        "cubic": 6
    }
    # 14 Bravais lattices (symbols and encodings)
    bravais_map = {
        "aP": 0,  # triclinic primitive
        "mP": 1, "mC": 2,  # monoclinic
        "oP": 3, "oC": 4, "oI": 5, "oF": 6,  # orthorhombic
        "tP": 7, "tI": 8,  # tetragonal
        "hP": 9, "hR": 10,  # hexagonal/trigonal
        "cP": 11, "cI": 12, "cF": 13  # cubic
    }
    # Map to abbreviations
    system_abbr = {
        "triclinic": "a",
        "monoclinic": "m",
        "orthorhombic": "o",
        "tetragonal": "t",
        "trigonal": "h",
        "hexagonal": "h",
        "cubic": "c"
    }
    features=np.zeros((len(df),15))
    for i,structure in enumerate(df[structure_column].values):
        try:
            feature=[]
            for x in structure.lattice.abc:
                feature.append(x)
            for x in structure.lattice.angles:
                feature.append(x)
            for x in structure.lattice.reciprocal_lattice.abc:
                feature.append(x)
            for x in structure.lattice.reciprocal_lattice.angles:
                feature.append(x)
            sga = SpacegroupAnalyzer(structure, symprec=0.01)
            spg_symbol = sga.get_space_group_symbol()
            spg_number = sga.get_space_group_number()
            crystal_system = sga.get_crystal_system()
            centering = spg_symbol[0]
            bravais = system_abbr[crystal_system] + centering
            crystal_system_id = crystal_system_map[crystal_system]
            bravais_id = bravais_map.get(bravais, -1)
            feature.append(crystal_system_id)
            feature.append(bravais_id)
            feature.append(spg_number)
            feature=np.array(feature)
            features[i,:]=feature
        except Exception as _:
            print(f'failed to calculate lattice features for {structure.formula}')
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features

def soap_features(df: pd.DataFrame,
                  soap_params = {'r_cut': 10.0, 'n_max': 8, 'l_max': 6, 'sigma': 1.0},
                  structure_column = 'structure'):
    """Calculate SOAP (Smooth Overlap of Atomic Positions) features for compounds.
    
    Computes SOAP descriptors averaged over all atoms in each structure.
    All atoms are treated as the same element 'X' for SOAP calculation.
    
    Args:
        df (pd.DataFrame): DataFrame containing pymatgen Structure objects.
        soap_params (dict, optional): Dictionary containing SOAP parameters:
            - r_cut (float): Cutoff radius for SOAP. Defaults to 10.0.
            - n_max (int): Maximum radial basis functions. Defaults to 8.
            - l_max (int): Maximum angular momentum. Defaults to 6.
            - sigma (float): Gaussian width parameter. Defaults to 1.0.
        structure_column (str, optional): Name of the column containing structures.
            Defaults to 'structure'.
    
    Returns:
        numpy.ndarray: Feature matrix of shape [num_compounds, soap_feature_dim].
            Each row is the mean SOAP feature vector over all atoms in the structure.
            NaN values are replaced with 0.0.
    """
    soap_featurizer = SOAP(species=['X'],  # or whatever elements you're using
                               r_cut=soap_params['r_cut'],
                               n_max=soap_params['n_max'],
                               l_max=soap_params['l_max'],
                               sigma=soap_params['sigma'],
                               periodic=True,
                               sparse=False)

    atoms = AseAtomsAdaptor.get_atoms(df.iloc[0][structure_column])
    atoms.set_chemical_symbols(["X"] * len(atoms))
    soap=soap_featurizer.create(atoms).mean(axis=0)
    soap_feat_len = len(soap)
    features=np.zeros((len(df),soap_feat_len))
    for i,struct in enumerate(df[structure_column].values):
        atoms = AseAtomsAdaptor.get_atoms(struct)
        atoms.set_chemical_symbols(["X"] * len(atoms))
        features[i,:]=soap_featurizer.create(atoms).mean(axis=0)
    
    features=np.nan_to_num(features, copy=True, nan=0.0, posinf=None, neginf=None)
    return features




