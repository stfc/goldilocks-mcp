import numpy as np
import pandas as pd
import torch
import joblib
from pymatgen.core.composition import Composition
from models.alignn import ALIGNN_PyG
from models.cgcnn import CGCNN_PyG
from models.cgcnn_graph import build_radius_cgcnn_graph_from_structure
from models.alignn_graph import build_alignn_graph_with_angles_from_structure
from models.atom_features_utils import atom_features_from_structure
from models.compound_features_utils import matminer_composition_features, matminer_structure_features
from models.compound_features_utils import soap_features, lattice_features

from huggingface_hub import hf_hub_download
from huggingface_hub import HfApi

data_type_np = np.float32
data_type_torch = torch.float32

def predict_kspacing(structure, model_name, confidence_level=0.95):
    """
    Predict the k-point spacing for a structure using machine learning models.
    
    This function predicts optimal k-point spacing (kdist) for Quantum Espresso
    calculations using either Random Forest (RF) or ALIGNN models. The models
    predict median k-point spacing along with confidence intervals.
    
    Args:
        structure (pymatgen.core.structure.Structure): Crystal structure to predict
            k-point spacing for.
        model_name (str): Model to use for prediction. Must be either 'RF' 
            (Random Forest) or 'ALIGNN' (Atomistic Line Graph Neural Network).
        confidence_level (float, optional): Confidence level for prediction intervals.
            Must be one of 0.95, 0.9, or 0.85. Defaults to 0.95.
    
    Returns:
        tuple: A tuple containing:
            - kdist (float): Predicted k-point spacing (median prediction)
            - kdist_upper (float): Upper bound of confidence interval
            - kdist_lower (float): Lower bound of confidence interval
    
    Raises:
        ValueError: If model_name is not 'RF' or 'ALIGNN'.
        ValueError: If confidence_level is not one of the supported values.
        FileNotFoundError: If required model files cannot be downloaded.
    
    Note:
        Models are automatically downloaded from Hugging Face Hub on first use.
        The function applies conformalized corrections to the confidence intervals
        based on the selected confidence level.
    
    Example:
        >>> from pymatgen.core.structure import Structure
        >>> structure = Structure.from_file("structure.cif")
        >>> kdist, kdist_upper, kdist_lower = predict_kspacing(
        ...     structure, model_name='ALIGNN', confidence_level=0.95
        ... )
        >>> print(f"Predicted k-point spacing: {kdist:.4f}")
    """
    # corrections are calculated when the models were trained and calibrated
    if model_name=='ALIGNN':
        if confidence_level==0.95:
            corr=0.0005
        elif confidence_level==0.9:
            corr=0.0025
        elif confidence_level==0.85:
            corr=-0.0018
    elif model_name=='RF':
        if confidence_level==0.95:
            corr=-0.0016
        elif confidence_level==0.9:
            corr=-0.0023
        elif confidence_level==0.85:
            corr=-0.0021

    comp=Composition(structure.formula)
    formula=Composition(Composition(structure.formula).get_integer_formula_and_factor()[0]).iupac_formula
    df=pd.DataFrame()
    df['id']=[0]
    df['structure']=[structure]
    df['formula']=[formula]
    df['composition']=[comp]
    
    # composition features
    composition_features = matminer_composition_features(df, ['ElementProperty', 'Stoichiometry', 'ValenceOrbital'])

    # structure features
    structure_features = matminer_structure_features(df, ['GlobalSymmetryFeatures','DensityFeatures'])

    # soap features
    soap = soap_features(df, soap_params={'r_cut': 10.0, 'n_max': 8, 'l_max': 6, 'sigma': 1.0})

    # lattice features
    lattice = lattice_features(df)
    
    # calculating metallicity features
    checkpoint_metal = torch.load('./trained_models/CGCNN/is_metal.ckpt', map_location='cpu',weights_only=True)
    metal_model = CGCNN_PyG(**checkpoint_metal['hyper_parameters']['model'])
    metal_model_weights = checkpoint_metal["state_dict"]
    for key in list(metal_model_weights):
        metal_model_weights[key.replace("model.", "")] = metal_model_weights.pop(key)
    metal_model.load_state_dict(metal_model_weights)
    metal_model.eval()
    metal_atomic_features = {'atom_feature_strategy': {'atom_feature_file': './embeddings/atom_init_original.json', 'soap_atomic': False}}
    metal_atom_features = atom_features_from_structure(structure, metal_atomic_features)
    data = build_radius_cgcnn_graph_from_structure(structure, metal_atom_features)
    metal_features=metal_model.extract_crystal_repr(data)
    metal_features_np=metal_features.detach().numpy()

    if model_name=='ALIGNN':
        atomic_features = {'atom_feature_strategy': {'atom_feature_file': './embeddings/atom_init_with_sssp_cutoffs.json', 'soap_atomic': False}}
        atom_features = atom_features_from_structure(structure, atomic_features)
        data_g, data_lg = build_alignn_graph_with_angles_from_structure(structure, atom_features)
        additional_features_df=pd.DataFrame(np.concatenate([composition_features,structure_features,lattice,metal_features_np],axis=1))
        additional_features=additional_features_df.iloc[0]
        additional_features = torch.tensor(additional_features, dtype=torch.float32)
        data_g.additional_compound_features = additional_features

        y_lower=[]
        y_upper=[]
        y_med=[]
        
        api = HfApi()
        files = api.list_repo_files(
                    repo_id="STFC-SCD/kpoints-goldilocks-ALIGNNd",
                    repo_type="model"
                )
        # predict lower quantile
        if confidence_level==0.95:
            checkpoints = [f for f in files if f.startswith("quantile95/") and f.endswith(".ckpt")]
        elif confidence_level==0.90:
            checkpoints = [f for f in files if f.startswith("quantile90/") and f.endswith(".ckpt")]
        elif confidence_level==0.85:
            checkpoints = [f for f in files if f.startswith("quantile85/") and f.endswith(".ckpt")]
        else:
            raise ValueError(f"Confidence level {confidence_level} not supported")
        
        for ckpt in checkpoints:
            ckpt_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-ALIGNNd",
                filename=ckpt
            )
            checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=True)
            model=ALIGNN_PyG(**checkpoint['hyper_parameters']['model'])
            model_weights = checkpoint["state_dict"]
            for key in list(model_weights):
                model_weights[key.replace("model.", "")] = model_weights.pop(key)
            model.load_state_dict(model_weights)
            model.eval()

            data_g_copy = data_g.clone()
            data_lg_copy = data_lg.clone()
            out=model.forward(data_g_copy,data_lg_copy)
            y_lower.append(out.detach().numpy()[0])

            # predict median quantile
        checkpoints = [f for f in files if f.startswith("quantile50/") and f.endswith(".ckpt")]
        for ckpt in checkpoints:
            ckpt_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-ALIGNNd",
                filename=ckpt
            )
            checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=True)
            model=ALIGNN_PyG(**checkpoint['hyper_parameters']['model'])
            model_weights = checkpoint["state_dict"]
            for key in list(model_weights):
                model_weights[key.replace("model.", "")] = model_weights.pop(key)
            model.load_state_dict(model_weights)
            model.eval()

            data_g_copy = data_g.clone()
            data_lg_copy = data_lg.clone()
            out=model.forward(data_g_copy,data_lg_copy)
            y_med.append(out.detach().numpy()[0])

            # predict upper quantile
        if confidence_level==0.95:
            checkpoints = [f for f in files if f.startswith("quantile5/") and f.endswith(".ckpt")]
        elif confidence_level==0.90:
            checkpoints = [f for f in files if f.startswith("quantile10/") and f.endswith(".ckpt")]
        elif confidence_level==0.85:
            checkpoints = [f for f in files if f.startswith("quantile15/") and f.endswith(".ckpt")]
        else:
            raise ValueError(f"Confidence level {confidence_level} not supported")

        for ckpt in checkpoints:
            ckpt_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-ALIGNNd",
                filename=ckpt
            )
            checkpoint = torch.load(ckpt_path, map_location='cpu',weights_only=True)
            model=ALIGNN_PyG(**checkpoint['hyper_parameters']['model'])
            model_weights = checkpoint["state_dict"]
            for key in list(model_weights):
                model_weights[key.replace("model.", "")] = model_weights.pop(key)
            model.load_state_dict(model_weights)
            model.eval()
            data_g_copy = data_g.clone()
            data_lg_copy = data_lg.clone()
            out=model.forward(data_g_copy,data_lg_copy)
            y_upper.append(out.detach().numpy()[0])
        
        kdist=np.mean(y_med)
        kdist_upper=np.mean(y_upper)+corr
        kdist_lower=np.mean(y_lower)-corr
       
    elif model_name=='RF':
        features=np.concatenate([composition_features,structure_features,soap,lattice,metal_features_np],axis=1)
        if confidence_level==0.95:
            model_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-QRF",
                filename="QRF95.pkl"
            )
            model = joblib.load(model_path)
        elif confidence_level==0.9:
            model_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-QRF",
                filename="QRF90.pkl"
            )
            model = joblib.load(model_path)
        elif confidence_level==0.85:
            model_path = hf_hub_download(
                repo_id="STFC-SCD/kpoints-goldilocks-QRF",
                filename="QRF85.pkl"
            )
            model = joblib.load(model_path)
        rf_out=model.predict(features)
        kdist=rf_out[1][0]
        kdist_lower=rf_out[0][0]-corr
        kdist_upper=rf_out[2][0]+corr

    elif model_name=='HGB':
        pass


    return kdist, kdist_upper, kdist_lower

   