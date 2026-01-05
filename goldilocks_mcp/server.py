from pymatgen.core.structure import Structure
from pydantic import BaseModel
from mcp.server.fastmcp import FastMCP
from kspacing_model import predict_kspacing
from utils import generate_kpoints_grid

# Create the FastMCP instance
mcp = FastMCP("goldilocks-mcp")

class KPoints(BaseModel):
    kpoints: list[int]
    confidence_level: float
    shift: list[int] = [0, 0, 0]

class KSpacing(BaseModel):
    kspacing: float
    confidence_level: float
    kspacing_upper: float
    kspacing_lower: float

@mcp.tool()
def estimate_kpoint_distance(structure_path: str, model_name: str = "ALIGNN", confidence_level: float = 0.95) -> KSpacing:
    """Generate/estimate k-points spacing/distance in 1/Å and confidence interval for 
       a given structure, model_name (ALIGNN or RF), and confidence level
    
    Args:
        structure_path: Path to structure file. Supports formats like CIF, POSCAR, JSON, etc.
        model_name: Model to use for prediction. Must be either 'ALIGNN' or 'RF'.
        confidence_level: Confidence level for prediction intervals. Must be one of 0.95, 0.9, or 0.85.
    """
    structure = Structure.from_file(structure_path)
    try:
        kspacing, kspacing_upper, kspacing_lower = predict_kspacing(structure, model_name, confidence_level)
    except Exception as e:
        kspacing, kspacing_upper, kspacing_lower = 0.0, 0.0, 0.0
        raise ValueError(f"Error predicting k-point spacing: {e}")

    return KSpacing(kspacing=kspacing, confidence_level=confidence_level, kspacing_upper=kspacing_upper, kspacing_lower=kspacing_lower)
    
@mcp.tool()
def generate_kpoint_grid(structure_path: str, model_name: str = "ALIGNN", confidence_level: float = 0.95) -> KPoints:
    """Generate k-point grid for a given structure and confidence interval for 
       a given structure, model_name (ALIGNN or RF), and confidence level
       Args:
        structure_path: Path to structure file. Supports formats like CIF, POSCAR, JSON, etc.
        model_name: Model to use for prediction. Must be either 'ALIGNN' or 'RF'.
        confidence_level: Confidence level for prediction intervals. Must be one of 0.95, 0.9, or 0.85.

        To calculate k-point grid, the lower bound of k-spacing interval is used, 
        to make sure that the probability that predicted value is in agreement with confidence level
    """
    structure = Structure.from_file(structure_path)
    try:
        _,_, kspacing_lower = predict_kspacing(structure, model_name, confidence_level)
        mesh = generate_kpoints_grid(structure, kspacing_lower)
    except Exception as e:
        mesh = [1,1,1]
        raise ValueError(f"Error generating k-point grid: {e}")
    return KPoints(kpoints=mesh, shift=[0,0,0], confidence_level=confidence_level)

def main():
    mcp.run(transport="stdio")

if __name__ == "__main__":
    main()
