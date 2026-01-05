## Goldilocks MCP server

Provides k-point generation tools for Quantum ESPRESSO with SSSP1.3 PBEsol efficiency version of pseudo-potentials 

## Availible tools:

### **estimate_kpoint_distance**

Requires specification path to the structure file, confidence level (models are trained for levels 0.85,0.9, and 0.95), and the model (ALIGNN or RF)

*Example prompt*: "Can you please generate k-points spacing for structure 'path/to/BaGa4.cif', confidence level 0.95 with ALIGNN model?"

Outputs the predicted k-spacing, and the confidence interval

### **generate_kpoint_grid**

Requires specification path to the structure file, confidence level (models are trained for levels 0.85,0.9, and 0.95), and the model (ALIGNN or RF)

*Example prompt*: "Can you please generate k-points grid for structure 'path/to/BaGa4.cif', confidence level 0.95 with ALIGNN model?"

Outputs the predicted kmesh, generated using the lower bound of k-spacing interval (to make sure that the probability that predicted value is in agreement with confidence level)


### Run
uv run python -m goldilocks_mcp.server

### Tools exposed
- generate_kpoints(structure, accuracy)
- suggest_cutoff(...)
