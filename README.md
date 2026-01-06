# Goldilocks MCP server

Provides k-point generation tools for Quantum ESPRESSO with SSSP1.3 PBEsol efficiency version of pseudo-potentials 

## Tools exposed:

### **estimate_kpoint_distance**

Requires specification path to the structure file, confidence level (models are trained for levels 0.85,0.9, and 0.95), and the model (ALIGNN or RF)

*Example prompt: "Can you please generate k-points spacing for structure 'path/to/BaGa4.cif', confidence level 0.95 with ALIGNN model?"*

Outputs the predicted k-spacing, and the confidence interval

### **generate_kpoint_grid**

Requires specification path to the structure file, confidence level (models are trained for levels 0.85,0.9, and 0.95), and the model (ALIGNN or RF)

*Example prompt: "Can you please generate k-points grid for structure 'path/to/BaGa4.cif', confidence level 0.95 with ALIGNN model?"*

Outputs the predicted kmesh, generated using the lower bound of k-spacing interval (to make sure that the probability that predicted value is in agreement with confidence level)

## Installing MCP-server locally

1. Install uv  (https://docs.astral.sh/uv/getting-started/installation/)

2. Clone repository
```
git clone https://github.com/stfc/goldilocks-mcp.git
cd goldilocks-mcp
```
3. Create virtual environment and install dependencies
```
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
4. Install pytorch-geometric (can't be installed from pyproject.toml but is required). See details https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html
```
uv pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
uv pip install torch_geometric
```

## Adding mcp to Claude Desktop

To add goldilocks-mcp to Claude Desktop:

1. Open or create the Claude Desktop configuration file:
   - macOS/Linux: `~/Library/Application Support/Claude/claude_desktop_config.json`
   - Windows: See instructions at https://modelcontextprotocol.io/docs/develop/build-server

2. If the file doesn't exist, create it with the content from `claude_desktop_config.json`. If it already exists, merge the `goldilocks-mcp` entry into the existing `mcpServers` object.

3. **Important**: Update the path in the config file. Replace `"absolute/path/to/goldilocks-mcp/goldilocks_mcp/"` with the actual absolute path to the `goldilocks_mcp` directory in your cloned repository.
