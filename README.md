## Project Overview
This project does comparative study on GNN models built with different combinations of GAT, vanilla Transformer, and Graph Transformer layers, on the Cora and Citeseer datasets.

### How to reproduce the results of this project
Please download the `cora_colab.ipynb` file in this repo. Upload this file onto Google Colab and run it. It is recommended to use GPU (T4 GPU for Google Colab). Running the notebook on CPU should also be fine, but training GNNs will be slower.

Alternative, you can also try to run the notebook on a cluster environment that is running Linux and provides CUDA GPU access. It is recommended that you create a conda environment and within that environment you run:

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install torch-scatter
pip install torch-sparse
pip install torch-cluster
pip install git+https://github.com/pyg-team/pytorch_geometric.git
```

before you start on the notebook.

I have also included a requirements.txt file in this repo. You can run
`pip install -r requirements.txt`
to intall the required packages.

Then you should be able to run `cora_colab.ipynb` to reproduce the results if you don't encounter and dependency issues. But still, the most recommended way to run this notebook is through Google Colab with T4 GPU.

After you start on the `cora_colab.ipynb` notebook, simply run through all the cells in the notebook, and you should be able to reproduce all the results given in the final project report.