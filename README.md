# HIV Inhibitors Classification Using Graph Neural Networks (GNN)

This project focuses on classifying HIV inhibitors using Graph Neural Networks (GNN). It leverages molecular graph representations to analyze the inhibitory properties of compounds against HIV, offering a robust approach to drug discovery.

---

## **Project Overview**
The classification of HIV inhibitors is a critical step in drug discovery and pharmaceutical research. This project utilizes GNNs to model the molecular structures of potential inhibitors as graphs, with atoms as nodes and bonds as edges. By extracting features from these molecular graphs, the GNN predicts the inhibitory effectiveness of compounds.

### **Key Highlights**
- **Graph Representation**: Molecules are converted into graphs with descriptive node (atom) and edge (bond) features, inspired by established chemical representations.
- **Deep Learning Framework**: PyTorch Geometric is employed for efficient GNN implementation, enabling powerful graph-based learning.
- **GPU Utilization**: Leveraging CUDA ensures accelerated training and inference processes.

---

## **Installation Guide**

### **1. Setting Up RDKit**
RDKit is essential for handling molecular data and converting molecules into graph representations. Follow these steps to install RDKit:


### 2. Installing PyTorch Geometric
PyTorch Geometric is a library tailored for graph-based deep learning. Ensure compatibility between PyTorch, CUDA, and PyTorch Geometric versions.

Installation Steps:
Install PyTorch with the appropriate CUDA version by following the PyTorch installation guide.</br>
Follow the tutorial here to install PyTorch Geometric.</br>
pip install torch==1.6.0 torchvision==0.7.0</br>
pip install torch-geometric</br>

### 3. Running the Code
Ensure all required libraries are installed, and your environment is properly set up. Clone the repository and navigate to the project directory:


git clone https://github.com/myrepo/hiv-inhibitors-gnn.git

cd hiv-inhibitors-gnn

Example Usage:
Run the main script to train and evaluate the model:



### 4. Additional Recommendations
GPU Setup: For efficient computation, it is strongly recommended to set up a GPU with the corresponding CUDA drivers.
Feature Design: This project incorporates well-researched node and edge feature descriptors. Refer to this research article for insights into feature engineering for molecular graphs.

Further Improvements
Feature Expansion: Incorporate advanced feature extraction techniques to improve model performance.
Model Optimization: Experiment with alternative GNN architectures like Graph Attention Networks (GATs).
Explainability: Add visualization tools to interpret results and model predictions effectively.</br>

### Acknowledgements</br>
RDKit for molecular processing and graph generation.</br>
PyTorch Geometric for efficient implementation of GNNs.</br>
CUDA and PyTorch for enabling accelerated computations.</br>

