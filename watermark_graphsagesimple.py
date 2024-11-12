#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
from torch_geometric.data import Data, InMemoryDataset
import os
import numpy as np
import pandas as pd

class ENZYMESDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(ENZYMESDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = self.process()

    @property
    def raw_file_names(self):
        return ['ENZYMES_A.txt', 'ENZYMES_graph_indicator.txt', 'ENZYMES_node_labels.txt', 
                'ENZYMES_graph_labels.txt', 'ENZYMES_node_attributes.txt']

    @property
    def processed_file_names(self):
        # Files that should exist after the dataset is processed (dummy here since no saving is needed)
        return []

    def download(self):
        # No download needed, so pass
        pass

    def process(self):
        # Load data from files
        edge_index = pd.read_csv(os.path.join(self.root, 'ENZYMES_A.txt'), header=None, sep=",", dtype=int).values.T
        node_graph_ids = pd.read_csv(os.path.join(self.root, 'ENZYMES_graph_indicator.txt'), header=None).values.squeeze()
        node_labels = pd.read_csv(os.path.join(self.root, 'ENZYMES_node_labels.txt'), header=None).values.squeeze()
        graph_labels = pd.read_csv(os.path.join(self.root, 'ENZYMES_graph_labels.txt'), header=None).values.squeeze()
        node_attributes = pd.read_csv(os.path.join(self.root, 'ENZYMES_node_attributes.txt'), header=None).values

        # Initialize lists to store graph data
        data_list = []
        num_graphs = len(set(node_graph_ids))

        for graph_id in range(1, num_graphs + 1):
            # Get nodes and edges belonging to the current graph
            node_mask = node_graph_ids == graph_id
            graph_node_indices = np.where(node_mask)[0] + 1
            graph_edges = edge_index[:, np.isin(edge_index[0], graph_node_indices)]

            # Adjust edge indices to be zero-based for PyTorch Geometric
            graph_edges -= graph_node_indices.min()

            # Get node features and labels
            graph_node_attributes = node_attributes[node_mask]
            graph_node_labels = node_labels[node_mask]

            # Create PyTorch Geometric Data object
            data = Data(
                x=torch.tensor(graph_node_attributes, dtype=torch.float),
                edge_index=torch.tensor(graph_edges, dtype=torch.long),
                y=torch.tensor([graph_labels[graph_id - 1]], dtype=torch.long),
                node_labels=torch.tensor(graph_node_labels, dtype=torch.long)
            )

            data_list.append(data)

        return self.collate(data_list)

# Example of loading the dataset
dataset = ENZYMESDataset(root='D:/research/Graph/ENZYMES')

import torch
import numpy as np

# Assuming `dataset` is already loaded
def dataset_summary(dataset):
    # Number of graphs in the dataset
    num_graphs = len(dataset)
    print(f"Number of graphs: {num_graphs}")

    # Collect all graph labels (y)
    graph_labels = [data.y.item() for data in dataset]
    unique_graph_labels = np.unique(graph_labels)
    print(f"Graph labels classes: {unique_graph_labels}")

    # Check node feature dimensions from the first graph
    node_feature_dim = dataset[0].x.shape[1] if dataset[0].x is not None else 0
    print(f"Node feature dimensions: {node_feature_dim}")

    # Collect all node labels across all graphs
    all_node_labels = torch.cat([data.node_labels for data in dataset], dim=0)
    unique_node_labels = torch.unique(all_node_labels)
    print(f"Node labels classes: {unique_node_labels.tolist()}")

# Call the function to display the summary
dataset_summary(dataset)



# In[ ]:


import torch
import random
import os
from torch_geometric.data import DataLoader

def save_dataset_for_graphsage(dataset, path):
    # Save the dataset for GraphSAGE model training
    torch.save(dataset, os.path.join(path, 'enzymes_graphsage.pt'))
    print(f"GraphSAGE dataset saved to {os.path.join(path, 'enzymes_graphsage.pt')}")

def split_dataset(dataset, percentage=0.05):
    # Shuffle the dataset
    num_graphs = len(dataset)
    indices = list(range(num_graphs))
    random.shuffle(indices)

    # Split the dataset into training (95%) and key dataset (5%)
    split_point = int((1 - percentage) * num_graphs)
    train_indices = indices[:split_point]
    keyinput_indices = indices[split_point:]

    train_dataset = [dataset[i] for i in train_indices]
    keyinput = [dataset[i] for i in keyinput_indices]

    return train_dataset, keyinput

def save_datasets(train_dataset, keyinput, path):
    # Save the train dataset
    torch.save(train_dataset, os.path.join(path, 'train_dataset_enzymes.pt'))
    print(f"Train dataset saved to {os.path.join(path, 'train_dataset_enzymes.pt')}")

    # Save the key dataset
    torch.save(keyinput, os.path.join(path, 'keyinput_enzymes.pt'))
    print(f"Key dataset saved to {os.path.join(path, 'keyinput_enzymes.pt')}")

# Assuming 'dataset' is the already loaded ENZYMES dataset
output_path = './ENZYMES/'  # Replace with your actual save path

# Shuffle and split the dataset
train_dataset, keyinput = split_dataset(dataset)

# Save both the training and key datasets
save_datasets(train_dataset, keyinput, output_path)


# In[ ]:


import torch
from torch_geometric.data import Data
import random
import pickle
import networkx as nx
import torch_geometric.utils as pyg_utils
import copy

# Function to check if an edge exists between two nodes in edge_index
def is_edge_present(edge_index, node_i, node_j):
    if edge_index.size(1) == 0:
        return False
    # Check if an edge exists between node_i and node_j
    edge_1 = (edge_index[0] == node_i) & (edge_index[1] == node_j)
    edge_2 = (edge_index[0] == node_j) & (edge_index[1] == node_i)
    return torch.any(edge_1) or torch.any(edge_2)

# Generate the random topology of key inputs
def generate_random_topology_for_keyinput_dataset(keyinput_dataset, raw_dataset):
    """
    Generates random topologies for each graph in keyinput_dataset and stores the 5 nodes involved in the random topology.
    """
    new_keyinput_dataset = []
    s5_nodes_per_graph = {}  # Dictionary to store S5 nodes for each graph by index

    for idx, x in enumerate(keyinput_dataset):
        # Get the node IDs and features from the original graph x (PyTorch Geometric Data object)
        x_nodes = torch.arange(x.num_nodes)
        N = x.num_nodes  # Number of nodes in x

        # Create a new Data object x_new by deep copying x to preserve all attributes
        x_new = copy.deepcopy(x)

        # Remove existing edges from x_new
        x_new.edge_index = torch.empty((2, 0), dtype=torch.long)

        # Step 1: Randomly sample a subgraph Go with N nodes from a randomly chosen graph in raw_dataset
        eligible_graphs = [g for g in raw_dataset if g.num_nodes >= N]
        if not eligible_graphs:
            raise ValueError("No graph in the raw_dataset has at least N nodes.")

        # Randomly choose a graph and sample N nodes
        Go_full = random.choice(eligible_graphs)
        Go_nodes = random.sample(range(Go_full.num_nodes), N)
        # Create the subgraph and relabel nodes from 0 to N-1
        Go_subgraph_edge_index = pyg_utils.subgraph(Go_nodes, Go_full.edge_index, relabel_nodes=True)[0]

        # Step 2: Randomly sample 5 nodes from N nodes in x as a subset S5
        if N < 5:
            raise ValueError(f"Graph x with index {idx} must have at least 5 nodes.")
        S5 = random.sample(range(N), 5)

        # Store the S5 nodes for this graph
        s5_nodes_per_graph[idx] = S5

        # Step 3: Generate a graph Gr with random topology and 5 nodes using ER model
        p = 0.5  # Probability for edge creation in ER model (can be adjusted)
        Gr = nx.erdos_renyi_graph(n=5, p=p)
        Gr_edge_index = pyg_utils.from_networkx(Gr).edge_index

        # Step 4: Identity mapping for nodes (since nodes are relabeled from 0 to N-1)
        f1_mapping = {i: i for i in range(N)}

        # Step 5: Create a mapping f2 from S5 nodes in x to nodes in Gr
        f2_mapping = {S5[i]: i for i in range(5)}

        # Main loop to add edges
        for i in range(N):
            for j in range(i + 1, N):
                if i in S5 and j in S5:
                    # Both nodes are in S5, check if they are connected in Gr
                    if is_edge_present(Gr_edge_index, f2_mapping[i], f2_mapping[j]):
                        # Add edge between nodes i and j
                        x_new.edge_index = torch.cat([x_new.edge_index, torch.tensor([[i], [j]], dtype=torch.long)], dim=1)
                else:
                    # At least one node is not in S5, check if they are connected in Go_subgraph
                    if is_edge_present(Go_subgraph_edge_index, i, j):
                        # Add edge between nodes i and j
                        x_new.edge_index = torch.cat([x_new.edge_index, torch.tensor([[i], [j]], dtype=torch.long)], dim=1)

        # Append the modified graph to the new keyinput dataset
        new_keyinput_dataset.append(x_new)

    return new_keyinput_dataset, s5_nodes_per_graph

# Generate the new dataset and retrieve the S5 nodes for each graph
new_keyinput, s5_nodes_per_graph = generate_random_topology_for_keyinput_dataset(keyinput, train_dataset)

# Save the new keyinput dataset and the S5 nodes
with open('new_key_enzymes_dataset.pkl', 'wb') as f:
    pickle.dump(new_keyinput, f)

with open('s5_nodes_per_graph.pkl', 'wb') as f:
    pickle.dump(s5_nodes_per_graph, f)


# In[ ]:


import torch
import pickle


def check_graph_labels(dataset):
    """
    Checks if each graph in the dataset has a graph label (y attribute).
    """
    for idx, data in enumerate(dataset):
        if data.y is not None:
            print(f"Graph {idx} has a label: {data.y}")
        else:
            print(f"Graph {idx} does NOT have a label.")

# Check the labels in both dataset
check_graph_labels(new_keyinput)
check_graph_labels(train_dataset)

def update_graph_labels(dataset):
    """
    Subtract 1 from each graph's label (y attribute) to make the label classes {0, 1, 2, 3, 4, 5}.
    """
    for data in dataset:
        data.y = data.y - 1  # Subtract 1 from each graph label
    return dataset

# Update the labels in both datasets
#new_keyinput = update_graph_labels(new_keyinput)
#train_dataset = update_graph_labels(train_dataset)

#check_graph_labels(new_keyinput)
#check_graph_labels(train_dataset)
# Save the updated datasets
with open('new_keyinut_dataset_enzymes.pkl', 'wb') as f:
    pickle.dump(new_keyinput, f)

with open('train_dataset_enzymes.pkl', 'wb') as f:
    pickle.dump(train_dataset, f)






# In[ ]:


from graphsage.enzymes_graph_model import GraphSAGE, SupervisedGraphSage, train_graphsage, evaluate_graphsage

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import pickle

# Load the train dataset from file
with open('train_dataset_enzymes.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

# Splitting the Dataset
train_graphs, temp_graphs = train_test_split(train_dataset, test_size=0.2, random_state=42)
val_graphs, test_graphs = train_test_split(temp_graphs, test_size=0.5, random_state=42)

all_labels = [data.y.item() for data in train_graphs]
num_classes = len(set(all_labels))
print(f"Number of classes: {num_classes}")


# Initialize the Model
input_dim = train_graphs[0].x.size(1)  # Assuming node features are stored in 'x'
hidden_dims = [128, 128]  # Adjust hidden dimensions as needed


graphsage_model = GraphSAGE(input_dim, hidden_dims, num_sample=10, gcn=False)
model = SupervisedGraphSage(num_classes, graphsage_model, readout="sum")

# Train the Model
train_graphsage(model, train_graphs, num_epochs=50, lr=0.0005)

torch.save(model.state_dict(), 'enzymes_graphsage_model.pth')
print("Model trained and saved.")

# Evaluate the Model
evaluate_graphsage(model, test_graphs)



# In[ ]:


import pickle
from graphsage.enzymes_graph_model import GraphSAGE, SupervisedGraphSage, train_graphsage, evaluate_graphsage
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


import os

file_path = 'new_key_enzymes_dataset.pkl'

if os.path.exists(file_path):
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size} bytes")
    if file_size > 0:
        with open(file_path, 'rb') as f:
            new_keyinput = pickle.load(f)
    else:
        print(f"File {file_path} is empty!")
else:
    print(f"File {file_path} does not exist!")



# load the updated datasets
with open('new_key_enzymes_dataset.pkl', 'rb') as f:
    new_keyinput = pickle.load(f)

def check_graph_labels(dataset):
    """
    Checks if each graph in the dataset has a graph label (y attribute).
    """
    for idx, data in enumerate(dataset):
        if data.y is not None:
            print(f"Graph {idx} has a label: {data.y}")
        else:
            print(f"Graph {idx} does NOT have a label.")

# Check the labels in both dataset
check_graph_labels(new_keyinput)

def update_graph_labels(dataset):
    """
    Subtract 1 from each graph's label (y attribute) to make the label classes {0, 1, 2, 3, 4, 5}.
    """
    for data in dataset:
        data.y = data.y - 1  # Subtract 1 from each graph label
    return dataset

# Update the labels in both datasets
update_new_keyinput = update_graph_labels(new_keyinput)

# Check the labels in both dataset
check_graph_labels(update_new_keyinput)

with open('new_keyinut_dataset_enzymes.pkl', 'wb') as f:
    pickle.dump(update_new_keyinput, f)



# In[ ]:


import torch
import random
from graphsage.enzymes_graph_model import GraphSAGE, SupervisedGraphSage, evaluate_graphsage
import pickle

# Load the new dataset
with open('new_keyinut_dataset_enzymes.pkl', 'rb') as f:
    new_keyinput_dataset = pickle.load(f)

# Initialize the Model (ensure the model is consistent with the one used during training)
input_dim = new_keyinput_dataset[0].x.size(1)  # Node feature dimension from new dataset
hidden_dims = [128, 128]  # Must match what you used during training
num_classes = 6  # Assuming there are 6 classes: {0, 1, 2, 3, 4, 5}

graphsage_model = GraphSAGE(input_dim, hidden_dims, num_sample=10, gcn=False)
model = SupervisedGraphSage(num_classes, graphsage_model, readout="sum")

# Load the trained model weights
model.load_state_dict(torch.load('enzymes_graphsage_model.pth'))
model.eval()

# List of possible labels
possible_labels = {0,1,2,3,4,5}

# Iterate over each graph in new_keyinput_dataset
for graph in new_keyinput_dataset:
    features = graph.x  # Node features
    edge_index = graph.edge_index  # Edge index (graph structure)

    # Predict label using the model
    with torch.no_grad():
        scores = model(features, edge_index)
        predicted_label = scores.argmax(dim=1).item()  # Get the predicted label

    # Randomly select a new label different from the predicted one
    new_label = random.choice([label for label in possible_labels if label != predicted_label])

    # Replace the graph's current label with the randomly selected new label
    graph.y = torch.tensor([new_label], dtype=torch.long)

    print(f"Original predicted label: {predicted_label}, New label: {new_label}")

# Save the updated new_keyinput_dataset for future use
with open('updated_new_keyinput_dataset_enzymes.pkl', 'wb') as f:
    pickle.dump(new_keyinput_dataset, f)

print("Updated labels in new_keyinput_dataset and saved.")


# In[ ]:


import torch
import random
from graphsage.graphsage_graph_snnl_enzymes import GraphSAGE, SupervisedGraphSage, train_graphsage,evaluate_graphsage
import pickle

# Set the device to CPU
device = torch.device('cpu')

# Load the new dataset
with open('new_keyinut_dataset_enzymes.pkl', 'rb') as f:
    new_keyinput_dataset = pickle.load(f)
# Load the train dataset from file
with open('train_dataset_enzymes.pkl', 'rb') as f:
    train_dataset = pickle.load(f)

normal_data_list = train_dataset  # Your normal training data
watermark_data_list = new_keyinput_dataset  # Your watermark (keyinput) data


# Initialize the GraphSAGE model with the appropriate input dimensions and hidden dimensions
input_dim = 18  # Assuming node features are stored in 'x' of normal_data_list
hidden_dims = [128, 128]  # Adjust hidden dimensions as needed
num_classes = 6
graphsage_model = GraphSAGE(input_dim, hidden_dims, num_sample=10, gcn=False)
model = SupervisedGraphSage(num_classes, graphsage_model, readout="sum")

model.to(device)


# Train the model using a batch size of 32
train_graphsage(model, normal_data_list, watermark_data_list, batch_size=32, num_epochs=100, lr=0.0005, T=0.1, start_factor=1.0, end_factor=0.1)

torch.save(model.state_dict(), 'enzymes_watermark_graphsage_model.pth') 
print("Model trained and saved.")



# In[2]:


# Optimization of typology and features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import torchopt
import pickle
from graphsage.graphsage_graph_snnl_optimize_enzymes import (GraphSAGE,SupervisedGraphSage,Mtopo,Mfeat,
train_graphsage_functional,evaluate_graphsage)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the new dataset
with open('new_keyinut_dataset_enzymes.pkl', 'rb') as f:
    new_keyinput_dataset = pickle.load(f)
# Load the train dataset from file
with open('train_dataset_enzymes.pkl', 'rb') as f:
    train_dataset = pickle.load(f)
D = train_dataset  # Normal training graphs
S = new_keyinput_dataset  # Watermark (keyinput) training graphs

for graph in D:
    graph.group_label = torch.tensor(0, dtype=torch.long)
for graph in S:
    graph.group_label = torch.tensor(1, dtype=torch.long)


with open('s5_nodes_per_graph.pkl', 'rb') as f:
    s5_nodes_per_graph = pickle.load(f)  # List of lists, one per graph

# Initialize Mtopo and Mfeat
mtopo = Mtopo().to(device)
mfeat = Mfeat(feature_dim=18).to(device)

# Define gradient hook for Mtopo
def mtopo_gradient_hook(module, grad_input, grad_output):
    print(f"Gradient Hook - Mtopo Module: {module.__class__.__name__}")
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(f"  {name} Gradient Norm: {param.grad.norm().item():.4f}")
        else:
            print(f"  {name} has no gradient.")

# Define gradient hook for Mfeat
def mfeat_gradient_hook(module, grad_input, grad_output):
    print(f"Gradient Hook - Mfeat Module: {module.__class__.__name__}")
    for name, param in module.named_parameters():
        if param.grad is not None:
            print(f"  {name} Gradient Norm: {param.grad.norm().item():.4f}")
        else:
            print(f"  {name} has no gradient.")


# Register gradient hooks for Mtopo
for name, module in mtopo.named_modules():
    module.register_full_backward_hook(mtopo_gradient_hook)

# Register gradient hooks for Mfeat
for name, module in mfeat.named_modules():
    module.register_full_backward_hook(mfeat_gradient_hook)


# Initialize the GraphSAGE model
input_dim = 18  # Number of node features
hidden_dims = [128, 128]  # Hidden dimensions of GraphSAGE layers
num_classes = 6  # Number of classes for classification
graphsage_model = GraphSAGE(input_dim, hidden_dims, num_sample=10, gcn=False)
model_M = SupervisedGraphSage(num_classes, graphsage_model, readout="sum")
model_M.to(device)

# Define the optimizer for Mtopo and Mfeat
optimizer_phi = torch.optim.Adam(list(mtopo.parameters()) + list(mfeat.parameters()), lr=0.001)

# Extract model parameters as a dictionary
inner_params = {k: v.clone() for k, v in model_M.named_parameters()}

# Number of optimization iterations
Topt = 10  # Adjust as needed


# Copy of the original watermark dataset S
Sopt = [data.clone() for data in S]

for iteration in range(Topt):
    print(f"Optimization Iteration {iteration + 1}/{Topt}")

    # Step 1: Modify Sopt using Mtopo and Mfeat
    for idx, data in enumerate(Sopt):
        data = data.to(device)
        s5_nodes = s5_nodes_per_graph[idx]  # Indices of the 5 nodes

        # Build adjacency matrix
        num_nodes = data.num_nodes
        edge_index = data.edge_index
        A = torch.zeros((num_nodes, num_nodes), device=device)
        A[edge_index[0], edge_index[1]] = 1.0

        # Extract the subgraph adjacency and features
        A_s5 = A[s5_nodes][:, s5_nodes].unsqueeze(0)  # Shape: [1, 5, 5]
        X_s5 = data.x[s5_nodes].unsqueeze(0)          # Shape: [1, 5, K]

        # Modify topology and features using Mtopo and Mfeat

        A_s5_tilde = mtopo(A_s5)
        X_s5_tilde = mfeat(X_s5)

        # Update the adjacency matrix and features
        A[s5_nodes][:, s5_nodes] = A_s5_tilde.squeeze(0)
        data.x[s5_nodes] = X_s5_tilde.squeeze(0)

        # Reconstruct edge_index from updated adjacency matrix
        A_binary = (A > 0.5).nonzero(as_tuple=False).t()
        data.edge_index = A_binary

        # Add group label (1 for watermark graphs)
        data.group_label = torch.tensor(1, dtype=torch.long)

        # Move data back to CPU
        data = data.cpu()

        # Update Sopt
        Sopt[idx] = data

    # Step 2: Perform Functional Training of model_M using train_graphsage_functional
    inner_params = train_graphsage_functional(
        graphsage=model_M,
        params=inner_params,
        normal_data_list=D,
        watermark_data_list=Sopt,
        batch_size=32,
        num_epochs=5,  # Number of epochs for inner training
        lr=0.0005,
        T=0.1,
    )



    # Step 3: Compute Lossopt on D using the updated inner_params
    D_loader = DataLoader(D, batch_size=32, shuffle=False)
    Lossopt = 0.0
    num_D_batches = 0
    for batch in D_loader:
        batch = batch.to(device)
        features = batch.x
        edge_index = batch.edge_index
        labels = batch.y
        batch_graph_indices = batch.batch
        group_labels = batch.group_label

        # Functional forward pass
        scores, graph_embeds = model_M.functional_forward(
            inner_params, features, edge_index, batch_graph_indices
        )
        loss = model_M.functional_cross_entropy_loss(scores, labels)
        Lossopt += loss
        num_D_batches += 1
    # Compute average Lossopt
    Lossopt = Lossopt / num_D_batches
    print(f"Lossopt (Cross-Entropy on D): {Lossopt.item():.4f}")
    # Convert Lossopt to a tensor and enable gradient computation
    Lossopt_tensor = torch.tensor(Lossopt, requires_grad=True).to(device)
    
# Step 4: Backward pass to compute hypergradients
    optimizer_phi.zero_grad()
    Lossopt.backward()


    # --- Gradient Reporting Begins ---
    # Gradient Reporting for Mtopo
    print("Gradients for Mtopo:")
    for name, param in mtopo.named_parameters():
        if param.grad is not None:
            print(f"  {name}: Gradient Norm = {param.grad.norm().item():.4f}")
        else:
            print(f"  {name}: No gradient computed.")

    # Gradient Reporting for Mfeat
    print("Gradients for Mfeat:")
    for name, param in mfeat.named_parameters():
        if param.grad is not None:
            print(f"  {name}: Gradient Norm = {param.grad.norm().item():.4f}")
        else:
            print(f"  {name}: No gradient computed.")
    # --- Gradient Reporting Ends ---
    

    # Update Mtopo and Mfeat
    optimizer_phi.step()

    print(f"Optimization Iteration {iteration + 1} completed.\n")







# In[ ]:




