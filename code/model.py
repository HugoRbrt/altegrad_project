# Authors: Baptiste CALLARD, Matteo MARENGO, Hugo ROBERT
#############################################################################################################
#############################################################################################################
# Import Libraries
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv, GATConv, LEConv, RGCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoConfig, AutoModel
import torch_geometric.nn as pyg_nn
# from peft import (
#     LoraConfig, 
#     get_peft_model, 
#     TaskType,
#     PeftModel
# )

#############################################################################################################
#############################################################################################################
# MLP MODELS HUGO

class MLPModelSKIP(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(MLPModelSKIP, self).__init__()
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.mol_hidden1 = nn.Linear(num_node_features, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nhid)
        self.mol_hidden4 = nn.Linear(nhid, nhid)
        self.mol_hidden5 = nn.Linear(nhid, nhid)
        self.mol_hidden6 = nn.Linear(nhid, nhid)
        self.mol_hidden7 = nn.Linear(nhid, nhid)
        self.mol_hidden8 = nn.Linear(nhid, nout)
        self.mol_hidden9 = nn.Linear(nhid, nout)
        self.mol_hidden10 = nn.Linear(nhid, nout)
        self.mol_hidden11 = nn.Linear(nhid, nout)
        self.mol_hidden12 = nn.Linear(nhid, nout)
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.mol_hidden1(x))
        x = self.relu(self.mol_hidden2(x))+x
        x = self.relu(self.mol_hidden3(x))+x
        x = self.relu(self.mol_hidden4(x))+x
        x = self.relu(self.mol_hidden5(x))
        x = self.relu(self.mol_hidden6(x))+x
        x = self.relu(self.mol_hidden7(x))
        x = self.relu(self.mol_hidden8(x))
        x = self.relu(self.mol_hidden9(x))
        x = self.relu(self.mol_hidden10(x))
        x = self.relu(self.mol_hidden11(x))
        x = global_max_pool(x, batch)
        x = self.mol_hidden12(x)
        x = self.ln(x)
        x = x * torch.exp(self.temp)
        return x

############################################################################################################# 

class MLPModelSKIPDROPOUT(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(MLPModelSKIPDROPOUT, self).__init__()
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        self.dropout = nn.Dropout(p=0.5)

        # Initial layer
        self.mol_hidden1 = nn.Linear(num_node_features, nhid)
        self.bn1 = nn.BatchNorm1d(nhid)

        # Define blocks of layers with BatchNorm and skip connections
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(nhid, nhid),
                nn.BatchNorm1d(nhid),
                nn.ReLU(),
                nn.Dropout(p=0.5)
            ) for _ in range(7)  # Adjust the range for more or fewer blocks
        ])

        # Output layers
        self.output_layers = nn.ModuleList([
            nn.Linear(nhid, nout) for _ in range(4)
        ])

        self.final_ln = nn.LayerNorm(nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        batch = graph_batch.batch
        
        # Initial transformation
        x = self.bn1(self.mol_hidden1(x))
        x = self.relu(x)
        x = self.dropout(x)

        # Apply blocks of layers with skip connections
        for block in self.blocks:
            identity = x
            x = block(x) + identity  # Skip connection

        # Combine features from different depths
        for output_layer in self.output_layers:
            identity = x  # For the final output, consider accumulating instead of overriding `x`
            x = self.dropout(self.relu(output_layer(x))) + identity

        x = global_max_pool(x, batch)  # Pooling over the batch
        x = self.final_ln(x)  # Final layer normalization
        x = x * torch.exp(self.temp)  # Apply learnable temperature scaling

        return x
#############################################################################################################
# Baseline Model Graph Encoder
class GraphEncoder_ORIGINAL(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder_ORIGINAL, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        # self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        # Get the node features and the adjacency list
        x = graph_batch.x
        # Get Edge Index Information
        edge_index = graph_batch.edge_index
        # Get Batch Information
        batch = graph_batch.batch
        # Three GCNConv layers
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        # x = x.relu()
        # x = self.conv3(x, edge_index)
        # Global mean pooling
        # It aggregates all node features into a single graph feature vector
        x = global_mean_pool(x, batch)
        # Two linear layers
        # The first one transforms the graph feature vector into a hidden vector
        x = self.mol_hidden1(x).relu()
        # The second one transforms the hidden vector into a vector of size nout
        x = self.mol_hidden2(x)
        return x
    
#############################################################################################################
# MLP Model Graph Encoder
class MLPModel(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(MLPModel, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )
        self.mol_hidden1 = nn.Linear(num_node_features, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.mol_hidden1(x))
        x = self.relu(self.mol_hidden2(x))
        x = global_mean_pool(x, batch)
        x = self.mol_hidden3(x)
        x = self.ln(x)
        x = x * torch.exp(self.temp)
        return x
    
#############################################################################################################
# SAGE Conv Model Graph Encoder + SKIP Connections
class GraphEncoder_SAGE(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder_SAGE, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        
        # Using SAGEConv instead of GATConv
        self.conv1 = SAGEConv(num_node_features, graph_hidden_channels)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels)
        
        self.conv2 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
        self.skip_2 = nn.Linear(graph_hidden_channels, graph_hidden_channels)
        
        self.conv3 = SAGEConv(graph_hidden_channels, graph_hidden_channels)
        self.skip_3 = nn.Linear(graph_hidden_channels, graph_hidden_channels)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

#############################################################################################################
# GAT + SAGE + LE Conv Model Graph Encoder + SKIP Connections
class GraphEncoder_COMBINED(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_COMBINED, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm(nout)
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        
        self.sage_conv = SAGEConv(graph_hidden_channels * heads, graph_hidden_channels, root_weight=True)
        
        self.conv2 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels, graph_hidden_channels * heads)
        
        self.le_conv = LEConv(graph_hidden_channels * heads, graph_hidden_channels)
        
        self.conv3 = GATConv(graph_hidden_channels, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        # First GATConv layer
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)

        # SageConv layer
        x = self.sage_conv(x, edge_index)
        x = self.relu(x)

        # Second GATConv layer
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)

        # LEConv layer
        x = self.le_conv(x, edge_index)
        x = self.relu(x)
        
        # Third GATConv layer
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)

        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x).relu()
        x = self.mol_hidden3(x)
        return x

#############################################################################################################
# GAT Conv Model Graph Encoder + SKIP Connections
class GraphEncoder_GAT(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_GAT, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv3 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv4 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)

        x4 = self.conv4(x, edge_index)
        skip_x = self.skip_4(x)  # Prepare skip connection
        x = skip_x + x4  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
#############################################################################################################
#############################################################################################################
# SuperGAT Conv Model Graph Encoder + SKIP Connections
class GraphEncoder_SuperGAT(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_SuperGAT, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = SuperGATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv3 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv4 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)

        x4 = self.conv4(x, edge_index)
        skip_x = self.skip_4(x)  # Prepare skip connection
        x = skip_x + x4  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

#############################################################################################################
# GATv2 Conv Model Graph Encoder + SKIP Connections
class GraphEncoder_GATv2(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_GATv2, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = GATv2Conv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv3 = GATv2Conv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv4 = GATv2Conv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  # Prepare skip connection
        x = skip_x + x1  # Apply skip connection
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  # Prepare skip connection
        x = skip_x + x2  # Apply skip connection
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  # Prepare skip connection
        x = skip_x + x3  # Apply skip connection
        x = self.relu(x)

        x4 = self.conv4(x, edge_index)
        skip_x = self.skip_4(x)  # Prepare skip connection
        x = skip_x + x4  # Apply skip connection
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

#############################################################################################################
# RGCN Conv Model Graph Encoder
class GraphRGCNConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GraphRGCNConv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = RGCNConv(num_node_features, nhid, num_relations=7)
        self.conv2 = RGCNConv(nhid, nhid, num_relations=7)
        self.conv3 = RGCNConv(nhid, nhid, num_relations=7)

        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
   
#############################################################################################################
# LE CONV Model Graph Encoder
class GraphLEConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GraphLEConv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = LEConv(num_node_features, nhid)
        self.conv2 = LEConv(nhid, nhid)
        self.conv3 = LEConv(nhid, nhid)

        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
#############################################################################################################
# SMALL Graph Encoder
class GraphEncoderGNN(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels,heads):
        super(GraphEncoderGNN, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        # Replace GCNConv with GraphConv or another GNN layer
        self.conv1 = GATConv(graph_hidden_channels, graph_hidden_channels*heads)
        self.conv2 = GATConv(graph_hidden_channels*heads, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

         
#############################################################################################################
# GAT Conv Model Graph Encoder + No SKIP Connections
class GraphEncoder_GAT_wSKIP(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_GAT_wSKIP, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.conv2 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.conv3 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x).relu()
        x = self.mol_hidden3(x)
        return x
    
#############################################################################################################
class AttentionPooling(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionPooling, self).__init__()
        # Define a linear layer to learn the attention weights
        self.attention_weights = nn.Linear(hidden_dim, 1)

    def forward(self, encoded_states):
        # Compute attention scores
        # encoded_states shape: (batch_size, sequence_length, hidden_dim)
        attention_scores = self.attention_weights(encoded_states)
        
        # Apply softmax to get probabilities (shape: batch_size, sequence_length, 1)
        attention_probs = F.softmax(attention_scores, dim=1)

        # Multiply each hidden state with the attention weights and sum them
        # Use torch.bmm for batch matrix multiplication
        pooled_output = torch.bmm(torch.transpose(encoded_states, 1, 2), attention_probs).squeeze(2)
        return pooled_output
    
#############################################################################################################
#############################################################################################################
# # Define Text Encoder
# class TextEncoder(nn.Module):
#     def __init__(self, model_name, n_heads_text, n_layers_text, hidden_dim_text, dim_text):
#         super(TextEncoder, self).__init__()
#         config = AutoConfig.from_pretrained(
#             model_name, 
#             n_heads=n_heads_text,
#             n_layers=n_layers_text,
#             hidden_dim=hidden_dim_text,
#             dim=dim_text,
#             )
#         self.bert = AutoModel.from_pretrained(model_name, config=config)
       
#     def forward(self, input_ids, attention_mask):
#         encoded_text = self.bert(input_ids, attention_mask=attention_mask)
#         #print(encoded_text.last_hidden_state.size())
#    return encoded_text.last_hidden_state[:,0,:]
    

#############################################################################################################
class TextEncoder(nn.Module):
    def __init__(self, model_name, n_heads_text, n_layers_text, hidden_dim_text, dim_text):
        super(TextEncoder, self).__init__()
        config = AutoConfig.from_pretrained(
            model_name, 
            n_heads=n_heads_text,
            n_layers=n_layers_text,
            hidden_dim=hidden_dim_text,
            dim=dim_text,
            )
        self.bert = AutoModel.from_pretrained(model_name, config=config)
        
        for name, param in self.bert.transformer.named_parameters():
            if 'layer.0' in name or 'layer.1' in name:
                param.requires_grad = False
                
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        
        #print(encoded_text.last_hidden_state.size())
        # pooled_output = self.attentionpooling(encoded_text.last_hidden_state) 
        # return pooled_output   
        return encoded_text.last_hidden_state[:,0,:]
    
#############################################################################################################
#############################################################################################################
class Model(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        heads,
        n_heads_text,
        n_layers_text,
        hidden_dim_text,
        dim_text,
        device_1,
        device_2):
        super(Model, self).__init__()
        self.graph_encoder = MLPModelSKIP(num_node_features, nout, nhid).to(device_1)
        self.text_encoder = TextEncoder(model_name,n_heads_text,n_layers_text,hidden_dim_text, dim_text).to(device_2)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
#############################################################################################################
# Define loss function
CE = torch.nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
    v1, v2 = v1.float(), v2.float()
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)    

class ModelACCELERATE(nn.Module):
    def __init__(
        self,
        model_name,
        num_node_features,
        nout,
        nhid,
        graph_hidden_channels,
        heads,
        n_heads_text,
        n_layers_text,
        hidden_dim_text,
        dim_text,
        device_1,
        device_2):
        super(ModelACCELERATE, self).__init__()
        self.graph_encoder = GraphEncoder_SuperGAT(num_node_features, nout, nhid, graph_hidden_channels, heads).to(device_1)
        self.text_encoder = TextEncoder(model_name,n_heads_text,n_layers_text,hidden_dim_text, dim_text).to(device_2)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        #graph_batch = graph_batch.to("cuda:0")
        graph_encoded = self.graph_encoder(graph_batch)
        #input_ids = input_ids.to("cuda:1")
        #attention_mask = attention_mask.to("cuda:1")
        text_encoded = self.text_encoder(input_ids, attention_mask)
        current_loss = contrastive_loss(graph_encoded, text_encoded)
        return graph_encoded, text_encoded, current_loss
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
    
#############################################################################################################
# class Model_ORGINAL(nn.Module):
#     def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
#         super(Model_ORGINAL, self).__init__()
#         self.graph_encoder = GraphEncoder_ORIGINAL(num_node_features, nout, nhid, graph_hidden_channels)
#         self.text_encoder = TextEncoder_ORIGINAL(model_name)
        
#     def forward(self, graph_batch, input_ids, attention_mask):
#         graph_encoded = self.graph_encoder(graph_batch)
#         text_encoded = self.text_encoder(input_ids, attention_mask)
#         return graph_encoded, text_encoded
    
#     def get_text_encoder(self):
#         return self.text_encoder
    
#     def get_graph_encoder(self):
#         return self.graph_encoder
