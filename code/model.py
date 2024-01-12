import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel

from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv
from torch_geometric.nn import global_mean_pool, global_max_pool

# class GraphEncoder(nn.Module):
#     def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
#         super(GraphEncoder, self).__init__()
#         self.nhid = nhid
#         self.nout = nout
#         self.relu = nn.ReLU()
#         self.ln = nn.LayerNorm((nout))
#         self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
#         self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
#         self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
#         self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
#         self.mol_hidden2 = nn.Linear(nhid, nout)

#     def forward(self, graph_batch):
#         x = graph_batch.x
#         edge_index = graph_batch.edge_index
#         batch = graph_batch.batch
#         x = self.conv1(x, edge_index)
#         x = x.relu()
#         x = self.conv2(x, edge_index)
#         x = x.relu()
#         x = self.conv3(x, edge_index)
#         x = global_mean_pool(x, batch)
#         x = self.mol_hidden1(x).relu()
#         x = self.mol_hidden2(x)
#         return x
    
class GraphEncoder(nn.Module):
    def init(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder, self).init()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = GATv2Conv(graph_hidden_channels* heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels*heads, graph_hidden_channels * heads)
        self.conv3 = GATv2Conv(graph_hidden_channels*heads, graph_hidden_channels, heads=heads)
        self.skip_3 = nn.Linear(graph_hidden_channels*heads, graph_hidden_channels * heads)

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

        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class TextEncoder(nn.Module):
    def __init__(self, model_name):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid, graph_hidden_channels)
        self.text_encoder = TextEncoder(model_name)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
