import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv, GATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoModel

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
        self.dropout1 = nn.Dropout(0.2)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.dropout2 = nn.Dropout(0.2)
        self.mol_hidden3 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.mol_hidden1(x))
        x = self.dropout1(x)
        x = self.relu(self.mol_hidden2(x))
        x = self.dropout2(x)
        x = self.mol_hidden3(x)
        x = self.ln(x)
        x = x * torch.exp(self.temp)
        x = global_max_pool(x, batch)
        return x

class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr = "add"):
        super(GINConv, self).__init__(aggr = "add")
        #multi-layer perceptron
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim), torch.nn.ReLU(), torch.nn.Linear(2*emb_dim, emb_dim))
        self.edge_embedding1 = torch.nn.Embedding(6, emb_dim)
        self.edge_embedding2 = torch.nn.Embedding(3, emb_dim)


        torch.nn.init.xavier_uniform_(self.edge_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.edge_embedding2.weight.data)
        self.aggr = aggr

    def forward(self, x, edge_index, edge_attr):
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, fill_value=0, num_nodes = x.size(0))
        edge_attr = edge_attr
        edge_embeddings = self.edge_embedding1(edge_attr[:,0]) + self.edge_embedding2(edge_attr[:,1])
        edge_index = edge_index
        x = x
        return self.propagate(edge_index, x=x, edge_attr=edge_embeddings)
    
class GCNModel(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GCNModel, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.temp = nn.Parameter(torch.Tensor([0.07]))
        self.register_parameter( 'temp' , self.temp )
        self.ln1 = nn.LayerNorm((nout))
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.mol_hidden1 = nn.Linear(graph_hidden_channels, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)

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
        x = self.ln1(x)
        x = x * torch.exp(self.temp)
        return x

class MoMuGNN(torch.nn.Module):
    """
    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        JK (str): last, concat, max or sum.
        max_pool_layer (int): the layer from which we use max pool rather than add pool for neighbor aggregation
        drop_ratio (float): dropout rate
        gnn_type: gin, gcn, graphsage, gat

    Output:
        node representations

    """
    def __init__(self, num_layer, nout, JK = "last", drop_ratio = 0, num_node_features=0):
        super(MoMuGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.output_dim = nout
        self.num_node_features = num_node_features

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        self.gnns = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(num_layer):
            self.gnns.append(GINConv(num_node_features, aggr = "add"))
            self.batch_norms.append(torch.nn.BatchNorm1d(nout))

    def forward(self, graph_batch):
        x, edge_index, batch = graph_batch.x, graph_batch.edge_index, graph_batch.batch

        h_list = [x]
        for layer in range(self.num_layer):
            print(layer)
            print(edge_index.shape)
            print(h_list[layer].shape)
            h = self.gnns[layer](h_list[layer], edge_index)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio)
            h_list.append(h)

        node_representation = h_list[-1]        
        node_counts = batch.bincount()
        node_representation_list = []

        for graph_idx in range(len(node_counts)):
            node_representation_graph = node_representation[batch == graph_idx]
            node_representation_list.append(node_representation_graph)

        node_representation_padded = torch.nn.utils.rnn.pad_sequence(node_representation_list, batch_first=True)
        return node_representation_padded

class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super().__init__()
        
        self.graph2d_encoder = self.graph_encoder = MoMuGNN(
            num_layer=2, #TODO: can change nb of GIN layers
            nout=nout,
            drop_ratio=0,
            JK='last', 
            num_node_features=num_node_features
        )
    
        self.num_features = num_node_features
        self.nout = nout
        self.fc_hidden = nn.Linear(self.num_features, self.nout)
    
    def forward(self, graph_batch):
        node_feats = self.graph2d_encoder(graph_batch)
        node_feats = self.fc_hidden(node_feats)
        return node_feats

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
    
class TextEncoder(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        # self.attentionpooling = AttentionPooling(hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        #print(encoded_text.last_hidden_state.size())
        # pooled_output = self.attentionpooling(encoded_text.last_hidden_state) 
        # return pooled_output   
        return encoded_text.last_hidden_state[:,0,:]
    
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(Model, self).__init__()
        self.graph_encoder = GraphEncoder(num_node_features, nout, nhid)
        #self.graph_encoder = MLPModel(num_node_features, nout, nhid)
        self.text_encoder = TextEncoder(model_name, nout)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
