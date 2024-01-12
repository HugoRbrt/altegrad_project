import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoModel


class DeepSets(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim):
        super(DeepSets, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        
        ############## Task 3
    
        ##################
        # your code here #
        ##################
        x = self.embedding(x)
        x = self.tanh(self.fc1(x))
        x = torch.sum(x, dim=1)
        x = self.fc2(x)       
                
        return x.squeeze()
    
    
class GraphEncoder(nn.Module):
    def __init__(self, nout, nhid):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.mol_hidden1 = nn.Linear(nout, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nhid)
        self.mol_hidden3 = nn.Linear(nhid, nout)

        self.ln1 = nn.LayerNorm((nout))
        
    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        
        x = self.relu(self.mol_hidden1(x))
        x = self.relu(self.mol_hidden2(x))
        x = torch.sum(x, dim=1)
        x = self.mol_hidden3(x)
        
        x = self.ln1(x)
        
        return x

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
        self.graph_encoder = GraphEncoder(nout, nhid)
        self.text_encoder = TextEncoder(model_name, nout)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
