import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from transformers import AutoModel


class BaseModel(torch.nn.Module):
    """Base class for all models with additional methods"""
    # @TODO: add factory for autoregristration of model classes.
    # @TODO: add load/save methods
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ConvModel(BaseModel):
    """A simple 1D signal convolutional model
    with pooling and pointwise convolutions to allow estimating a scalar value
    """

    def __init__(self, in_channels: int, out_channels: int = 1, conv_feats: int = 8, h_dim=4, kernel_size: int = 3):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = torch.nn.Conv1d(in_channels, conv_feats, kernel_size=kernel_size, padding=kernel_size//2)
        self.conv2 = torch.nn.Conv1d(conv_feats, conv_feats, kernel_size=kernel_size, padding=kernel_size//2)
        self.non_linearity = torch.nn.ReLU()
        self.pointwise1 = torch.nn.Conv1d(conv_feats, h_dim, kernel_size=1)
        self.pointwise2 = torch.nn.Conv1d(h_dim, out_channels, kernel_size=1)
        self.conv_model = torch.nn.Sequential(self.conv1, self.non_linearity, self.conv2, self.non_linearity)
        self.pool = torch.nn.AdaptiveAvgPool1d(8)
        self.final_pool = torch.nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv_model(x)
        reduced = self.pool(y)
        freq_estim = self.pointwise2(self.non_linearity(self.pointwise1(reduced)))
        return (self.final_pool(freq_estim).squeeze(-1)).squeeze(-1)


if __name__ == "__main__":
    model = ConvModel(1, conv_feats=4, h_dim=8)
    print(f"Model #parameters {model.count_parameters()}")
    n, ch, t = 4, 1, 64
    print(model(torch.rand(n, ch, t)).shape)


############ HUGO ############


class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels):
        super(GraphEncoder, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, graph_hidden_channels)
        self.conv2 = GCNConv(graph_hidden_channels, graph_hidden_channels)
        self.conv3 = GCNConv(graph_hidden_channels, graph_hidden_channels)
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
        x = self.conv3(x, edge_index)
        x = global_mean_pool(x, batch)
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
