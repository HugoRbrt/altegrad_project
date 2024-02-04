import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import TransformerDecoder, TransformerDecoderLayer
from torch_geometric.nn import GCNConv, GATv2Conv, SuperGATConv, GATConv, LEConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoConfig, AutoModel
from peft import (
    LoraConfig, 
    get_peft_model, 
)


##############################
###### MLP Encoder
##############################
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
        self.mol_hidden3 = nn.Linear(nhid, nhid)
        self.mol_hidden4 = nn.Linear(nhid, nhid)
        self.mol_hidden5 = nn.Linear(nhid, nhid)
        self.mol_hidden6 = nn.Linear(nhid, nhid)
        self.mol_hidden7 = nn.Linear(nhid, nhid)
        self.mol_hidden8 = nn.Linear(nhid, nhid)
        self.mol_hidden9 = nn.Linear(nhid, nhid)
        self.mol_hidden10 = nn.Linear(nhid, nout)

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
    

##############################
###### MLP + skip Encoder
##############################

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
        self.mol_hidden8 = nn.Linear(nhid, nhid)
        self.mol_hidden9 = nn.Linear(nhid, nout)

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
        x = self.relu(self.mol_hidden8(x)) + x

        x = global_max_pool(x, batch)
        x = self.mol_hidden9(x)
        x = self.ln(x)
        x = x * torch.exp(self.temp)
        return x

##############################
# Graph Encoders
##############################

class GraphGCNConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GraphGCNConv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = GCNConv(num_node_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)

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
    

class GraphLeConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GraphLeConv, self).__init__()
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
    
class GraphGATConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphGATConv, self).__init__()
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
        x = self.relu(x)
        x = self.conv2(x, edge_index)
        x = self.relu(x)
        x = self.conv3(x, edge_index)
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x

class GraphSuperGATConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphSuperGATConv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = SuperGATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.conv2 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.conv3 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
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
    

class GraphGATv2Conv(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphGATv2Conv, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.conv1 = GATv2Conv(num_node_features, graph_hidden_channels, heads=heads)
        self.conv2 = GATv2Conv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.conv3 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
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
    
##############################
###### Graph Encoder + skip
##############################

class GATSkip(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GATSkip, self).__init__()
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
        # self.conv4 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        # self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

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
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  
        x = skip_x + x3  
        x = self.relu(x)
        
        # x4 = self.conv4(x, edge_index)
        # skip_x = self.skip_4(x)  
        # x = skip_x + x4  
        # x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x


class GATv2Skip(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GATv2Skip, self).__init__()
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
        # self.conv4 = GATv2Conv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        # self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  
        x = skip_x + x1  
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  
        x = skip_x + x3  
        x = self.relu(x)
        
        # x4 = self.conv4(x, edge_index)
        # skip_x = self.skip_4(x)  
        # x = skip_x + x4  
        # x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
class SuperGatSkip(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(SuperGatSkip, self).__init__()
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
        # self.conv4 = SuperGATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        # self.skip_4 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)

        self.mol_hidden1 = nn.Linear(graph_hidden_channels * heads, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  
        x = skip_x + x1  
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  
        x = skip_x + x3  
        x = self.relu(x)
        
        # x4 = self.conv4(x, edge_index)
        # skip_x = self.skip_4(x)  
        # x = skip_x + x4  
        # x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
    
class GCNConvSkip(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GCNConvSkip, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GCNConv(num_node_features, nhid)
        self.skip_1 = nn.Linear(num_node_features, nhid)
        self.conv2 = GCNConv(nhid, nhid)
        self.skip_2 = nn.Linear(nhid, nhid)
        self.conv3 = GCNConv(nhid, nhid)
        self.skip_3 = nn.Linear(nhid, nhid)
        # self.conv4 = GCNConv(nhid, nhid)
        # self.skip_4 = nn.Linear(nhid, nhid)

        self.mol_hidden1 = nn.Linear(nhid, nhid)
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
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  
        x = skip_x + x3  
        x = self.relu(x)
        
        # x4 = self.conv4(x, edge_index)
        # skip_x = self.skip_4(x)  
        # x = skip_x + x4  
        # x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x
    
    
class LEConvSkip(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(LEConvSkip, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = LEConv(num_node_features, nhid)
        self.skip_1 = nn.Linear(num_node_features, nhid)
        self.conv2 = LEConv(nhid, nhid)
        self.skip_2 = nn.Linear(nhid, nhid)
        self.conv3 = LEConv(nhid, nhid)
        self.skip_3 = nn.Linear(nhid, nhid)

        self.mol_hidden1 = nn.Linear(nhid, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)

    def forward(self, graph_batch):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  
        x = skip_x + x1  
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x) 
        x = skip_x + x3  
        x = self.relu(x)
        
        x = global_max_pool(x, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        return x   

############################
# Text Encoder
############################
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
        self.bert = AutoModel.from_pretrained(
            model_name, 
            config=config,
            )
        
        for name, param in self.bert.transformer.named_parameters():
            if 'layer.0' in name or 'layer.1' in name:
                param.requires_grad = False
                
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
          
        return encoded_text.last_hidden_state[:,0,:]

# LoRa Text Encoder

class TextEncoder_lora(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super(TextEncoder_lora, self).__init__() 
        # Define the LoRA Configuration
        self.lora_config = LoraConfig(
            r=8, 
            lora_alpha=32,
            lora_dropout=0.05, 
            target_modules=["query", "key","value"], 
            bias='none',
        )
        self.bert = AutoModel.from_pretrained(model_name)
        self.peft_model = get_peft_model(self.bert, 
                            self.lora_config)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.peft_model(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:,0,:]
 
class Model(nn.Module):
    def __init__(
        self, 
        model_name, 
        num_node_features, 
        nout, 
        nhid, 
        graph_hidden_channels, 
        heads,
        device_1,
        device_2,
        n_heads_text, 
        n_layers_text, 
        hidden_dim_text, 
        dim_text,
        ):
        super(Model, self).__init__()
        # For model with attention
        self.graph_encoder = GCNConvSkip(num_node_features, nout, nhid).to(device_1)
        self.text_encoder = TextEncoder(model_name, n_heads_text, n_layers_text, hidden_dim_text, dim_text).to(device_2)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder


#########################################
#  Cross Modal Model
########################################

class GraphEncoder_v2_cross(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads, temp):
        super(GraphEncoder_v2_cross, self).__init__()
        self.nhid = nhid
        self.nout = nout
        self.relu = nn.ReLU()
        self.ln = nn.LayerNorm((nout))
        self.conv1 = GATConv(num_node_features, graph_hidden_channels, heads=heads)
        self.skip_1 = nn.Linear(num_node_features, graph_hidden_channels * heads)
        self.conv2 = GATConv(graph_hidden_channels * heads, graph_hidden_channels, heads=heads)
        self.skip_2 = nn.Linear(graph_hidden_channels * heads, graph_hidden_channels * heads)
        self.conv3 = GCNConv(graph_hidden_channels * heads, nout)
        self.skip_3 = nn.Linear(graph_hidden_channels * heads, nout)

        self.ln1 = nn.LayerNorm((nout))
        self.mol_hidden1 = nn.Linear(nout, nhid)
        self.mol_hidden2 = nn.Linear(nhid, nout)
        
        self.temp = temp

    def forward(self, graph_batch, with_latent=False):
        x = graph_batch.x
        edge_index = graph_batch.edge_index
        batch = graph_batch.batch
        x1 = self.conv1(x, edge_index)
        skip_x = self.skip_1(x)  
        x = skip_x + x1  
        x = self.relu(x)
        
        x2 = self.conv2(x, edge_index)
        skip_x = self.skip_2(x)  
        x = skip_x + x2  
        x = self.relu(x)
        
        x3 = self.conv3(x, edge_index)
        skip_x = self.skip_3(x)  
        z = skip_x + x3  
        # x = self.relu(z)
        
        x = global_max_pool(z, batch)
        x = self.mol_hidden1(x).relu()
        x = self.mol_hidden2(x)
        x = self.ln1(x)
        x = x * torch.exp(self.temp)
        if with_latent:
            return x, z
        else:
            return x

############################
#### Text Encoder
############################

class TextEncoder_cross(nn.Module):
    def __init__(self, model_name, n_heads_text, n_layers_text, hidden_dim_text, dim_text, nhid, nout, temp, device_2):
        super(TextEncoder_cross, self).__init__()
        config = AutoConfig.from_pretrained(
            model_name, 
            n_heads=n_heads_text,
            n_layers=n_layers_text,
            hidden_dim=hidden_dim_text,
            dim=dim_text,
            )
        self.bert = AutoModel.from_pretrained(
            model_name, 
            config=config,
            )
        
        self.device = device_2
        
        self.cross_modal_decoder = TransformerDecoder(TransformerDecoderLayer(dim_text, 12, nhid), num_layers=1).to(device_2)
        self.text_hidden1 = nn.Linear(dim_text, nhid).to(device_2)
        self.text_hidden2 = nn.Linear(nhid, nout).to(device_2)
        self.ln2 = nn.LayerNorm((nout)).to(device_2)
        self.temp = temp
        
        for name, param in self.bert.transformer.named_parameters():
            if 'layer.0' in name or 'layer.1' in name:
                param.requires_grad = False
                
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
            
    def forward(self, input_ids, attention_mask, graph_batch, graph_latent):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        
        # Credit : Text2Mol: Cross-Modal Molecule Retrieval with Natural Language Queries
        node_features = torch.zeros((graph_batch.num_graphs, 512, graph_latent.shape[1])).to(self.device)
        for i, p in enumerate(graph_batch.ptr):
          if p == 0: 
            old_p = p
            continue
          node_features[i - 1, :p-old_p, :] = graph_latent[old_p:torch.min(p, old_p + 512), :]
          old_p = p
        node_features = torch.transpose(node_features, 0, 1)
        ##
        tgt_mask, memory_mask, tgt_key_padding_mask, memory_key_padding_mask = None, None, None, None
        text_output = self.cross_modal_decoder(encoded_text['last_hidden_state'].transpose(0,1), node_features,
                                     tgt_key_padding_mask=attention_mask==0, memory_key_padding_mask=None)
        text_x = torch.tanh(self.text_hidden1(text_output[0,:,:])) #[CLS] pooler
        text_x = self.text_hidden2(text_x)
        text_x = self.ln2(text_x)
        text_x = text_x * torch.exp(self.temp)
        return text_x
    
class Model_cross(nn.Module):
    def __init__(
        self, 
        model_name, 
        num_node_features, 
        nout, 
        nhid, 
        graph_hidden_channels, 
        heads,
        device_1,
        device_2,
        n_heads_text, 
        n_layers_text, 
        hidden_dim_text, 
        dim_text,
        ):
        super(Model_cross, self).__init__()
        self.temp = nn.Parameter(torch.Tensor([0.07])).to(device_2)
        self.graph_encoder = GraphEncoder_v2_cross(num_node_features, nout, nhid, graph_hidden_channels, heads, self.temp).to(device_1)
        self.text_encoder = TextEncoder_cross(model_name, n_heads_text, n_layers_text, hidden_dim_text, dim_text, nhid, nout, self.temp, device_2).to(device_2)
        
    
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_proj, graph_latent  = self.graph_encoder(graph_batch, with_latent=True)
        text_x = self.text_encoder(input_ids, attention_mask, graph_batch, graph_latent)
        
        return graph_proj, text_x
        
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder