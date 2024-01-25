import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, MFConv, GATv2Conv, SuperGATConv, GATConv, LEConv, RGCNConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from transformers import AutoModel
# from peft import (
#     LoraConfig, 
#     get_peft_model, 
#     TaskType,
#     PeftModel
# )

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
    

    
class GraphEncoder_v2(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder_v2, self).__init__()
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
     
class GraphConv(nn.Module):
    def __init__(self, num_node_features, nout, nhid):
        super(GraphConv, self).__init__()
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
    
class GraphEncoder(nn.Module):
    def __init__(self, num_node_features, nout, nhid, graph_hidden_channels, heads):
        super(GraphEncoder, self).__init__()
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
    def __init__(self, model_name, nout):
        super(TextEncoder, self).__init__()
        self.bert = AutoModel.from_pretrained(
            pretrained_model_name_or_path=model_name,
            n_head=12,
            n_layer=2,
            # hidden_dim=hidden_dim,
            dim=nout,
            )
        # for name, param in self.bert.transformer.named_parameters():
        #     if 'layer.0' in name or 'layer.1' in name or 'layer.2' in name or 'layer.3' in name or 'layer.4' in name or 'layer.5' in name:
        #         param.requires_grad = False
                
        # for param in self.bert.embeddings.parameters():
        #     param.requires_grad = False
        # self.attentionpooling = AttentionPooling(hidden_dim)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.bert(input_ids, attention_mask=attention_mask)
        
        #print(encoded_text.last_hidden_state.size())
        # pooled_output = self.attentionpooling(encoded_text.last_hidden_state) 
        # return pooled_output   
        return encoded_text.last_hidden_state[:,0,:]

class TextEncoder_lora(nn.Module):
    def __init__(self, model_name, hidden_dim):
        super(TextEncoder_lora, self).__init__() 
#         Define the LoRA Configuration
        self.lora_config = LoraConfig(
            r=8, # Rank Number
            lora_alpha=32, # Alpha (Scaling Factor)
            lora_dropout=0.05, # Dropout Prob for Lora
            target_modules=["query", "key","value"], # Which layer to apply LoRA, usually only apply on MultiHead Attention Layer
            bias='none',
#             task_type=TaskType.SEQ_CLS # Seqence to Classification Task
        )
        self.bert = AutoModel.from_pretrained(model_name)
        self.peft_model = get_peft_model(self.bert, 
                            self.lora_config)
        
    def forward(self, input_ids, attention_mask):
        encoded_text = self.peft_model(input_ids, attention_mask=attention_mask)
        return encoded_text.last_hidden_state[:,0,:]
 
class Model(nn.Module):
    def __init__(self, model_name, num_node_features, nout, nhid, graph_hidden_channels, heads, device_1, device_2):
        super(Model, self).__init__()
        # self.graph_encoder = MLPModel(num_node_features, nout, nhid)
        # self.graph_encoder = GraphConv(num_node_features, nout, nhid).to(device_1)
        self.graph_encoder = GraphEncoder_v2(num_node_features, nout, nhid, graph_hidden_channels, heads).to(device_1)
        self.text_encoder = TextEncoder(model_name, nout).to(device_2)
        
    def forward(self, graph_batch, input_ids, attention_mask):
        graph_encoded = self.graph_encoder(graph_batch)
        text_encoded = self.text_encoder(input_ids, attention_mask)
        
        return graph_encoded, text_encoded
    
    def get_text_encoder(self):
        return self.text_encoder
    
    def get_graph_encoder(self):
        return self.graph_encoder
