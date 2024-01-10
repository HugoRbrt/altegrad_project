from .model import Model
from .data_loader import GraphTextDataset, GraphDataset, TextDataset
from torch import optim
from sklearn.metrics.pairwise import cosine_similarity
import time
from torch_geometric.data import DataLoader
import wandb
import torch
import uuid
import os
import pandas as pd
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from transformers import AutoTokenizer


CE = torch.nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)


def run_experiment(cfg, cpu=False, no_wandb=False):
    """this function allows to run an experiments with the given configuration in cfg
    (see local_train.py->configure_experiment for the format of cfg)
    You can add configurations possibilities in the cfg

    Args:
        cfg (dict): contains all informations to run the experiments
        cpu (bool, optional): if True, force CPU. Defaults to False.
        no_wandb (bool, optional): if True, disable wandb. Defaults to False.
    """
    if not no_wandb:
        run = wandb.init(
        project="text2mol",
        name=cfg['name_exp'],
        config=cfg,
        )
    nb_epochs = cfg['nb_epochs']
    batch_size = cfg['batch_size']
    learning_rate =cfg['learning_rate']
    model_name =cfg['model_name']
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("/kaggle/input/nlplsv3/kaggle/working/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='train', tokenizer=tokenizer)

    device = "cpu" if cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(model_name=model_name, num_node_features=cfg['num_node_features'], nout=cfg['nout'], nhid=cfg['nhid'], graph_hidden_channels=cfg['graph_hidden_channels']) # nout = bert model hidden dim
    model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate,
                                    betas=(0.9, 0.999),
                                    weight_decay=0.01)

    epoch = 0
    loss = 0
    losses = []
    count_iter = 0
    time1 = time.time()
    printEvery = 50
    best_validation_loss = 1000000

    for i in range(nb_epochs):
        print('-----EPOCH{}-----'.format(i+1))
        model.train()
        for batch in train_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            current_loss = contrastive_loss(x_graph, x_text)   
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()
            loss += current_loss.item()
            
            count_iter += 1
            if count_iter % printEvery == 0:
                time2 = time.time()
                print("Iteration: {0}, Time: {1:.4f} s, training loss: {2:.4f}".format(count_iter,
                                                                            time2 - time1, loss/printEvery))
                if not no_wandb:
                    wandb.log({
                        "epoch": i, 'loss/train': loss/printEvery,
                    })
                losses.append(loss)
                loss = 0 
        model.eval()       
        val_loss = 0        
        for batch in val_loader:
            input_ids = batch.input_ids
            batch.pop('input_ids')
            attention_mask = batch.attention_mask
            batch.pop('attention_mask')
            graph_batch = batch
            x_graph, x_text = model(graph_batch.to(device), 
                                    input_ids.to(device), 
                                    attention_mask.to(device))
            current_loss = contrastive_loss(x_graph, x_text)   
            val_loss += current_loss.item()
        best_validation_loss = min(best_validation_loss, val_loss)
        print('-----EPOCH'+str(i+1)+'----- done.  Validation loss: ', str(val_loss/len(val_loader)) )
        if not no_wandb:
            wandb.log({
                "epoch": i, 'loss/val':  str(val_loss/len(val_loader)),
            })
        if best_validation_loss==val_loss:
            print('validation loss improved saving checkpoint...')
            save_path = os.path.join('./', 'model'+str(i)+'.pt')
            torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'validation_accuracy': val_loss,
            'loss': loss,
            }, save_path)
            print('checkpoint saved to: {}'.format(save_path))


    print('loading best model...')
    checkpoint = torch.load(save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    graph_model = model.get_graph_encoder()
    text_model = model.get_text_encoder()

    test_cids_dataset = GraphDataset(root='/kaggle/input/nlplsv3/kaggle/working/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='/kaggle/input/nlplsv3/kaggle/working/test_text.txt', tokenizer=tokenizer)

    idx_to_cid = test_cids_dataset.get_idx_to_cid()

    test_loader = DataLoader(test_cids_dataset, batch_size=batch_size, shuffle=False)

    graph_embeddings = []
    for batch in test_loader:
        for output in graph_model(batch.to(device)):
            graph_embeddings.append(output.tolist())

    test_text_loader = TorchDataLoader(test_text_dataset, batch_size=batch_size, shuffle=False)
    text_embeddings = []
    for batch in test_text_loader:
        for output in text_model(batch['input_ids'].to(device), 
                                attention_mask=batch['attention_mask'].to(device)):
            text_embeddings.append(output.tolist())

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv('submission.csv', index=False)
    
    if not no_wandb:
        
        submission_artifact = wandb.Artifact('submission:'+str(uuid.uuid1()), type='csv')
        submission_artifact.add_file('submission.csv')
        wandb.log_artifact(submission_artifact)
        
        model_artifact = wandb.Artifact('model:'+str(uuid.uuid1()), type='model')
        model_artifact.add_file(save_path)
        wandb.log_artifact(model_artifact)
        wandb.finish()
