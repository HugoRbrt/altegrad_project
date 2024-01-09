import sys
import argparse
from typing import Optional
import torch
import logging
from pathlib import Path
import wandb
import json
from tqdm import tqdm
from torch_geometric.data import DataLoader
from torch.utils.data import DataLoader as TorchDataLoader
import numpy as np
from transformers import AutoTokenizer
import torch
from torch import optim
from sklearn.metrics.pairwise import cosine_similarity
import time
import os
import pandas as pd
from data_loader import GraphTextDataset, GraphDataset, TextDataset
from model import Model
from configuration import GIT_USER
from shared import (
    ROOT_DIR, OUTPUT_FOLDER_NAME,
    ID, NAME, NB_EPOCHS,
    TRAIN, VALIDATION, TEST,
)
WANDB_AVAILABLE = False
try:
    WANDB_AVAILABLE = True
    import wandb
except ImportError:
    logging.warning("Could not import wandb. Disabling wandb.")
    pass

CE = torch.nn.CrossEntropyLoss()

def contrastive_loss(v1, v2):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)

def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser

def configure_experiment(name_exp: str, nb_epochs: int, batch_size: int, learning_rate: float, model_name: str, scheduler: str, graph_pooling: str, graph_model: str, text_model: str, with_attention_pooling: bool, with_lora: bool, comment: str) -> dict:
    cfg = {
    'who': GIT_USER,
    'name_exp': name_exp,
    'nb_epochs': nb_epochs,
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'model_name': model_name,
    'num_node_features': 300,
    'nout':  768,
    'nhid': 300,
    'graph_hidden_channels': 300,
    'nhead_graph': 20,
    'with_attention_pooling':True,
    'with_lora':True,
    'text_model':'Roberta',
    'scheduler':'cosine',
    'graph_pooling':'maxpooling',
    'graph_model':'GAT 4 layers',
    'comment': '',
    }
    return cfg

def run_experiment(cfg, args):
    if not args.no_wandb:
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

    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = Model(model_name=model_name, num_node_features=300, nout=768, nhid=300, graph_hidden_channels=300) # nout = bert model hidden dim
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
                if not args.no_wandb:
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
        if not args.no_wandb:
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
    
    if not args.no_wandb:
        submission_artifact = wandb.Artifact('submission', type='results')
        submission_artifact.add_file('submission.csv')
        wandb.log_artifact(submission_artifact)
        
        model_artifact = wandb.Artifact('model', type='model')
        model_artifact.add_file(save_path)
        wandb.log_artifact(model_artifact)
        wandb.finish()

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    #do it for each experiments:
    cfg = configure_experiment(name_exp="baseline", nb_epochs=1, batch_size=32, learning_rate=2e-5, model_name='distilbert-base-uncased', scheduler='cosine', graph_pooling='maxpooling', graph_model='GAT 4 layers', text_model='Roberta', with_attention_pooling=True, with_lora=True, comment='I run with ...')
    run_experiment(cfg, args)
    #and you can redo another one