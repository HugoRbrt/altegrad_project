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
import time
import os
import pandas as pd
from shared import (
    ROOT_DIR, OUTPUT_FOLDER_NAME,
    ID, NAME, NB_EPOCHS,
    TRAIN, VALIDATION, TEST,
)
#from experiments import get_experiment_config, get_training_content
WANDB_AVAILABLE = False
try:
    WANDB_AVAILABLE = True
    import wandb
except ImportError:
    logging.warning("Could not import wandb. Disabling wandb.")
    pass


def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, required=True, help="Experiment id")
    parser.add_argument("-o", "--output-dir", type=str, default=ROOT_DIR/OUTPUT_FOLDER_NAME, help="Output directory")
    parser.add_argument("-nowb", "--no-wandb", action="store_true", help="Disable weights and biases")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    return parser


def training_loop(
    model, optimizer, dl_dict: dict, config: dict,
    device: str = "cuda", wandb_flag: bool = False,
    output_dir: Path = None
):
    for n_epoch in tqdm(range(config[NB_EPOCHS])):
        current_loss = {TRAIN: 0., VALIDATION: 0., TEST: 0.}
        for phase in [TRAIN, VALIDATION, TEST]:
            if phase == TRAIN:
                model.train()
            else:
                model.eval()
            for x, y in tqdm(dl_dict[phase], desc=f"{phase} - Epoch {n_epoch}"):
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == TRAIN):
                    y_pred = model(x)
                    loss = torch.nn.functional.mse_loss(y_pred, y)
                    if phase == TRAIN:
                        loss.backward()
                        optimizer.step()
                current_loss[phase] += loss.item()
            current_loss[phase] /= (len(dl_dict[phase]))
        for phase in [VALIDATION, TEST]:
            print(f"{phase}: Epoch {n_epoch} - Loss: {current_loss[phase]:.3e}")
        if output_dir is not None:
            with open(output_dir/f"metrics_{n_epoch}.json", "w") as f:
                json.dump(current_loss, f)
        if wandb_flag:
            wandb.log(current_loss)
    if output_dir is not None:
        torch.save(model.cpu().state_dict(), output_dir/"last_model.pt")
    return model


def train(config: dict, output_dir: Path, device: str = "cuda", wandb_flag: bool = False):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Training experiment {config[ID]} on device {device}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir/"config.json", "w") as f:
        json.dump(config, f)
    model, optimizer, dl_dict = get_training_content(config, device=device)
    model.to(device)
    if wandb_flag:
        wandb.init(
            project="mva-pepites",
            name=config[NAME],
            tags=["debug"],
            config=config
        )
    model = training_loop(model, optimizer, dl_dict, config, device=device,
                          wandb_flag=wandb_flag, output_dir=output_dir)

    if wandb_flag:
        wandb.finish()


def train_main(argv):
    parser = get_parser()
    args = parser.parse_args(argv)
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    for exp in args.exp:
        config = get_experiment_config(exp)
        print(config)
        output_dir = Path(args.output_dir)/config[NAME]
        logging.info(f"Training experiment {config[ID]} on device {device}...")
        train(config, device=device, output_dir=output_dir, wandb_flag=not args.no_wandb)


########## HUGO ##########

from data_loader import GraphTextDataset, GraphDataset, TextDataset
from model import Model

def contrastive_loss(v1, v2):
    logits = torch.matmul(v1,torch.transpose(v2, 0, 1))
    labels = torch.arange(logits.shape[0], device=v1.device)
    return CE(logits, labels) + CE(torch.transpose(logits, 0, 1), labels)
    
if __name__ == "__main__":
    CE = torch.nn.CrossEntropyLoss()

    model_name = 'distilbert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    gt = np.load("./data/token_embedding_dict.npy", allow_pickle=True)[()]
    val_dataset = GraphTextDataset(root='./data/', gt=gt, split='val', tokenizer=tokenizer)
    train_dataset = GraphTextDataset(root='./data/', gt=gt, split='train', tokenizer=tokenizer)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    nb_epochs = 5
    batch_size = 32
    learning_rate = 2e-5

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
        if best_validation_loss==val_loss:
            print('validation loss improoved saving checkpoint...')
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

    test_cids_dataset = GraphDataset(root='./data/', gt=gt, split='test_cids')
    test_text_dataset = TextDataset(file_path='./data/test_text.txt', tokenizer=tokenizer)

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


    from sklearn.metrics.pairwise import cosine_similarity

    similarity = cosine_similarity(text_embeddings, graph_embeddings)

    solution = pd.DataFrame(similarity)
    solution['ID'] = solution.index
    solution = solution[['ID'] + [col for col in solution.columns if col!='ID']]
    solution.to_csv('submission.csv', index=False)