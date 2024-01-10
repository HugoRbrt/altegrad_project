import sys
import argparse
from typing import Optional
import torch
import uuid
import logging
from pathlib import Path
import torch
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
from code import run_experiment

def get_parser(parser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    if parser is None:
        parser = argparse.ArgumentParser(description="Train a model")
    parser.add_argument("-e", "--exp", nargs="+", type=int, help="Experiment id")
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
    'comment': '',
    }
    return cfg

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    args.exp = uuid.uuid4().int
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    #do it for each experiments:
    cfg = configure_experiment(name_exp="baseline", nb_epochs=1, batch_size=32, learning_rate=2e-5, model_name='distilbert-base-uncased', scheduler='cosine', graph_pooling='maxpooling', graph_model='GAT 4 layers', text_model='Roberta', with_attention_pooling=True, with_lora=True, comment='I run with ...')
    run_experiment(cfg, cpu=args.cpu, no_wandb=args.no_wandb)