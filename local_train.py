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
from configuration import CFG_EXPERIMENTS as cfg
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

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args(sys.argv[1:])
    if not WANDB_AVAILABLE:
        args.no_wandb = True
    print(args.exp)
    print(cfg.keys())
    assert args.exp in cfg.keys(), f"Experiment {args.exp} not found in configuration.py"
    print("running experiment {}".format(args.exp))
    print(cfg[args.exp])
    run_experiment(cfg[args.exp], cpu=args.cpu, no_wandb=args.no_wandb)