import gym
import numpy as np
from typing import List, Tuple, Any
import random
from collections import deque
import torch
import torch.nn as nn
import wandb
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from tqdm import tqdm
import PIL
import argparse
import dataclasses
import torch
import wandb
import gym
from collections import namedtuple

from blackjack_dqn.agents import DQNAgent
from blackjack_dqn.hyperparameters import DQNHyperparameters
from blackjack_dqn.utils import Policy


POLICIES = {
    "dqn": Policy(DQNAgent, DQNHyperparameters),
}

# run = wandb.init(project="rl_dqn_blackjack", entity="aydenkhalil619", config=hyperparameters)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, default="dqn")
    parser.add_argument("--env", type=str, default="Blackjack-v1")

    parser.add_argument("--wandb", action="store_true", default=True)
    parser.add_argument("--no-wandb", action="store_false", dest="wandb")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    hp = POLICIES[args.policy].params()
    
