from dataclasses import dataclass
from blackjack_dqn.agents import BaseAgent
from blackjack_dqn.hyperparameters import BaseHyperparameters

@dataclass
class Policy:
    agent: BaseAgent
    params: BaseHyperparameters
