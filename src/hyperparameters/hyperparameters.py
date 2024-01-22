from dataclasses import dataclass


@dataclass
class BaseHyperparameters:
    num_episodes: int = 100000
    max_steps_per_episode: int = 100
    learning_rate: float = 0.0001
    discount_rate: float = 0.99
    max_exploration_rate: float = 1
    min_exploration_rate: float = 0.01
    exploration_decay_rate: float = 0.00007
    replay_memory_size: int = 1000
    batch_size: int = 256


@dataclass
class DQNHyperparameters:
    target_update_frequency: int = 50 # per episode
    metric_measure_frequency: int = 500 # per episode
    policy_measure_frequency: int = 10000 # per episode