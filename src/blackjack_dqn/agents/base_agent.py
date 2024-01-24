from abc import ABC, abstractmethod
import pickle

class BaseAgent(ABC):
    """
    Base class for all agents
    """

    @abstractmethod
    def __init__(self, env, hyperparameters, load_path):
        raise NotImplementedError()

    @abstractmethod
    def act(self, state):
        raise NotImplementedError()

    @abstractmethod
    def learn(self):
        raise NotImplementedError()

    def save(self, save_path):
        with open(save_path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, load_path):
        with open(load_path, 'rb') as f:
            return pickle.load(f)