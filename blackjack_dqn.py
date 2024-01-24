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

# Define your hyperparameters
hyperparameters = {
    "num_episodes": 100000,
    "max_steps_per_episode": 100,
    "learning_rate": 0.0001,
    "discount_rate": 0.99,
    "max_exploration_rate": 1,
    "min_exploration_rate": 0.01,
    "exploration_decay_rate": 0.00007,
    "replay_memory_size": 1000,
    "batch_size": 256,
    "target_update_frequency": 50, # per episode
    "metric_measure_frequency": 500, # per episode
    "policy_measure_frequency": 10000 # per episode
}

# Start a new run
run = wandb.init(project="rl_dqn_blackjack", entity="aydenkhalil619", config=hyperparameters)


class EpsilonGreedyStrategy:
    def __init__(self, start, end, decay):
        self.start = start
        self.end = end
        self.decay = decay
    
    def get_exploration_rate(self, current_step):
        return self.end + (self.start - self.end) * np.exp(-1. * current_step * self.decay)
    
    def select_action(self, env: gym.Env, state: torch.Tensor, policy_net: nn.Module, current_step: int):
        exploration_rate = self.get_exploration_rate(current_step)
        if exploration_rate > np.random.rand():
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action = policy_net(state).argmax().item()
        return action

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Tuple[Any]]:
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

def get_policy_df(policy_net: nn.Module) -> pd.DataFrame:
    # Define the row and column indices as seen in the image
    hand_totals = ['4', '5', '6', '7', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', 'AA', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9']
    dealer_up_cards = ['2', '3', '4', '5', '6', '7', '8', '9', '10', 'A']

    policy_df = pd.DataFrame(index=hand_totals, columns=dealer_up_cards)

    for hand in hand_totals:
        for upcard in dealer_up_cards:
            if hand == 'AA':
                state_hand = 12
                ace = True
            elif 'A' in hand:
                state_hand = 11 + int(hand[1])
                ace = True
            else:
                state_hand = int(hand)
                ace = False
            state = torch.tensor([state_hand, int(upcard.replace('A', '1')), ace], dtype=torch.float32)
            action = policy_net(state).argmax().item()
            if action == 1:
                policy_df.loc[hand, upcard] = 'H'
            else:
                policy_df.loc[hand, upcard] = 'S'
    return policy_df


def log_policy_df(policy_df: pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.heatmap(policy_df.replace({'H': 1, 'S': 0}), annot=policy_df.values, fmt='', cmap=['red', 'green'], linewidths=.5)
    plt.xlabel('Dealer Upcard')
    plt.ylabel('Hand Total')
    plt.title('Model Policy')

    # Convert plot to a PIL Image object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    
    # Log the image to wandb
    wandb.log({"Model Policy": [wandb.Image(image, caption="Model Policy")]})


env = gym.make('Blackjack-v1')

# Initialize the policy network with random weights
policy_net = DQN(
    input_dim=len(env.observation_space), 
    output_dim=env.action_space.n
)

# Create the target network as a clone of the policy network
target_net = DQN(
    input_dim=len(env.observation_space), 
    output_dim=env.action_space.n
)

optimizer = torch.optim.Adam(policy_net.parameters(), lr=wandb.config.learning_rate)
loss_fn = nn.MSELoss()

# Initialize the strategy
strategy = EpsilonGreedyStrategy(
    start=wandb.config.max_exploration_rate, 
    end=wandb.config.min_exploration_rate, 
    decay=wandb.config.exploration_decay_rate
)

replay_memory = ReplayMemory(capacity=wandb.config.replay_memory_size)
epoch_reward = 0
epoch_loss = 0

for n_episode in tqdm(range(wandb.config.num_episodes)):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float32)
    done = False
    
    for t in range(wandb.config.max_steps_per_episode):
        # Navigation Step
        # Select Action
        action = strategy.select_action(env, state, policy_net, n_episode)
        
        # Take Action
        next_state, reward, done, _, _ = env.step(action)
        next_state = torch.tensor(next_state, dtype=torch.float32)
        epoch_reward += reward

        # Store the transition in memory
        replay_memory.push(state, action, reward, next_state, done)

        # Training Step
        # 1. Sample random transitions from replay memory and prepare
        transitions = replay_memory.sample(wandb.config.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)
        batch_state = torch.stack(batch_state)
        batch_action = torch.tensor(batch_action, dtype=torch.int64)
        batch_reward = torch.tensor(batch_reward, dtype=torch.float32)
        batch_next_state = torch.stack(batch_next_state)
        batch_done = torch.tensor(batch_done, dtype=torch.bool)

        # 2. Compute Current Q Values (Q(s,a)) using policy network
        # Pass the batch of states through the policy network
        all_q_values = policy_net(batch_state)
        current_q_values = all_q_values.gather(1, batch_action.unsqueeze(-1)).squeeze(-1) # Select the Q-values of the taken actions
        
        # 3. Compute Target Q Values (Q(s',a')) using target network
        # Computing the maximum of Q-values for the next states (max term)
        all_target_q_values = target_net(batch_next_state)
        max_next_q_values = all_target_q_values.max(1).values.detach()
        
        # Compute the expected Q-values (using Bellman)
        target_q_values =  batch_reward + (wandb.config.discount_rate * max_next_q_values * (1 - batch_done.float()))

        # 4. Compute the loss and update the weights of the policy network
        # Compute the loss
        loss = loss_fn(current_q_values, target_q_values)
        epoch_loss += loss.item()

        # Zero the gradients
        optimizer.zero_grad()

        # Compute the gradients
        loss.backward()

        # Update the weights
        optimizer.step()

        # Final Step
        # Update the state
        state = next_state
        if done:
            break
    
    
    wandb.log({"Exploration Rate": strategy.get_exploration_rate(n_episode)})

    if n_episode > 0 and n_episode % wandb.config.metric_measure_frequency == 0:
        wandb.log({
            "Win Rate": (epoch_reward + wandb.config.metric_measure_frequency / 2) / wandb.config.metric_measure_frequency,
            "Epoch Reward": epoch_reward / wandb.config.metric_measure_frequency,
            "Epoch Loss": epoch_loss / wandb.config.metric_measure_frequency
        })
        epoch_reward = 0
        epoch_loss = 0

    # Update the target network, copying all weights and biases in DQN
    if n_episode > 0 and n_episode % wandb.config.target_update_frequency == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if n_episode > 0 and n_episode % wandb.config.policy_measure_frequency == 0:
        policy_df = get_policy_df(policy_net)
        log_policy_df(policy_df)


run.finish() # type: ignore
