import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gym

# Hyperparameters
from algorithm_comparison.units.network import CCAgent

BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Replay memory
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Transitions
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# env = gym.make('IceHockeyNoFrameskip-v4')
env_name = 'CartPole-v1'
env = gym.make(env_name)
# env = make_atari('PongNoFrameskip-v4')
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

# DQN Network
# input_size, output_size
policy_net = CCAgent(input_size, output_size).to(device)
target_net = CCAgent(input_size, output_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters())
loss_fn = nn.SmoothL1Loss()

# Replay memory
memory = ReplayMemory(MEMORY_CAPACITY)

# Epsilon function
steps_done = 0


def epsilon_greedy_policy(state):
    global steps_done
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    return policy_net.act(state, epsilon)


# Train function
def train():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.cat(
        [torch.tensor(s, dtype=torch.float32).unsqueeze(0) for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.tensor(batch.state, dtype=torch.float32).to(device)
    action_batch = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(device)
    reward_batch = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1).to(device)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    # This is where policy_net is used for the predicted action values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = loss_fn(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


def init(name):
    global env
    global input_size
    global output_size
    global policy_net
    global target_net
    global env_name
    # env_name = 'Acrobot-v1'
    env_name = name
    env = gym.make(env_name)
    # env = make_atari('PongNoFrameskip-v4')
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    # DQN Network
    # input_size, output_size
    policy_net = CCAgent(input_size, output_size).to(device)
    target_net = CCAgent(input_size, output_size).to(device)

    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()


rewards = []

# 使用DQN进行训练
def rollout(name, model, num_episodes):
    init(name)
    if model is not None:
        for name, _ in policy_net.named_parameters():
            policy_net.state_dict()[name] = model.state_dict()[name]
    target_net.load_state_dict(policy_net.state_dict())
    for i_episode in range(num_episodes):
        # Reset the environment and get the initial state
        state = env.reset()
        # state = preprocess_frame(state)
        done = False
        total_reward = 0

        while not done:
            # Select an action using the epsilon-greedy policy
            action = epsilon_greedy_policy(state)

            # Take the action and observe the next state and reward
            next_state, reward, done, _ = env.step(action)

            # Update the total reward
            total_reward += reward

            # Store the transition in the replay memory
            if done:
                memory.push(state, action, None, reward)
            else:
                memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # state = preprocess_frame(state)
            # Train the network
            train()

        # Print the total reward for the episode
        rewards.append(total_reward)
        # print(f"Episode {i_episode}: Total Reward = {total_reward}")
    return policy_net, rewards

