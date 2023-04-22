import os
import random
from collections import namedtuple

import numpy as np
import torch

# BATCH_SIZE = 64
from matplotlib import pyplot as plt

from algorithm_comparison.units.atari_wrappers import make_atari, wrap_deepmind
import torch.optim as optim
import torch.nn as nn

from algorithm_comparison.units.network import AGAgent

BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.01
EPS_DECAY = 500
TARGET_UPDATE = 10
MEMORY_CAPACITY = 10000

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Epsilon function
steps_done = 0



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


def epsilon_greedy_policy(state, policy_net):
    global steps_done
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    return policy_net.act(state, epsilon)


# Replay memory
memory = ReplayMemory(MEMORY_CAPACITY)


# Train function
def train(memory, policy_net, target_net, loss_fn, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # atari
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), dtype=torch.bool).to(device)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]).to(device)
    state_batch = torch.tensor(torch.tensor([item.detach().cpu().numpy() for item in batch.state]).to(device),
                               dtype=torch.float32).squeeze(1).to(device)

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


# env_names = ['AlienNoFrameskip-v4', 'IceHockeyNoFrameskip-v4', 'PongNoFrameskip-v4']
# env_names = ['CartPole-v0', 'Acrobot-v1', 'MountainCar-v0']
# env_name = 'CartPole-v0'
env_name = 'AlienNoFrameskip-v4'
lr = 2e-4
eps = 0.001
# def control_dqn_rollout(env, num_episodes, rewards, policy_net, target_net, loss_fn, optimizer, memory=memory):
env = make_atari(env_name)
env = wrap_deepmind(env, scale=False, frame_stack=True)
# DQN Network

input_size = 4
output_size = 2
# input_size, output_size
policy_net = AGAgent(input_size, output_size).to(device)
target_net = AGAgent(input_size, output_size).to(device)

target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer and loss function
optimizer = optim.Adam(policy_net.parameters(), lr=lr, eps=eps)
loss_fn = nn.SmoothL1Loss()
num_episodes = 300
rewards = []

filename = os.path.basename(__file__).split('.')[0]
for i_episode in range(num_episodes):
    # Reset the environment and get the initial state
    state = env.reset()
    # state = preprocess_frame(state)
    done = False
    total_reward = 0

    state = torch.from_numpy(state._force().transpose(2, 0, 1)[None] / 255).float().to(device)

    while not done:

        # Select an action using the epsilon-greedy policy
        action = epsilon_greedy_policy(state, policy_net)

        # Take the action and observe the next state and reward
        next_state, reward, done, _ = env.step(action)

        # next_state = preprocess_frame(next_state)
        next_state = torch.from_numpy(next_state._force().transpose(2, 0, 1)[None] / 255).float().to(device)

        # Update the total reward
        total_reward += reward

        # Store the transition in the replay memory
        if done:
            memory.push(state, action, None, reward)
        else:
            memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Train the network
        train(memory, policy_net, target_net, loss_fn, optimizer)

    target_net.load_state_dict(policy_net.state_dict())
    # Print the total reward for the episode
    # episodes.append(i_episode)
    rewards.append(total_reward)
    if i_episode == 0:
        best_model = policy_net
        best_reward = total_reward
        torch.save(best_model.state_dict(), 'model\\' + filename + env_name + '_atari_DQN.pth')
    elif total_reward > best_reward:
        best_model = policy_net
        torch.save(best_model.state_dict(), 'model\\' + filename + env_name + '_atari_DQN.pth')
    print(f"Episode {i_episode}: Total Reward = {total_reward}")


def plot_rewards(rewards, title):
    # plt.title(title)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode in ' + title)
    plt.show()


plot_rewards(rewards, env_name + ' with DQN')
