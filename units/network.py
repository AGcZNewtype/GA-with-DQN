#Nerual network agent
import random

import cv2
import torchvision.transforms as T
import torch
from torch import nn
import numpy as np

import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN Agent
class AGAgent(nn.Module):
    def __init__(self, in_channels=4, num_actions=5):
        super(AGAgent, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)
        self.action_space = num_actions

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.reshape(x.size(0), -1)))
        return self.fc5(x)

    def act(self, state, epsilon):
        with torch.no_grad():
            #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            # state =
            #state = preprocess_frame(state)
            q_values = self(state)
            # valid_actions = self.action_space.n
            if random.random() < epsilon:
                return random.randrange(self.action_space)
            else:
                # choose action with highest Q-value within valid range
                return q_values[0, :self.action_space].argmax().item()


def preprocess_frame(frame):
    # Crop and resize the frame
    frame = frame[35:195, :, :]
    frame = cv2.resize(frame, (80, 80))
    # Convert to grayscale and normalize
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = frame / 255.0
    # Convert to tensor and stack 4 frames
    frame = torch.from_numpy(frame).float()
    transform = T.Compose([T.ToPILImage(), T.Grayscale(num_output_channels=1), T.Resize((80, 80)), T.ToTensor()])
    frame = transform(frame).unsqueeze(0)
    state = torch.cat(tuple(frame for _ in range(4)), dim=1)
    return state


class AtariNet1(nn.Module):
    def __init__(self, observation_space, action_space):
        super(AtariNet1, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        # self.hidden_size = hidden_size
        self.conv1 = nn.Conv2d(in_channels=observation_space, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=6 * 6 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=action_space)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            #state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            # state =
            #state = preprocess_frame(state)
            q_values = self(state)
            # valid_actions = self.action_space.n
            if random.random() < epsilon:
                return random.randrange(self.action_space)
            else:
                # choose action with highest Q-value within valid range
                return q_values[0, :self.action_space].argmax().item()



class CCAgent(nn.Module):
    def __init__(self, observation_space, action_space, hidden_size=128):
        super(CCAgent, self).__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.hidden_size = hidden_size

        self.fc1 = nn.Linear(observation_space, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_space)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def act(self, state, epsilon):
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = self(state)
            # valid_actions = self.action_space.n
            if random.random() < epsilon:
                return random.randrange(self.action_space)
            else:
                # choose action with highest Q-value within valid range
                return q_values[0, :self.action_space].argmax().item()
