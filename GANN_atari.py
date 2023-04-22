import os
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from algorithm_comparison.units.atari_wrappers import make_atari, wrap_deepmind
from algorithm_comparison.units.network import AGAgent

POPULATION_SIZE = 10
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 0.5
TOURNAMENT_SIZE = 2
CROSSOVER_PROBABILITY = 0.1


def evaluate_network(net, env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state = torch.from_numpy(state._force().transpose(2, 0, 1)[None] / 255).float()
        q_values = net(state)

        action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward


# Net = PongNet
# Net = CartPoleNet

def gaussian_mutation(net, std=0.01):
    for param in net.parameters():
        if param.requires_grad:
            noise = torch.empty_like(param).normal_(mean=0, std=std)
            mutated_param = param.data + noise
            param.data = mutated_param
    return net


def mutate_network(net, Net, input_size, output_size):
    new_net = Net(input_size, output_size)
    for name, param in net.named_parameters():
        if random.random() < MUTATION_PROBABILITY:
            param_shape = param.shape
            mutation = torch.randn(param_shape)
            new_param = param + mutation
            # setattr(new_net, name, nn.Parameter(new_param))
            setattr(new_net, name.replace('.', '_'), nn.Parameter(new_param))
        else:
            # setattr(new_net, name, nn.Parameter(param))
            setattr(new_net, name.replace('.', '_'), nn.Parameter(param))
    new_net = gaussian_mutation(new_net)
    return new_net


def crossover(parent1, parent2, Net, input_size, output_size):
    child = Net(input_size, output_size)
    for name, param1 in parent1.named_parameters():
        if random.random() < CROSSOVER_PROBABILITY:
            if name in parent2.state_dict():
                param2 = parent2.state_dict()[name]
            else:
                param2 = param1
        else:
            param2 = param1
        setattr(child, name.replace('.', '_'), nn.Parameter(param2))
    return child


def tournament_select(fitness, num_parents, population):
    parents = []
    for i in range(num_parents):
        # tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament = [random.randint(0, TOURNAMENT_SIZE) for _ in range(TOURNAMENT_SIZE)]
        tournament_fitnesses = [fitness[p] for p in tournament]
        winner = population[tournament_fitnesses.index(max(tournament_fitnesses))]
        parents.append(winner)
    return parents


filename = os.path.basename(__file__).split('.')[0]
best_model = None
best_fitness = -np.inf

env_name = 'AlienNoFrameskip-v4'

# env = gym.make(env_name)
env = make_atari(env_name)
env = wrap_deepmind(env, scale=False, frame_stack=True)

action_space = env.action_space
Net = AGAgent

input_size = env.observation_space.shape[2]
output_size = env.action_space.n


population = [Net(input_size, output_size) for _ in range(POPULATION_SIZE)]
rewards = []

for generation in range(NUM_GENERATIONS):
    fitnesses = [evaluate_network(net, env) for net in population]
    # sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    sort_fitness = [(key, value) for key, value in enumerate(fitnesses)]
    sort_fitness = sorted(sort_fitness, key=lambda x: x[1], reverse=True)
    sorted_population = [population[x[0]] for x in sort_fitness]
    top_individual = sorted_population[0]
    rewards.append(sort_fitness[0][1])
    print("Generation:", generation, "Fitness:", sort_fitness)
    parents = []
    for j in range(POPULATION_SIZE):
        p1, p2 = tournament_select(fitnesses, 2, population)
        parents.append((p1, p2))

    offspring = []
    for parent1, parent2 in parents:
        child = crossover(parent1, parent2, Net, input_size, output_size)
        child = mutate_network(child, Net, input_size, output_size)
        offspring.append(child)

    new_population = population + offspring
    new_fitnesses = [evaluate_network(net, env) for net in offspring]
    new_fitnesses = fitnesses + new_fitnesses
    if generation == 0:
        best_model = sorted_population[0]
        best_fitness = sort_fitness[0]
        torch.save(best_model, 'model\\'+filename + '_' + env_name + '_gnn')
    elif best_fitness > sort_fitness[0]:
        best_model = sorted_population[0]
        torch.save(best_model, 'model\\'+filename + '_' + env_name + '_gnn')
    sort_fitness = [(key, value) for key, value in enumerate(new_fitnesses)]
    sort_fitness = sorted(sort_fitness, key=lambda x: x[1], reverse=True)
    sorted_population = [new_population[x[0]] for x in sort_fitness]
    new_population = sorted_population[:POPULATION_SIZE]

    population = new_population


def plot_rewards(rewards, title):
    # plt.title(title)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode in ' + title)
    plt.show()

plot_rewards(rewards, env_name + ' with GNN')