import os
import random

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn

from algorithm_comparison.units.DQN_net import rollout
from algorithm_comparison.units.network import CCAgent

POPULATION_SIZE = 10
NUM_GENERATIONS = 100
MUTATION_PROBABILITY = 0.5
TOURNAMENT_SIZE = 2
CROSSOVER_PROBABILITY = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#高斯变异
def gaussian_mutation(net, std=0.01):
    for param in net.parameters():
        if param.requires_grad:
            noise = torch.empty_like(param).normal_(mean=0, std=std) #增加噪声
            mutated_param = param.data + noise
            param.data = mutated_param
    return net

#神经网络变异
def mutate_network(net, Net, input_size, output_size):
    new_net = Net(input_size, output_size).to(device)         #创建神经网络结构
    for name, param in net.named_parameters():
        if random.random() < MUTATION_PROBABILITY:  #判断是否进行变异
            param_shape = param.shape
            mutation = torch.randn(param_shape).to(device)     #增加随机参数结构
            new_param = param + mutation
            # setattr(new_net, name, nn.Parameter(new_param))
            setattr(new_net, name.replace('.', '_'), nn.Parameter(new_param))
        else:
            # setattr(new_net, name, nn.Parameter(param))
            setattr(new_net, name.replace('.', '_'), nn.Parameter(param))
    new_net = gaussian_mutation(new_net)
    return new_net

#交叉
def crossover(parent1, parent2, Net, input_size, output_size):
    child = Net(input_size, output_size).to(device)
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


#父代选择
def tournament_select(fitness, num_parents, population):
    parents = []
    for i in range(num_parents):
        # tournament = random.sample(population, TOURNAMENT_SIZE)
        tournament = [random.randint(0, TOURNAMENT_SIZE) for _ in range(TOURNAMENT_SIZE)]
        tournament_fitnesses = [fitness[p] for p in tournament]
        winner = population[tournament_fitnesses.index(max(tournament_fitnesses))]
        parents.append(winner)
    return parents

#评估网络
def evaluate_network(net, env):
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        state = torch.tensor(state, dtype=torch.float32).to(device)
        q_values = net(state)

        action = torch.argmax(q_values).item()
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
    return total_reward


filename = os.path.basename(__file__).split('.')[0]
best_model = None
best_fitness = -np.inf

env_name = 'CartPole-v1'

env = gym.make(env_name)

#引入用于Classic Control的神经网络结构
Net = CCAgent

#根据任务观测空间和动作空间创建神经网络输入输出结构
input_size = env.observation_space.shape[0]
output_size = env.action_space.n

#初始化种群，创建种群中的个体神经网络
population = [Net(input_size, output_size).to(device) for _ in range(POPULATION_SIZE)]
rewards = []
for generation in range(NUM_GENERATIONS):
    fitnesses = [evaluate_network(net, env) for net in population] #计算适应度
    # sorted_population = [x for _, x in sorted(zip(fitnesses, population), reverse=True)]
    sort_fitness = [(key, value) for key, value in enumerate(fitnesses)]     #随机组群适应度排序
    sort_fitness = sorted(sort_fitness, key=lambda x: x[1], reverse=True)
    sorted_population = [population[x[0]] for x in sort_fitness]
    top_individual = sorted_population[0]                           #择优
    rewards.append(sort_fitness[0][1])
    print("Generation:", generation, "Fitness:", sort_fitness)
    parents = []
    for j in range(POPULATION_SIZE):                                #创建父代
        p1, p2 = tournament_select(fitnesses, 2, population)
        parents.append((p1, p2))

    offspring = []                                                  #创建子代
    for parent1, parent2 in parents:
        child = crossover(parent1, parent2, Net, input_size, output_size)       #交叉
        child = mutate_network(child, Net, input_size, output_size)             #变异
        offspring.append(child)
    # select the one of offspring to DQN
    ind = offspring[0]
    ind, _ = rollout(env_name, ind, 10)
    offspring[0] = ind
    new_population = population + offspring
    new_fitnesses = [evaluate_network(net, env) for net in offspring]           #评估新种群
    new_fitnesses = fitnesses + new_fitnesses
    if generation == 0:
        best_model = sorted_population[0]
        best_fitness = sort_fitness[0]
        torch.save(best_model.state_dict(), filename + '_' + env_name + '_gnn.pth')
    elif best_fitness > sort_fitness[0]:                                        #保存最优模型
        best_model = sorted_population[0]
        torch.save(best_model.state_dict(), filename + '_' + env_name + '_gnn.pth')
    sort_fitness = [(key, value) for key, value in enumerate(new_fitnesses)]     #适应度排序
    sort_fitness = sorted(sort_fitness, key=lambda x: x[1], reverse=True)
    sorted_population = [new_population[x[0]] for x in sort_fitness]
    new_population = sorted_population[:POPULATION_SIZE]                         #择优出新种群

    population = new_population


def plot_rewards(rewards, title):
    # plt.title(title)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Rewards per Episode in ' + title)
    plt.show()


plot_rewards(rewards, env_name + ' with GADQN')
