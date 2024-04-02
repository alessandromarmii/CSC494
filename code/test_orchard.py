from environment.orchard import * 

from agents.optimalQAgent import OptimalQLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch



env = OrchardEnv(agents=[], max_apples=1000)
agents = [OptimalQLearningAgent(env.observation_space, env.action_space) for _ in range(3)]
state_dict_0 = torch.load("model_weights/quasi_optimal_agent0.pth")
state_dict_1 = torch.load("model_weights/quasi_optimal_agent1.pth")
state_dict_2 = torch.load("model_weights/quasi_optimal_agent0.pth")

agents[0].q_network.load_state_dict(state_dict_0)
agents[1].q_network.load_state_dict(state_dict_1)
agents[2].q_network.load_state_dict(state_dict_2)


for agent in agents:
    env.add_agent(agent)


test_total_rewards = []
agent_test_rewards = [[] for _ in agents]  # Initialize reward storage for each agent

num_test_episodes = 1
for episode in range(num_test_episodes):
    observation = env.reset()
    print(observation)
