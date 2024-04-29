from agents.agentWithFeedback import agentWithFeedback
from agents.agentWithFeedback import solve_attention_optimization_problem
from agents.qNetworks.QNetworks import NonSelfishQNetwork
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch
import time
import os


num_agents = 3

# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [agentWithFeedback(env.observation, env.action_space, attention_allocation=[0, 0]) for _ in range(num_agents)]

# Load state dicts
for i, agent in enumerate(agents):
    agent.feedback_importance = 0.5
    env.add_agent(agent) # add to environment
    agent.q_network_selfish.load_state_dict(torch.load(f"final_weights/decentralized/feedback/selfish/agent{i}.pth"))
    agent.q_network_expected_feedback.load_state_dict(torch.load(f"final_weights/decentralized/feedback/expected/agent{i}.pth"))

# PROBLEM SET-UP GLOBAL VARIABLES:

alpha = 1  # Delay Sensitivity

num_agents = 3 

rate_budget = num_agents # Max attention that an agent can allocate

input_size = np.prod(env.observation.shape) + 2

# Initialize NonSelfishQNetwork models
optimal_models = [NonSelfishQNetwork(input_size, env.action_space.n, 600) for _ in range(num_agents)]

# Load models
for i, model in enumerate(optimal_models):
    model.load_state_dict(torch.load(f"final_weights/decentralized/nonselfish/agent{i}.pth"))


# Testing loop
test_total_rewards = []
agent_test_rewards = [[] for _ in agents]  # Initialize reward storage for each agent

num_test_episodes = 500
for episode in range(num_test_episodes):
    observations = env.reset()
    
    episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent

    while True:
        
        actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]
        next_observations, done, info = env.step(actions)

        for i, agent in enumerate(agents):
            agent_reward = 1 if i in info["rewarded agents"] else 0

            episode_rewards[i] += agent_reward

        observations = next_observations
        if done:
            break

    total_episode_reward = sum(episode_rewards)  # Sum of rewards for all agents in the episode
    test_total_rewards.append(total_episode_reward)  # Store total episode reward
    for i in range(len(agents)):
        agent_test_rewards[i].append(episode_rewards[i])

# Calculate and print average test reward
average_test_reward = np.mean(test_total_rewards)  # Calculate average total reward across all test episodes
average_test_rewards_per_agent = [np.mean(agent_test_rewards[i]) for i in range(len(agents))]  # Calculate average reward per agent

print("Average Test Reward: ", average_test_reward)
for i in range(num_agents):
    print(f'Average Test Reward for Agent {i}: {average_test_rewards_per_agent[i]}')

plt.figure(figsize=(10, 6))
plt.plot(test_total_rewards, label="Total Reward per Episode")
plt.title("Testing Performance of 3 Agents with Feedback")
plt.hlines(average_test_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward')
plt.xlabel("Episodes")
plt.ylabel("Reward")
plt.legend()
# plt.savefig(f'images/agents_with_feedback/training_results_episode_{ep}.png')
plt.close()

