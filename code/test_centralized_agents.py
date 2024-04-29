# epochs_to_run are 4400, 4200, 4600, 4000
import matplotlib.pyplot as plt
from agents.centralizedAgent import QLearningAgentCentralized
from agents.agent import Agent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
import torch


# Load the model
env = OrchardEnv(agents=[], max_apples=1000)

agents = [QLearningAgentCentralized(env.observation, env.action_space, learning_rate=0.0001), Agent(), Agent()]

num_agents = len(agents)

for agent in agents:
    env.add_agent(agent)

state_dict = torch.load(f'/final_weights/centralized/weights.pth')
agents[0].q_network.load_state_dict(state_dict)

# Testing loop
test_total_rewards = []
agent_test_rewards = [[] for _ in agents]  # Initialize reward storage for each agent

num_test_episodes = 500
for episode in range(num_test_episodes):
    observations = env.reset()
    observations = np.concatenate(observations)

    episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent

    while True:

        actions = agents[0].select_action(observations, test=True)
        next_observations, done, info = env.step(actions)

        next_observations = np.concatenate(next_observations)

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

print(f"Average Total Reward (3 Quasi-Optimal Q-Learning Agents): {average_test_reward}")
for i, average_reward in enumerate(average_test_rewards_per_agent):
    print(f"Agent {i+1} collected {average_reward} apples per episode on average.")

# Plot testing performance
plt.figure(figsize=(12, 5))
plt.plot(test_total_rewards, label='Total Reward per Episode during Testing')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Testing Performance of 3 Centralized QLearning Agents')
plt.hlines(average_test_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward')
plt.legend()
plt.show()
