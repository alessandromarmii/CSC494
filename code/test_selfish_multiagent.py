from agents.basicQAgent import QLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np


# Test 3 selfish (i.e. maximizing individual reward, not social) basic Q agents in 8x8 grid.

# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [QLearningAgent(env.observation, env.action_space) for _ in range(3)]
for agent in agents:
    env.add_agent(agent)

# Training loop
num_episodes = 8000
training_rewards = []
batch_size = 250

# Initialize a buffer for each agent
episode_buffers = {i: [] for i in range(len(agents))}
counters = {i: 0 for i in range(len(agents))} 

for episode in range(num_episodes):

    if episode > 1 and episode % 200 == 0:
        print(episode, np.mean(training_rewards[-100:]))

    observations = env.reset()

    episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent

    while True:
        actions = [agent.select_action(observation) for agent, observation in zip(agents, observations)]
        next_observations, done, info = env.step(actions)

        for i, agent in enumerate(agents):
            agent_reward = 1 if i in info["rewarded agents"] else 0
            episode_rewards[i] += agent_reward

            # Add experience to the agent's buffer
            episode_buffers[i].append((observations[i], actions[i], agent_reward, next_observations[i], done))
            counters[i] += 1

            # Check if buffer is ready for batch update
            if counters[i] >= batch_size:
                states, actions, rewards, next_states, dones = zip(*episode_buffers[i])
                agent.update_q_function(states, actions, rewards, next_states, dones)
                episode_buffers[i] = []  # Clear the buffer after updating
                counters[i] = 0

        observations = next_observations
        if done:
            break

    reward_per_episode = sum(episode_rewards)  # reward by agent
    training_rewards.append(reward_per_episode)


# Plot training performance
plt.figure(figsize=(12, 5))
plt.plot(training_rewards, label='Total Reward per Episode during Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance of 3 QLearning Agents')
plt.legend()
plt.show()

# Testing loop
test_total_rewards = []
agent_test_rewards = [[] for _ in agents]  # Initialize reward storage for each agent

num_test_episodes = 500
for episode in range(num_test_episodes):

    observations = env.reset()

    episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent

    while True:
    
        # actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]
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

print(f"Average Total Reward (3 Selfish Q-Learning Agents): {average_test_reward}")
for i, average_reward in enumerate(average_test_rewards_per_agent):
    print(f"Agent {i+1} collected {average_reward} apples per episode on average.")

# Plot testing performance
plt.figure(figsize=(12, 5))
plt.plot(test_total_rewards, label='Total Reward per Episode during Testing')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Testing Performance of 3 QLearning Agents')
plt.legend()
plt.show()

import os
import torch

for i, agent in enumerate(agents):
    file_name = f'model_weights/selfish_multiagent_agent{i}.pth'

    # Check if file already exists
    if os.path.isfile(file_name):
        user_input = input(f"{file_name} already exists. Do you want to overwrite it? (y/n): ")
        if user_input.lower() != 'y':
            print(f"Not saving {file_name}.")
            continue

    # Replace 'q_table' with the actual attribute name if it's different
    torch.save(agent.q_network.state_dict(), f'model_weights/selfish_multiagent_agent{i}.pth')
    print(f"Saved {file_name}.")