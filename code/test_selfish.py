from agents.basicQAgent import QLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
import torch


# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [QLearningAgent(env.observation, env.action_space) for _ in range(3)]

for agent in agents:
    env.add_agent(agent)

state_dict_0 = torch.load("final_weights/decentralized/selfish/agent0.pth")
state_dict_1 = torch.load("final_weights/decentralized/selfish/agent1.pth")
state_dict_2 = torch.load("final_weights/decentralized/selfish/agent2.pth")

agents[0].q_network.load_state_dict(state_dict_0)
agents[1].q_network.load_state_dict(state_dict_1)
agents[2].q_network.load_state_dict(state_dict_2)

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

    total_episode_reward = episode_rewards  # Sum of rewards for all agents in the episode

    test_total_rewards.append(total_episode_reward)  # Store total episode reward
    for i in range(len(agents)):
        agent_test_rewards[i].append(episode_rewards[i])


# Calculate and print average test reward
average_test_reward = np.mean(test_total_rewards)  # Calculate average total reward across all test episodes
average_test_rewards_per_agent = [np.mean(agent_test_rewards[i]) for i in range(len(agents))]  # Calculate average reward per agent

print(f"Average Total Reward (3 Selfish Q-Learning Agents): {average_test_reward}")
for i, average_reward in enumerate(average_test_rewards_per_agent):
    print(f"Agent {i} collected {average_reward} apples per episode on average.")

# Plot testing performance
plt.figure(figsize=(12, 5))
plt.plot(test_total_rewards, label='Total Reward per Episode during Testing')
plt.xlabel('Episode')
plt.hlines(average_test_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward')
plt.ylabel('Total Reward')
plt.title('Testing Performance of 3 Selfish Decentralized Agents')
plt.legend()
# plt.savefig(f'images/selfish/training_results.png')
