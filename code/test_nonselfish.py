from agents.optimalQAgent import OptimalQLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
import torch

# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [OptimalQLearningAgent(env.observation, env.action_space, learning_rate=0.001, model_layer_size=600) for _ in range(3)]

state_dict_0 = torch.load("final_weights/decentralized/nonselfish/agent0.pth")
state_dict_1 = torch.load("final_weights/decentralized/nonselfish/agent1.pth")
state_dict_2 = torch.load("final_weights/decentralized/nonselfish/agent2.pth")

agents[0].q_network.load_state_dict(state_dict_0)
agents[1].q_network.load_state_dict(state_dict_1)
agents[2].q_network.load_state_dict(state_dict_2)


for agent in agents:
    agent.epsilon_decay = 0.99997
    agent.epsilon_min = 0
    # agent.epsilon = 0.25
    env.add_agent(agent)

training_is_complete = False
# num_episodes = 100000
num_episodes = 0
training_rewards = []
training_rewards_by_agent  = [[] for _ in range(3)]
batch_size = 250

episode_buffers = {i: [None] * batch_size for i in range(len(agents))}
counter = {i: 0 for i in range(len(agents))}

for episode in range(num_episodes):

    if episode > 1 and episode % 200 == 0:
        avg = np.mean(training_rewards[-200:])
        agent1 =  np.mean(training_rewards_by_agent[0][-200:])
        agent2 =  np.mean(training_rewards_by_agent[1][-200:])
        agent3 =  np.mean(training_rewards_by_agent[2][-200:])

        if avg > 634:
            for i, agent in enumerate(agents):
                file_name = f'model_weights/quasi_optimal_agent{i}_{episode}.pth'
                # Replace 'q_table' with the actual attribute name if it's different
                torch.save(agent.q_network.state_dict(), f'model_weights/quasi_optimal_agent{i}_3_{episode}.pth')
                print(f"Saved {file_name}.")


    observations = env.reset()
    episode_rewards = 0  # Initialize episode rewards for each agent

    episode_rewards_by_agent = [0] * len(agents)

    while True:
        actions = [agent.select_action(observation) for agent, observation in zip(agents, observations)]
        next_observations, done, info = env.step(actions)

        tot_reward = len(info['rewarded agents']) if info['rewarded agents'] else 0
        episode_rewards += tot_reward

        for i, agent in enumerate(agents):
            agent_reward = 1 if i in info["rewarded agents"] else 0
            episode_rewards_by_agent[i] += agent_reward
            # Add experience to the agent's buffer

            episode_buffers[i][counter[i]] = (observations[i], actions[i], tot_reward, next_observations[i], done)
            counter[i] += 1
            # Check if buffer is ready for batch update
            if counter[i] == batch_size:
                states, actions, rewards, next_states, dones = zip(*episode_buffers[i])
                agent.update_q_function(states, actions, rewards, next_states, dones)
                counter[i] = 0
        

        observations = next_observations
        if done:
            break

    total_reward = episode_rewards # reward by agent
    training_rewards.append(total_reward)
    for i in range(len(agents)):
        training_rewards_by_agent[i].append(episode_rewards_by_agent[i])


if num_episodes:
    # Plot training performance
    plt.figure(figsize=(12, 5))
    plt.plot(training_rewards, label='Total Reward per Episode during Training')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Performance of 3 QLearning Agents')
    plt.legend()
    plt.show()
    plt.savefig(f'images/nonselfish/training_results.png')


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

print(f"Average Total Reward (3 Quasi-Optimal Q-Learning Agents): {average_test_reward}")
for i, average_reward in enumerate(average_test_rewards_per_agent):
    print(f"Agent {i+1} collected {average_reward} apples per episode on average.")

# Plot testing performance
plt.figure(figsize=(12, 5))
plt.plot(test_total_rewards, label='Total Reward per Episode during Testing')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Testing Performance of 3 Non-Selfish Decentralized Agents')
plt.hlines(average_test_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward')
plt.legend()
plt.savefig(f'images/nonselfish/testing_results.png')
