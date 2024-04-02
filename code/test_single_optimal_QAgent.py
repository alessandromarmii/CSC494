from agents.basicQAgent import QLearningAgent
from agents.optimalQAgent import OptimalQLearningAgent
from agents.singleOptimalQAgent import SingleOptimalQLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
import threading
import torch 

"""
Test selfish (i.e. maximizing individual reward, not social) individual basic Q agent in 8x8 grid.

A greedy agent is optimal in this scenario. 
The closer the Q-agent's performance is to the performance of an individual greedy agent, the more effective the learning.
"""

# Function to plot rewards
def plot_rewards(rewards, title):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Reward per episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.hlines(np.mean(rewards), 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward')
    plt.title(title)
    plt.legend()
    plt.show()

# Test selfish QLearningAgent
env = OrchardEnv(agents=[], max_apples=1000)
agent = SingleOptimalQLearningAgent(env.observation_space, env.action_space, model_layer_size=600)
state_dict = torch.load("model_weights/single_optimal_agent.pth")
agent.q_network.load_state_dict(state_dict)

agent.learning_rate = 0.0001 # smaller by 3x
agent.epsilon_decay = 0.9999
agent.epsilon = 0.1
agent.gamma = 0.95

env.add_agent(agent)

stop_training = False  # Global flag to control when to stop training

def check_for_stop():
    global stop_training
    while True:
        user_input = input()
        if  "stop" in user_input.lower():
            stop_training = True
            break

# Start a thread that listens for the "stop" command
stop_thread = threading.Thread(target=check_for_stop)
stop_thread.start()

# Training loop
# num_episodes = 50000
num_episodes = 0
training_rewards = []
batch_size = 250

for episode in range(num_episodes):
    
    if episode > 1 and episode % 200 == 0:
        if stop_training:  # Check the global flag
            print("Stop command received, interrupting the training.")
            break
        avg = np.mean(training_rewards[-200:])
        print(episode, avg)
        if avg > 620 and np.mean(training_rewards[-400:]) > 620:
            print("We interrupt the training and start testing.")
            break
        
    observation = env.reset()

    observation = agent._combine_state(observation, agent.location)

    episode_reward = 0
    
    episode_buffer = []  # Buffer to store experiences from the current episode

    batch_counter = 0

    while True:
        action = agent.select_action(observation)

        new_observation, done, info = env.step([action])

        new_observation = agent._combine_state(new_observation, agent.location)

        agent_reward = 1 if 0 in info["rewarded agents"] else 0
        
        # Store the experience in the episode buffer
        episode_buffer.append((observation, action, agent_reward, new_observation, done))
        
        batch_counter += 1

        observation = new_observation

        episode_reward += agent_reward
        
        if done:
            break

        # If the episode buffer has experiences, use them to update the Q-function
        if batch_counter >= batch_size:
            # Unpack the episode's experiences
            states, actions, rewards, next_states, dones = zip(*episode_buffer)
            
            # Update the Q-function with the entire batch from the episode
            agent.update_q_function(states, actions, rewards, next_states, dones)

            episode_buffer = []
            batch_counter = 0

    training_rewards.append(episode_reward)


# stop_thread.join()
# Plot training performance
if num_episodes:
    plot_rewards(training_rewards, 'Training Performance of Single QLearningAgent')

env = OrchardEnv(agents=[], max_apples=1000)
env.add_agent(agent)

testing = True
# Testing loop
while testing:
        num_test_episodes = 500
        test_rewards = []

        for episode in range(num_test_episodes):
            observation = env.reset()
            observation = agent._combine_state(observation, agent.location)
            episode_reward = 0
            i = 0
            while True:
                i += 1
                action = agent.select_action(observation, test=True)
                observation, done, info = env.step([action])
                observation = agent._combine_state(observation, agent.location)
                episode_reward += 1 if 0 in info["rewarded agents"] else 0
                if done:
                    break

            test_rewards.append(episode_reward)

        # Print average test performance
        average_test_reward = sum(test_rewards) / num_test_episodes
        print(f"Average Test Reward of a single basic and selfish QLearning agent: {average_test_reward}")
        if average_test_reward > 623:
            testing = False

# Plot testing performance
plot_rewards(test_rewards, 'Testing Performance of Single QLearningAgent')

import os
import torch

file_name = f'model_weights/single_optimal_agent.pth'
# Check if file already exists
if os.path.isfile(file_name):
    user_input = input(f"{file_name} already exists. Do you want to overwrite it? (y/n): ")
    if user_input.lower() != 'y':
        print(f"Not saving {file_name}.")
    else:
        # Replace 'q_table' with the actual attribute name if it's different
        torch.save(agent.q_network.state_dict(), f'model_weights/single_optimal_agent_2.pth')
        print(f"Saved {file_name}.")
else:
    # Replace 'q_table' with the actual attribute name if it's different
    torch.save(agent.q_network.state_dict(), f'model_weights/single_optimal_agent_2.pth')
    print(f"Saved {file_name}.")

stop_thread.join()