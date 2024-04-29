from agents.agentWithFeedback import agentWithFeedback
from agents.agentWithFeedback import solve_attention_optimization_problem
from agents.qNetworks.QNetworks import NonSelfishQNetwork
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch
import time
import tracemalloc
import os


num_agents = 3

# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [agentWithFeedback(env.observation, env.action_space, attention_allocation=[0, 0]) for _ in range(num_agents)]

# Load state dicts
model_paths = ["model_weights/selfish_multiagent_agent{}.pth".format(i) for i in range(num_agents)]
for agent, path in zip(agents, model_paths):
    agent.q_network_selfish.load_state_dict(torch.load(path))
    agent.feedback_importance = 0.5
    env.add_agent(agent) # add to environment

# PROBLEM SET-UP GLOBAL VARIABLES:

alpha = 1  # Delay Sensitivity

num_agents = 3 

rate_budget = num_agents # Max attention that an agent can allocate

input_size = np.prod(env.observation.shape) + 2

# Initialize NonSelfishQNetwork models
optimal_models = [NonSelfishQNetwork(input_size, env.action_space.n, 600) for _ in range(num_agents)]

# Load models
for i, model in enumerate(optimal_models):
    model.load_state_dict(torch.load(f"model_weights/quasi_optimal_agent{i}_3_2400.pth"))

# Training loop
num_episodes = 50000
training_rewards = []
batch_size = 250


# Initialize a buffer for each agent
episode_apples_buffers = {i: [] for i in range(len(agents))}
counter_apples_buffers = {i: 0 for i in range(len(agents))}

episode_feedback_buffers = {i: [] for i in range(len(agents))}
counter_feedback_buffers = {i: 0 for i in range(len(agents))}


episode_actions = {i: [] for i in range(num_agents)} # maps agent number to list of (action, state) tuples
episode_Bvals = {i: {j: [] for j in range(num_agents) if j != i } for i in range(num_agents)}

# tracemalloc.start()

for episode in range(num_episodes):

    # snapshot1 = tracemalloc.take_snapshot()
    
    if episode > 1 and episode % 100 == 0:
        print(episode, np.mean(training_rewards[-100:]))
        print(agents[0].attention_allocation)
        print(agents[1].attention_allocation)
        print(agents[2].attention_allocation)
        
        for i in range(3):
            torch.save(agents[i].q_network_selfish.state_dict(), f'model_weights/agents_with_feedback/agent{i}/selfish/episode_{episode}.pth')
            torch.save(agents[i].q_network_expected_feedback.state_dict(), f'model_weights/agents_with_feedback/agent{i}/expected/episode_{episode}.pth')
        

    observations = env.reset()

    episode_rewards = 0  # Initialize episode rewards for each agent

    for i in range(num_agents):
        episode_actions[i] = []
        for j in range(num_agents):
            if j != i:
                episode_Bvals[i][j] = [] 

    while True:
        
        actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]

        for i in range(len(agents)):
            episode_actions[i].append((actions[i], observations[i]))
        
        next_observations, done, info = env.step(actions)

        tot_reward = len(info['rewarded agents']) if info['rewarded agents'] else 0 # number of agents that picked up an apple
        episode_rewards += tot_reward

        feed_total = np.zeros(num_agents)

        for i, agent in enumerate(agents):

            # Add experience to the agent's buffer  
            episode_apples_buffers[i].append((observations[i], actions[i], tot_reward, next_observations[i], done))
            counter_apples_buffers[i] += 1   
            feedbacks, B_values = agent.provide_feedback(observations[:i] + observations[i+1:], actions[:i] + actions[i+1:], agents[:i] + agents[i+1:], optimal_models)
            
            # feedbacks is torch tensor [T/F, T/F]. 
            for j in range(num_agents - 1):
                index = j if j < i else j + 1  # we skip i in the feedbacks
                episode_Bvals[i][index].append(B_values[j])
                if feedbacks[j]:
                    feed_total[index] += 1

            # Check if buffer is ready for batch update
            if counter_apples_buffers[i] >= batch_size:
                states, past_actions, rewards, next_states, dones = zip(*episode_apples_buffers[i])
                agent.update_q_function(states, past_actions, rewards, next_states, dones)
                episode_apples_buffers[i] = []  # Clear the buffer after updating
                counter_apples_buffers[i] = 0   # Reset counter

        for i in range(num_agents):
            episode_feedback_buffers[i].append((observations[i], actions[i], feed_total[i], next_observations[i], done))
            counter_feedback_buffers[i] += 1

            if counter_feedback_buffers[i] >= batch_size: # Check if buffer is ready for batch update

                states, past_actions, rewards, next_states, dones = zip(*episode_feedback_buffers[i])
                agent.update_expected_feedback_function(states, past_actions, rewards, next_states, dones)

                episode_feedback_buffers[i] = []  # Clear the buffer after updating
                counter_feedback_buffers[i] = 0
                pass

        observations = next_observations
        if done:
            break

    training_rewards.append(episode_rewards) # total episode reward

    for i, agent in enumerate(agents):
        agent.attention_allocation = solve_attention_optimization_problem(episode_Bvals[i], alpha=alpha, rate_budget=rate_budget )


# Plot training performance
plt.figure(figsize=(12, 5))
plt.plot(training_rewards, label='Total Reward per Episode during Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance of 3 QLearning Agents')
plt.legend()
plt.show()
