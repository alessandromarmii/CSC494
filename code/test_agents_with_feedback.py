from agents.agentWithFeedback import agentWithFeedback
from agents.qNetworks.QNetworks import QNetworkOptimalAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch
import time


num_agents = 3

# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [agentWithFeedback(env.observation, env.action_space, attention_allocation=[0, 0]) for _ in range(num_agents)]

# Load state dicts
model_paths = ["model_weights/selfish_multiagent_agent{}.pth".format(i) for i in range(num_agents)]
for agent, path in zip(agents, model_paths):
    agent.q_network_selfish.load_state_dict(torch.load(path))
    env.add_agent(agent) # add to environment

# PROBLEM SET-UP GLOBAL VARIABLES:

alpha = 1  # Delay Sensitivity

num_agents = 3 

rate_budget = num_agents # Max attention that an agent can allocate

input_size = np.prod(env.observation.shape) + 2

# Initialize QNetworkOptimalAgent models
optimal_models = [QNetworkOptimalAgent(input_size, env.action_space.n, 600) for _ in range(num_agents)]

# Load models
for i, model in enumerate(optimal_models):
    model.load_state_dict(torch.load(f"model_weights/quasi_optimal_agent{i}_3_2400.pth"))

# Training loop
num_episodes = 1000
training_rewards = []
batch_size = 250

episode_start_time = time.time()
time_spent = {
    'deep_copy': 0,
    'action_selection': 0,
    'env_step': 0,
    'feedback_processing': 0,
    'buffer_update': 0,
    'combining_state': 0,
    'providing_feedback': 0
}

# Initialize a buffer for each agent
episode_apples_buffers = {i: [] for i in range(len(agents))}
counter_apples_buffers = {i: 0 for i in range(len(agents))}

episode_feedback_buffers = {i: [] for i in range(len(agents))}
counter_feedback_buffers = {i: 0 for i in range(len(agents))}

for episode in range(num_episodes):

    print("episode: ", episode)
    
    if episode > 1 and episode % 20 == 0:
        print(episode, np.mean(training_rewards[-20:]))

    observation = env.reset()
    # Deep copying
    start_time = time.time()

    observations = env.reset()

    episode_rewards = 0  # Initialize episode rewards for each agent

    episode_actions = {i: [] for i in range(num_agents)} # maps agent number to list of (action, state) tuples
    episode_Bvals = {i: {j: [] for j in range(num_agents) if j != i } for i in range(num_agents)}
    
    while True:
        
        start_time = time.time()
        actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]
        time_spent['action_selection'] += time.time() - start_time

        for i in range(len(agents)):
            episode_actions[i].append((actions[i], observation))
        
        start_time = time.time()
        next_observations, done, info = env.step(actions)
        time_spent['env_step'] += time.time() - start_time

        tot_reward = len(info['rewarded agents']) if info['rewarded agents'] else 0
        episode_rewards += tot_reward

        start_time = time.time()
        time_spent['deep_copy'] += time.time() - start_time

        feed_total = np.zeros(num_agents)

        for i, agent in enumerate(agents):

            # Add experience to the agent's buffer  
            # episode_apples_buffers[i].append((observations[i], actions[i], tot_reward, next_observations[i], done))
            # counter_apples_buffers[i] += 1   
            start_time = time.time()
            feedbacks, B_values = agent.provide_feedback(observations[:i] + observations[i+1:], actions[:i] + actions[i+1:], agents[:i] + agents[i+1:], optimal_models)
            
            time_spent['providing_feedback'] += time.time() - start_time

            # feedbacks is torch tensor [T/F, T/F]. 

            for j in range(num_agents - 1):
                if j < i:
                    episode_Bvals[i][j].append(B_values[j])
                    if feedbacks[j]:
                        feed_total[j] += 1
                else: # because we skip i in the feedbacks
                    episode_Bvals[i][j + 1].append(B_values[j])
                    if feedbacks[j]:
                        feed_total[j + 1] += 1 
        
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

                start_time = time.time()
                states, past_actions, rewards, next_states, dones = zip(*episode_feedback_buffers[i])
                agent.update_expected_feedback_function(states, past_actions, rewards, next_states, dones)
                time_spent['feedback_processing'] += time.time() - start_time

                episode_feedback_buffers[i] = []  # Clear the buffer after updating
                counter_feedback_buffers[i] = 0
                pass

        observations = next_observations
        if done:
            break

    average_reward_per_agent = episode_rewards # reward by agent
    training_rewards.append(average_reward_per_agent)

    episode_end_time = time.time()
    episode_duration = episode_end_time - episode_start_time
  #  print(f"Episode {episode} duration: {episode_duration:.2f} seconds")
    for operation, duration in time_spent.items():
        percentage = (duration / episode_duration) * 100
 #       print(f"{operation}: {percentage:.2f}%")



# Plot training performance
plt.figure(figsize=(12, 5))
plt.plot(training_rewards, label='Total Reward per Episode during Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance of 3 QLearning Agents')
plt.legend()
plt.show()
