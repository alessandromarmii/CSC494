from agents.agentWithFeedback import agentWithFeedback
from agents.qNetworks.QNetworks import QNetworkOptimalAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch






# Initialize environment and agents
env = OrchardEnv(agents=[], max_apples=1000)
agents = [agentWithFeedback(env.observation_space, env.action_space, attention_allocation=[0, 0]) for _ in range(3)]

selfish_state_dict_0 = torch.load("model_weights/selfish_multiagent_agent0.pth")
selfish_state_dict_1 = torch.load("model_weights/selfish_multiagent_agent1.pth")
selfish_state_dict_2 = torch.load("model_weights/selfish_multiagent_agent2.pth")

agents[0].q_network_selfish.load_state_dict(selfish_state_dict_0)
agents[1].q_network_selfish.load_state_dict(selfish_state_dict_1)
agents[2].q_network_selfish.load_state_dict(selfish_state_dict_2)

for agent in agents:
    env.add_agent(agent)

# PROBLEM SET-UP GLOBAL VARIABLES:

alpha = 1  # Delay Sensitivity

num_agents = 3 

rate_budget = num_agents # Max attention that an agent can allocate

input_size = np.prod(env.observation_space.shape) + 2

optimal_models = [QNetworkOptimalAgent(input_size, env.action_space.n, 600) for i in range(num_agents)]

state_dict_0 = torch.load("model_weights/quasi_optimal_agent0_3_2400.pth")
state_dict_1 = torch.load("model_weights/quasi_optimal_agent1_3_2400.pth")
state_dict_2 = torch.load("model_weights/quasi_optimal_agent2_3_2400.pth")

optimal_models[0].load_state_dict(state_dict_0)
optimal_models[1].load_state_dict(state_dict_1)
optimal_models[2].load_state_dict(state_dict_2)

# Training loop
num_episodes = 1000
training_rewards = []
batch_size = 250

# Initialize a buffer for each agent
episode_apples_buffers = {i: [] for i in range(len(agents))}
episode_feedback_buffers = {i: [] for i in range(len(agents))}

for episode in range(num_episodes):

    print("episode: ", episode)
    
    if episode > 1 and episode % 20 == 0:
        print(episode, np.mean(training_rewards[-20:]))

    observation = env.reset()
    observations = [deepcopy(observation) for _ in range(len(agents))]
    
    for i in range(len(agents)):
        observations[i] = agents[i]._combine_state(observations[i], agents[i].location)

    episode_rewards = 0  # Initialize episode rewards for each agent

    episode_actions = {i: [] for i in range(num_agents)} # maps agent number to list of (action, state) tuples
    episode_Bvals = {i: {j: [] for j in range(num_agents) if j != i } for i in range(num_agents)}
    
    while True:
        
        actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]

        for i in range(len(agents)):
            episode_actions[i].append((actions[i], observation))
        
        next_observation, done, info = env.step(actions)

        tot_reward = len(info['rewarded agents']) if info['rewarded agents'] else 0
        episode_rewards += tot_reward

        next_observations = [deepcopy(next_observation) for _ in range(num_agents)]

        feed_total = torch.zeros(num_agents, 1)

        for i, agent in enumerate(agents):

            next_observations[i] = agent._combine_state(next_observations[i], agent.location)
            # Add experience to the agent's buffer

            episode_apples_buffers[i].append((observations[i], actions[i], tot_reward, next_observations[i], done))

            feedbacks, B_values = agent.provide_feedback(observations[:i] + observations[i+1:], actions[:i] + actions[i+1:], agents[:i] + agents[i+1:], optimal_models)
        
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
            if len(episode_apples_buffers[i]) >= batch_size:
                states, past_actions, rewards, next_states, dones = zip(*episode_apples_buffers[i])
                agent.update_q_function(states, past_actions, rewards, next_states, dones)
                episode_apples_buffers[i] = []  # Clear the buffer after updating
        
        for i in range(num_agents):
            episode_feedback_buffers[i].append((observations[i], actions[i], feed_total[i], next_observations[i], done))

            if len(episode_feedback_buffers[i]) >= batch_size: # Check if buffer is ready for batch update
                states, past_actions, rewards, next_states, dones = zip(*episode_feedback_buffers[i])
                agent.update_expected_feedback_function(states, past_actions, rewards, next_states, dones)
                episode_feedback_buffers[i] = []  # Clear the buffer after updating
                pass

        observations = next_observations
        if done:
            break

    average_reward_per_agent = episode_rewards # reward by agent
    training_rewards.append(average_reward_per_agent)


# Plot training performance
plt.figure(figsize=(12, 5))
plt.plot(training_rewards, label='Total Reward per Episode during Training')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.title('Training Performance of 3 QLearning Agents')
plt.legend()
plt.show()
