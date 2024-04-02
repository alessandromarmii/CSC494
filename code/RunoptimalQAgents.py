from agents.optimalQAgent import OptimalQLearningAgent
import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
import numpy as np
from copy import deepcopy
import torch


# Test 3 selfish (i.e. maximizing individual reward, not social) basic Q agents in 8x8 grid.

for step in ["2400"]:
    # Initialize environment and agents
    env = OrchardEnv(agents=[], max_apples=1000)
    agents = [OptimalQLearningAgent(env.observation_space, env.action_space,weight_decay=1e-6, model_layer_size=600) for _ in range(3)]

    f1 = "model_weights/quasi_optimal_agent0_3" + "_" + step + ".pth"
    f2 =  "model_weights/quasi_optimal_agent1_3" + "_" + step + ".pth"
    f3 =  "model_weights/quasi_optimal_agent2_3" + "_" + step + ".pth"

    state_dict_0 = torch.load(f1)
    state_dict_1 = torch.load(f2)
    state_dict_2 = torch.load(f3)

    agents[0].q_network.load_state_dict(state_dict_0)
    agents[1].q_network.load_state_dict(state_dict_1)
    agents[2].q_network.load_state_dict(state_dict_2)

    for agent in agents:
        env.add_agent(agent)

    # Training loop
    num_episodes = 0
    training_rewards = []
    training_rewards_by_agent  = [[] for _ in range(3)]
    batch_size = 250
    # Initialize a buffer for each agent
    episode_buffers = {i: [] for i in range(len(agents))}


    for episode in range(num_episodes):

        if episode > 1 and episode % 200 == 0:
            avg = np.mean(training_rewards[-200:])
            agent1 =  np.mean(training_rewards_by_agent[0][-200:])
            agent2 =  np.mean(training_rewards_by_agent[1][-200:])
            agent3 =  np.mean(training_rewards_by_agent[2][-200:])
                
            print(episode, avg, agent1, agent2, agent3)
            if avg > 636 and np.mean(training_rewards[-400:]) > 635:
                print("We interrupt the training and start testing.")
                break

        observation = env.reset()
        observations = [deepcopy(observation) for _ in range(len(agents))]
        
        for i in range(len(agents)):
            observations[i] = agents[i]._combine_state(observations[i], agents[i].location)

        episode_rewards = 0  # Initialize episode rewards for each agent

        episode_rewards_by_agent = [0] * len(agents)

        while True:
            actions = [agent.select_action(observation) for agent, observation in zip(agents, observations)]
            next_observation, done, info = env.step(actions)

            tot_reward = len(info['rewarded agents']) if info['rewarded agents'] else 0
            episode_rewards += tot_reward

            next_observations = [deepcopy(next_observation) for _ in range(len(agents))]

            for i, agent in enumerate(agents):
                agent_reward = 1 if i in info["rewarded agents"] else 0
                episode_rewards_by_agent[i] += agent_reward

                next_observations[i] = agent._combine_state(next_observations[i], agent.location)
                # Add experience to the agent's buffer

                episode_buffers[i].append((observations[i], actions[i], tot_reward, next_observations[i], done))

                # Check if buffer is ready for batch update
                if len(episode_buffers[i]) >= batch_size:
                    states, actions, rewards, next_states, dones = zip(*episode_buffers[i])
                    agent.update_q_function(states, actions, rewards, next_states, dones)
                    episode_buffers[i] = []  # Clear the buffer after updating

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

    testing = True
    test_count = 0
    while testing and test_count < 5:
            test_count += 1
            # Testing loop
            test_total_rewards = []
            agent_test_rewards = [[] for _ in agents]  # Initialize reward storage for each agent

            num_test_episodes = 500
            for episode in range(num_test_episodes):
                observation = env.reset()
                observations = [deepcopy(observation) for _ in range(len(agents))]
                for i in range(len(agents)):
                    observations[i] = agents[i]._combine_state(observations[i], agents[i].location)

                episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent

                while True:
                    
                    actions = [agent.select_action(observation, test=True) for agent, observation in zip(agents, observations)]
                    next_observation, done, info = env.step(actions)

                    next_observations = [deepcopy(next_observation) for _ in range(len(agents))]

                    for i, agent in enumerate(agents):
                        agent_reward = 1 if i in info["rewarded agents"] else 0

                        episode_rewards[i] += agent_reward

                        next_observations[i] = agent._combine_state(next_observations[i], agent.location)

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
            
            print("using weights at step: ", step)
            print(f"Average Total Reward (3 Quasi-Optimal Q-Learning Agents): {average_test_reward}")
            for i, average_reward in enumerate(average_test_rewards_per_agent):
                print(f"Agent {i+1} collected {average_reward} apples per episode on average.")

            if average_test_reward > 637:
                testing = False

    if testing == False: # we got > 636!
        # Plot testing performance
        plt.figure(figsize=(12, 5))
        plt.plot(test_total_rewards, label='Total Reward per Episode during Testing')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Testing Performance of 3 QLearning Agents')
        plt.legend()
        plt.show()


    """import os
    import torch

    for i, agent in enumerate(agents):
        file_name = f'model_weights/quasi_optimal_agent{i}.pth'

        # Check if file already exists
        if os.path.isfile(file_name):
            user_input = input(f"{file_name} already exists. Do you want to overwrite it? (y/n): ")
            if user_input.lower() != 'y':
                print(f"Not saving {file_name}.")
                continue

        # Replace 'q_table' with the actual attribute name if it's different
        torch.save(agent.q_network.state_dict(), f'model_weights/quasi_optimal_agent{i}.pth')
        print(f"Saved {file_name}.")"""
