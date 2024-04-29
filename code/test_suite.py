import matplotlib.pyplot as plt
from environment.orchard import OrchardEnv
from agents.greedyAgent import GreedyApplePickerAgent
from agents.randomAgent import RandomAgent


def _test_greedy_agents(env, agents, num_test_episodes):
    test_total_rewards = []
    individual_agent_rewards = [[] for _ in agents]  # Initialize a list to track rewards for each agent

    for episode in range(num_test_episodes):
        observations = env.reset()  # Reset environment and get initial observations
        
        if len(agents) > 1:
            observations = observations[0]

        episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent to 0

        while True:
            actions = [agent.select_action(observations) for agent in agents]  # Get action from each agent
            observations, done , info = env.step(actions)  # Environment step based on actions
            if len(agents) > 1:
                observations = observations[0]
            for i in range(len(agents)):
                if i in info["rewarded agents"]:
                    episode_rewards[i] += 1  # Increment reward for the agent if it picked an apple

            if done:
                break

        for i, reward in enumerate(episode_rewards):
            individual_agent_rewards[i].append(reward)  # Track rewards for each agent

        total_reward = sum(episode_rewards)
        test_total_rewards.append(total_reward)

    # Calculate and print statistics for multi-agent case
    for i, rewards in enumerate(individual_agent_rewards):
        average_reward = sum(rewards) / num_test_episodes
        print(f"Agent {i+1}: Average Apples Picked: {average_reward}")

    average_total_reward = sum(map(sum, individual_agent_rewards)) / (len(agents) * num_test_episodes)
    print(f"Average Total Reward (All Agents): {average_total_reward}")

    return test_total_rewards

def _test_random_agents(env, agents, num_test_episodes):
    test_total_rewards = []
    individual_agent_rewards = [[] for _ in agents]  # Initialize a list to track rewards for each agent

    for episode in range(num_test_episodes):
        observations = env.reset()  # Reset environment and get initial observations
        episode_rewards = [0 for _ in agents]  # Initialize episode rewards for each agent to 0

        while True:
            actions = [agent.select_action() for agent in agents]  # Get action from each agent
            observations, done , info = env.step(actions)  # Environment step based on actions

            for i in range(len(agents)):
                if i in info["rewarded agents"]:
                    episode_rewards[i] += 1  # Increment reward for the agent if it picked an apple

            if done:
                break

        for i, reward in enumerate(episode_rewards):
            individual_agent_rewards[i].append(reward)  # Track rewards for each agent

        total_reward = sum(episode_rewards)
        test_total_rewards.append(total_reward)

    # Calculate and print statistics for multi-agent case
    for i, rewards in enumerate(individual_agent_rewards):
        average_reward = sum(rewards) / num_test_episodes
        print(f"Agent {i+1}: Average Apples Picked: {average_reward}")

    average_total_reward = sum(map(sum, individual_agent_rewards)) / (len(agents) * num_test_episodes)
    print(f"Average Total Reward (All Agents): {average_total_reward}")

    return test_total_rewards

def test_single_random_agent(grid_size=8):
    # Testing with a single agent
    env = OrchardEnv(agents=[])
    single_agent = RandomAgent()
    env.add_agent(single_agent)

    num_test_episodes = 500
    single_agent_rewards = _test_random_agents(env, [single_agent], num_test_episodes)
    average_single_agent_reward = sum(single_agent_rewards) / num_test_episodes
    print(f"Average Test Reward (Single Random Apple Picker Agent in 8x8 grid): {average_single_agent_reward}")

    # Plotting the rewards
    plt.figure(figsize=(12, 6))  
    episodes = list(range(1, num_test_episodes + 1))

    # Plot for single agent
    plt.plot(episodes, single_agent_rewards, label='Single Agent Reward', color='blue')
    plt.hlines(average_single_agent_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Reward (Single Agent)')
    plt.title('Single Random Agent - Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Test Reward')
    plt.legend()
    plt.show()
    return

def test_random_multiagent(grid_size=8, num_agents=3):

    # Testing with 3 agents
    env = OrchardEnv(agents=[])  # Initialize environment with 3 agents
    agents = [RandomAgent() for _ in range(num_agents)]  # Initialize agents
    for agent in agents:
        env.add_agent(agent)

    num_test_episodes = 500
    multi_agent_rewards = _test_random_agents(env, agents, num_test_episodes)
    average_multi_agent_reward = sum(multi_agent_rewards) / num_test_episodes
    print(f"Average Test Reward (3 Random Agents in 8x8 grid): {average_multi_agent_reward}")
    # Initialize the plot
    plt.figure(figsize=(12, 6))
    episodes = list(range(1, num_test_episodes + 1))
    # Plotting the rewards for multi-agent
    plt.plot(episodes, multi_agent_rewards, label=f'3 Random Agents Average Reward', color='green')
    plt.hlines(average_multi_agent_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward (3 Agents)')
    plt.title('3 Random Agents - Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Test Reward')
    plt.legend()
    # Display the plot
    plt.show()
    return

def test_single_greedy_agent(grid_size=8):
    # Testing with a single agent
    env = OrchardEnv(agents=[])
    single_agent = GreedyApplePickerAgent()
    env.add_agent(single_agent)
    num_test_episodes = 500
    single_agent_rewards = _test_greedy_agents(env, [single_agent], num_test_episodes)
    average_single_agent_reward = sum(single_agent_rewards) / num_test_episodes
    print(f"Average Test Reward (Single Greedy Apple Picker Agent in 8x8 grid): {average_single_agent_reward}")
    # Plot for single agent
    episodes = list(range(1, num_test_episodes + 1))
    # Plot for single agent
    plt.figure(figsize=(12,6))
    plt.plot(episodes, single_agent_rewards, label='Single Agent Reward', color='blue')
    plt.hlines(average_single_agent_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Reward (Single Agent)')
    plt.title('Single Greedy Agent - Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Test Reward')
    plt.legend()
    plt.show()
    return

def test_greedy_multiagent(grid_size=8):
    # Testing with 3 agents
    env = OrchardEnv(agents=[])  # Initialize environment with 3 agents
    agents = [GreedyApplePickerAgent() for _ in range(3)]  # Initialize agents
    for agent in agents:
        env.add_agent(agent)

    num_test_episodes = 500
    multi_agent_rewards = _test_greedy_agents(env, agents, num_test_episodes)
    average_multi_agent_reward = sum(multi_agent_rewards) / num_test_episodes
    print(f"Average Test Reward (3 Greedy Agents in 8x8 grid): {average_multi_agent_reward}")
    episodes = list(range(1, num_test_episodes + 1))
    plt.figure(figsize=(12,6))
    plt.plot(episodes, multi_agent_rewards, label='3 Greedy Agents Average Reward', color='green')
    plt.hlines(average_multi_agent_reward, 1, num_test_episodes, colors='red', linestyles='dashed', label='Average Total Reward (3 Agents)')
    plt.title('3 Greedy Agents - Total Reward per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Test Reward')
    plt.legend()
    plt.show()
    return

def run_tests(grid_size=8):

    print(f"To test a single random agent in {grid_size}x{grid_size} grid input 1")
    print(f"To test three random agents in an {grid_size}x{grid_size} grid input 2")
    print(f"To test a single greedy agent in {grid_size}x{grid_size} grid input 3")
    print(f"To test three greedy agents in an {grid_size}x{grid_size} grid input 4")
    print("To exit input 0")
    user_input = input("Choose desired test: ")

    user_input = int(user_input[0])
    testing = True

    while testing:
        if user_input == 0:
            testing = False
            break
        elif user_input == 1:
            test_single_random_agent(grid_size)
        elif user_input == 2:
            test_random_multiagent(grid_size)
        elif user_input == 3:
            test_single_greedy_agent(grid_size)
        elif user_input == 4:
            test_greedy_multiagent(grid_size)
        else:
            print("invalid input")
        
        print("\n \n \n")
        print(f"To test a single random agent in {grid_size}x{grid_size} grid input 1")
        print(f"To test three random agents in an {grid_size}x{grid_size} grid input 2")
        print(f"To test a single greedy agent in {grid_size}x{grid_size} grid input 3")
        print(f"To test three greedy agents in an {grid_size}x{grid_size} grid input 4")
        print("To exit input 0")
        user_input = input("Choose desired test: ")
        user_input = int(user_input[0])
        
if __name__=='__main__':
    run_tests()