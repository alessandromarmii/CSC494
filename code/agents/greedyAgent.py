import numpy as np
from .agent import Agent

class GreedyApplePickerAgent(Agent):
    def __init__(self, location=(0,0)):
        super(GreedyApplePickerAgent, self).__init__(location)
        self.location = location

    def select_action(self, observation):
        
        observation = observation[:-2]
        # Reshape the array into 8x8x2
        observation = observation.reshape((8, 8, 2))

        # Extract relevant information from the observation
        apples = observation[:, :, 0]  # Assuming the first channel represents the number of apples

        # Get the coordinates of the agent
        agent_row, agent_col = np.where(observation[:, :, 1] > 0) # this assumes there is only one agent

        if len(agent_row) == 0:
            # No agent found in the observation
            return 0  # Move Up as a default action

        agent_row, agent_col = agent_row[0], agent_col[0]

        # Find the coordinates of all apples in the observation
        apple_rows, apple_cols = np.where(apples > 0)

        if len(apple_rows) == 0:
            # No apples found, stay in the same spot
            return 0  # Stay in the same spot

        # Calculate distances to all apples
        distances = np.abs(apple_rows - agent_row) + np.abs(apple_cols - agent_col)

        # Find the index of the closest apple
        closest_apple_idx = np.argmin(distances)

        # Calculate the direction towards the closest apple
        move_up = apple_rows[closest_apple_idx] < agent_row
        move_down = apple_rows[closest_apple_idx] > agent_row
        move_left = apple_cols[closest_apple_idx] < agent_col
        move_right = apple_cols[closest_apple_idx] > agent_col

        # Choose the action based on the direction towards the closest apple
        if move_up:
            return 0  # Move Up
        elif move_down:
            return 1  # Move Down
        elif move_left:
            return 2  # Move Left
        elif move_right:
            return 3  # Move Right
        else:
            return 4  # Stay in the same spot
