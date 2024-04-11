import numpy as np

# basic Agent class. Will be used as parent class for more complex agents.
class Agent():
    def __init__(self, location=(0, 0)):
        self.location = location

    def set_location(self, location):
        self.location = location

    def get_location(self):
        return self.location

    def select_action(self, observation):
        return 4 # stay still

    def _combine_state(self, observation):
        # Flatten the observation and append the agent's location
        location = np.array(self.location, dtype=np.float32)
        state = np.append(observation, location)
        return state