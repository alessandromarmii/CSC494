import numpy as np
from .agent import Agent
import random


class RandomAgent(Agent):
    def __init__(self, location=(0,0)):
        super(RandomAgent, self).__init__(location)
        self.location = location

    def select_action(self):
        return random.choice([0,1,2,3,4])
