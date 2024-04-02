# basic Agent class. Will be used as parent class for more complex agents.
class Agent():
    def __init__(self, location=(0, 0)):
        self.location = location

    def set_location(self, location):
        self.location = location

    def select_action(self, observation):
        return 4 # stay still