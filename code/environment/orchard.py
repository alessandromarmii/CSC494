from gym import Env, spaces
import numpy as np
import random
from collections import defaultdict
from typing import List, Optional
from enum import Enum
from agents.agent import Agent


class Action:
  MOVE_UP = 0
  MOVE_DOWN = 1
  MOVE_LEFT = 2
  MOVE_RIGHT = 3
  STAY = 4

class Index:
  APPLE = 0
  AGENT = 1


class OrchardEnv(Env):
  def __init__(self, grid_size=(8, 8), spawn_prob=0.01,
               agents: Optional[List[Agent]]=[Agent()], max_apples=1000,
               max_steps=1000):
    """
    Constructor to create an orchard environment for multiple agents to
    work with.
    """
    # environment configuration
    super(OrchardEnv, self).__init__()
    self.grid_side = grid_size
    self.spawn_prob = spawn_prob
    self.agents = agents
    self.num_agents = len(agents)
    self.max_apples = max_apples
    self.current_step = 0
    self.max_steps = max_steps
    self.grid_size = grid_size
    # self.apples_picked = 0

    # action space: 4 directions and staying
    self.action_space = spaces.Discrete(5)

    # observation space: 3D tensor
    # First 2 indices specify row and column.
    # Last index is indexed using Index.APPLE or Index.AGENT.

    self.observation_space = spaces.Box(
      low=np.zeros((grid_size[0], grid_size[1], 2), dtype=np.float32),
      high=np.stack([np.full((grid_size[0], grid_size[1]), max_apples),
                   np.full((grid_size[0], grid_size[1]), self.num_agents)], axis=-1),
      shape=(grid_size[0], grid_size[1], 2),
      dtype=np.float32
    )

    # Initialize grids to keep track of apple and agent states
    # Agents are randomly placed in the orchard to begin with
    self.apples = np.zeros((grid_size[0], grid_size[1]))
    self.agent_positions = (np.random.choice(grid_size[0], self.num_agents),
                        np.random.choice(grid_size[1], self.num_agents))

    # we assign gent locations
    for agent, agent_row, agent_column in zip(agents, *self.agent_positions):
        agent.set_location((agent_row, agent_column))


  def add_agent(self, agent):
      """
      Adds agent to agents in the environment. It adds it at random Location.
      """
      # randomly initialize location
      location = (np.random.choice(self.grid_size[0]),
                  np.random.choice(self.grid_size[1]))

      np.append(self.agent_positions[0], location[0])
      np.append(self.agent_positions[1], location[1])

      agent.set_location(location)

      self.agents.append(agent)
      self.num_agents += 1
      return



  def _is_done(self):
    """
    Returns whether we have completed max_steps many steps.
    """
    return self.current_step >= self.max_steps

  def _get_observations(self):
    """
    Returns the observation generated from the orchard with dimensions
    grid_size[0] X grid_size[1] X 2.

    Tells you number of agents and apples for each point in the grid.
    """
    # Initialize empty observation matrix.
    observation = np.zeros((self.grid_size[0], self.grid_size[1], 2),
                            dtype=np.float32)

    # Populate observation of number of apples.
    observation[:, :, Index.APPLE] = self.apples

    # Populate observation of number of agents
    for agent_row, agent_col in zip(*self.agent_positions):
      observation[agent_row, agent_col, Index.AGENT] += 1

    return observation

  def reset(self):
    # reset steps to 0.
    self.current_step = 0

    # reset the apple and agent positions
    self.apples = np.zeros((self.grid_size[0], self.grid_size[1]))
    self.agent_positions = (np.random.choice(self.grid_size[0], self.num_agents),
                        np.random.choice(self.grid_size[1], self.num_agents))


    observation = np.zeros((self.grid_size[0], self.grid_size[1], 2), dtype=np.float32)

    # Populate observation of the number of apples.
    observation[:, :, Index.APPLE] = self.apples

    # Populate observation of the number of agents
    for agent_row, agent_col in zip(*self.agent_positions):
      observation[agent_row, agent_col, Index.AGENT] += 1
    
    return observation

  def _move_agent(self, agent_id, action):
    """
    Update the position of an agent.
    """

    current_row = self.agent_positions[0][agent_id]
    current_col = self.agent_positions[1][agent_id]

    if action == Action.MOVE_UP:      
      new_row = max(0, current_row - 1)
      new_col = current_col
      self.agent_positions[0][agent_id] = new_row
    elif action == Action.MOVE_DOWN:
      new_row = min(self.grid_size[0] - 1, current_row + 1)
      new_col = current_col
      self.agent_positions[0][agent_id] = new_row
    elif action == Action.MOVE_LEFT:
      new_col = max(0, current_col - 1)
      new_row = current_row
      self.agent_positions[1][agent_id] = new_col
    elif action == Action.MOVE_RIGHT:
      new_col = min(self.grid_size[1] - 1, current_col + 1)
      new_row = current_row
      self.agent_positions[1][agent_id] = new_col
    else:
      new_row = current_row
      new_col = current_col

    self.agents[agent_id].set_location((new_row, new_col))

  def _move_agents(self, actions):
    """
    Update the positions of all agents.
    """
    for agent_id, action in enumerate(actions):
      self._move_agent(agent_id, action)

  def _pickup_apples(self):
      """
      Pick up apples in each cell based on the number of agents
      and apples present.

      Returns a set of agents that have picked up an apple.
      """
      # Group agents by their locations
      agent_locations = defaultdict(list)
      for agent_id, (agent_row, agent_col) in enumerate(zip(*self.agent_positions)):
        agent_locations[(agent_row, agent_col)].append(agent_id)


      # Distribute apples in each cell.
      rewarded_agents = []
      for (row, col), agents_in_cell in agent_locations.items():
        num_apples_in_cell = int(self.apples[row, col])
        num_agents_in_cell = len(agents_in_cell)

        if num_agents_in_cell > num_apples_in_cell:
          self.apples[row, col] = 0
          agents_to_pick_apples = random.sample(agents_in_cell, num_apples_in_cell)
          rewarded_agents.extend(agents_to_pick_apples)
        else:
          self.apples[row, col] -= num_agents_in_cell
          rewarded_agents.extend(agents_in_cell)

      return set(rewarded_agents)

  def _spawn_apples(self):
    """
    Spawn apples in the orchard based on the spawn probability.
    """
    tot_apples = np.sum(self.apples)

    for row in range(self.grid_size[0]):
      for col in range(self.grid_size[1]):
        new_apple_can_spawn = random.random() < self.spawn_prob
        num_apples_below_max = tot_apples < self.max_apples # we haven't reached max number of apples in grid.
        if new_apple_can_spawn and num_apples_below_max:
          self.apples[row, col] += 1
          tot_apples += 1
        elif not num_apples_below_max:
          return # we reached max
        """if new_apple_can_spawn:
          self.apples[row, col] += 1
          tot_apples += 1"""
          


  def step(self, actions):
    """
    Performs 1 step in the orchard environment.
    """
    self._move_agents(actions)
    rewarded_agents = self._pickup_apples()
    self._spawn_apples()

    self.current_step += 1

    done = self._is_done()
    observations = self._get_observations()
    info = {
      "rewarded agents": rewarded_agents
    }

    return observations, done, info

  def render(self):
    for row in range(self.grid_size[0]):
      for col in range(self.grid_size[1]):
        cell_info = f"[apples: {int(self.apples[row, col])}, agents: {np.sum((self.agent_positions[0] == row) & (self.agent_positions[1] == col))}]"
        print(cell_info, end=" ")
      print()

