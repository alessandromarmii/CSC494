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
    self.tot_apples = 0
    self.current_step = 0
    self.max_steps = max_steps
    self.grid_size = grid_size
    self.random_grid = np.random.rand(self.grid_size[0], self.grid_size[1])  # this is used to pick whether or not an apple spawns in a grid point

    # action space: 4 directions and staying
    self.action_space = spaces.Discrete(5)

    # observation: 3D tensor, grid to keep track of apple and agent locations.
    # First 2 indices specify row and column.
    # Last index is indexed using Index.APPLE or Index.AGENT.

    self.observation = np.zeros((self.grid_size[0], self.grid_size[1], 2), dtype=np.float32)

    # Initialize grids to keep track of apple and agent states
    # Agents are randomly placed in the orchard to begin with
    agent_positions = (np.random.choice(grid_size[0], self.num_agents),
                        np.random.choice(grid_size[1], self.num_agents)) # randomly assign the

    # we assign gent locations
    for agent, agent_row, agent_column in zip(agents, *agent_positions):
        agent.set_location((agent_row, agent_column))
        self.observation[agent_row, agent_column, Index.AGENT] += 1


  def add_agent(self, agent):
      """
      Adds agent to agents in the environment. It adds it at random Location.
      """
      # randomly initialize location
      location = (np.random.choice(self.grid_size[0]),
                  np.random.choice(self.grid_size[1]))

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
    if len(self.agents) == 1:
       return self.agents[0]._combine_state(self.observation)

    return [agent._combine_state(self.observation) for agent in self.agents]

  def reset(self):
    # reset steps to 0.
    self.current_step = 0
    self.tot_apples = 0

    # reset the apple and agent positions
    agent_positions = (np.random.choice(self.grid_size[0], self.num_agents),
                        np.random.choice(self.grid_size[1], self.num_agents))

    self.observation = np.zeros((self.grid_size[0], self.grid_size[1], 2))

    i = 0
    for agent_row, agent_col in zip(*agent_positions):
      self.observation[agent_row, agent_col, Index.AGENT] += 1
      self.agents[i].set_location((agent_row, agent_col))
      i += 1

    return self._get_observations()

  def _move_agent(self, agent_id, action):
    agent = self.agents[agent_id]
    current_row, current_col = agent.get_location()
    self.observation[current_row, current_col, Index.AGENT] -= 1  # Decrement agent count in old cell

    # Compute new position based on the action
    if action == Action.MOVE_UP:
        new_row = max(0, current_row - 1)
    elif action == Action.MOVE_DOWN:
        new_row = min(self.grid_size[0] - 1, current_row + 1)
    else:
        new_row = current_row

    if action == Action.MOVE_LEFT:
        new_col = max(0, current_col - 1)
    elif action == Action.MOVE_RIGHT:
        new_col = min(self.grid_size[1] - 1, current_col + 1)
    else:
        new_col = current_col

    agent.set_location((new_row, new_col))  # Update agent's position
    self.observation[new_row, new_col, Index.AGENT] += 1  # Increment agent count in new cell

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
      for agent_id, agent in enumerate(self.agents):
        agent_row, agent_col = agent.get_location()
        agent_locations[(agent_row, agent_col)].append(agent_id)

      # Distribute apples in each cell.
      rewarded_agents = []
      for (row, col), agents_in_cell in agent_locations.items():
        num_apples_in_cell = int(self.observation[row, col, Index.APPLE])
        num_agents_in_cell = len(agents_in_cell)

        if num_agents_in_cell > num_apples_in_cell:
          self.tot_apples -= num_apples_in_cell
          self.observation[row, col, Index.APPLE] = 0
          agents_to_pick_apples = random.sample(agents_in_cell, num_apples_in_cell)
          rewarded_agents.extend(agents_to_pick_apples)
        else:
          self.tot_apples -= num_agents_in_cell
          self.observation[row, col, Index.APPLE] -= num_agents_in_cell
          rewarded_agents.extend(agents_in_cell)

      return set(rewarded_agents)

  def _spawn_apples(self):
    """
    Spawn apples in the orchard based on the spawn probability.
    """
    for row in range(self.grid_size[0]):
        for col in range(self.grid_size[1]):
            if self.tot_apples >= self.max_apples:
                return  # Stop if max apples reached
            
            new_apple_can_spawn = self.random_grid[row, col] < self.spawn_prob
            if new_apple_can_spawn:
                self.observation[row, col, Index.APPLE] += 1
                self.tot_apples += 1

    # Update the random grid values for next iteration
    self.random_grid = np.random.rand(self.grid_size[0], self.grid_size[1])


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
        if int(self.observation[row, col, Index.APPLE]) == 0 and int(self.observation[row, col, Index.AGENT]) == 0:
           continue
        cell_info = f"{row},{col}: " + f"[ap: {int(self.observation[row, col, Index.APPLE])}, ag: {self.observation[row, col, Index.AGENT]}]"
        print(cell_info, end=" ")
      print()