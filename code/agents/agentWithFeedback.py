from .agent import *
from .qNetworks.QNetworks import *
import numpy as np
import random

class agentWithFeedback(Agent):
  def __init__(self, observation_space, action_space, location=(0,0), attention_allocation=[], attention_bound=1, learning_rate=0.001, gamma=0.95, feedback_learning_rate=0.001, feedback_importance=0.5):
    super(agentWithFeedback, self).__init__(location)
    self.observation_space = observation_space
    self.action_space = action_space

    self.learning_rate = learning_rate

    self.gamma = gamma
    
    self.feedback_learning_rate = feedback_learning_rate  # we use two different rates for selfish Q and expected feedback Q function
    self.feedback_importance = feedback_importance # how important feedback is to agent

    self.attention_allocation = attention_allocation # this is the initialization. 

    self.attention_bound = attention_bound # M in the paper

    self.alpha = 0.01 # delay sensitivity in the paper.

    if sum(attention_allocation) == 0:
       attention_allocation = [attention_bound / len(attention_allocation)] * len(attention_allocation)
    elif sum(attention_allocation) > attention_bound:
       attention_allocation *= (attention_bound / sum(attention_allocation)) # we scale it down
    else:
       attention_allocation *= (sum(attention_allocation) / attention_bound) # we scale up
    
    self.attention_allocation = attention_allocation

    # selfish Q-network
    input_size = np.prod(observation_space.shape) + 2 # grid info + location
    output_size = action_space.n

    self.q_network_selfish = QNetworkOptimalAgent(input_size, output_size, model_layer_size=600)
    self.optimizer = optim.Adam(self.q_network_selfish.parameters(), lr=learning_rate)

    # expected feedback Q Network
    self.q_network_expected_feedback = QNetworkOptimalAgent(input_size, output_size, model_layer_size=600) # state -> # expected feedback for each action
    self.optimizer2 = optim.Adam(self.q_network_expected_feedback.parameters(), lr=learning_rate)


    # Epsilon-greedy exploration
    self.epsilon = 1.0
    self.epsilon_decay = 0.99997
    self.epsilon_min = 0.03


  def p_of_interest(self, value):
     return np.e**(-value/2)
  
  def q_ability(self, value):
     return np.e**(-value/2)

  def _combine_state(self, observation, location):
        # Flatten the observation and append the agent's location

        location = np.array(location, dtype=np.float32)

        state = np.append(observation, location)
        return state

  def select_action(self, state, test=False):
    if np.random.rand() < self.epsilon and not test: # we explore
      return np.random.choice(self.action_space.n)
    else:
      # state is observation with location appended
      q_values = self.q_network_selfish(torch.Tensor(state))  # selfish q values!

      expected_rewards = self.q_network_expected_feedback(torch.Tensor(state))
    
      # Normalize q_values
      q_values /= q_values.max().clamp(min=1)  # No need for dim=1 in a 1D tensor

      # Normalize expected_rewards
      expected_rewards /= expected_rewards.max().clamp(min=1)

      # Assuming self.feedback_importance is defined
      feedback_importance = 0.5  # Example value

      # Combine and select the action with the highest value.
      # This assumes q_values and expected_rewards are now properly normalized and have the same shape.
      actions = torch.argmax(q_values + feedback_importance * expected_rewards).tolist()

      return actions
    
   
  def provide_feedback(self, states, actions, actors, optimal_models):
    # states = numpy array of states (state_size * num_states)
    # actions = action performed, actor that performed it, for each state ((num_agents-1) * num_states)
    # actors = list of actors

    states_tensor = torch.Tensor(np.array(states))  # Convert states to tensor (maybe already tensors??)
    actions_tensor = torch.tensor(actions)   # Assuming actions is numpy array and action is at index 0

    # Calculate expected rewards for all states at once
    expected_rewards = self.q_network_expected_feedback(states_tensor)
    expected_rewards /= expected_rewards.argmax(dim=1, keepdim=True).clamp(min=1)

    # Extract the value of the performed action for each state
    
    action_values = expected_rewards.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

    # Calculate probability of interest for each action
    p = self.p_of_interest((action_values - 1).abs()) 

    feedbacks = []

    for actor_index in range(len(actors)):
        attention_on_actor = self.attention_allocation[actor_index]
        if attention_on_actor > 0:
            delay_value = np.exp(-self.alpha / attention_on_actor)
            prob = delay_value * p[actor_index]  # delay * probability of interest 
        else:
            prob = 0
        feedbacks.append(random.random() < prob)  # T / F for each state
    
    B_vals = []

    for actor_index, actor in enumerate(actors):
        expected_rewards_actor = actor.q_network_expected_feedback(states_tensor[actor_index, :])
        expected_rewards_actor /= expected_rewards_actor.abs().max() # normalize

        q_values_actor = actor.q_network_selfish(states_tensor[actor_index, :])
        q_values_actor /= q_values_actor.abs().max() # normalize  # normalize

        optimal_q_values = (optimal_models[actor_index])(states_tensor[actor_index, :])
        optimal_q_values /= optimal_q_values.abs().max() # normalize 

        vals = (q_values_actor + self.feedback_importance * expected_rewards_actor) / (1 + self.feedback_importance) - optimal_q_values
        s = vals.abs().sum()
  
        q = self.q_ability(s)  # Assuming q_ability can process batches

        B_vals.append(p * q)

    return feedbacks, B_vals  # Assuming we want feedback per actor for each state


  def update_q_function(self, states, actions, rewards, next_states, dones):
    # Convert to tensors and ensure correct shapes

    #  states = torch.tensor(states, dtype=torch.float32)  # Shape: [batch_size, state_dim]
    states = torch.tensor(np.array(states), dtype=torch.float32)  # Shape: [batch_size, state_dim]


   # next_states = torch.tensor(next_states, dtype=torch.float32)  # Shape: [batch_size, state_dim]

    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # Shape: [batch_size, state_dim]

    actions = torch.tensor(actions, dtype=torch.long)  # Shape: [batch_size, 1]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # Shape: [batch_size]
    dones = torch.tensor(dones, dtype=torch.float32)  # Shape: [batch_size]

    # Get Q-values for all actions in current combined states
    q_values = self.q_network_selfish(states)

    # Select the Q-value for the action taken, using gather
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Get Q-values for all actions in next combined states using target network
    next_q_values = self.q_network_selfish(next_states)

    # Select max Q-value among next actions
    max_next_q_values = torch.max(next_q_values, dim=1)[0]

    # Compute the target Q values
    target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Gradient descent
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()

    # Decay epsilon for exploration
    self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


  def update_expected_feedback_function(self, states, actions, rewards, next_states, dones):
     
    #  states = torch.tensor(states, dtype=torch.float32)  # Shape: [batch_size, state_dim]
    states = torch.tensor(np.array(states), dtype=torch.float32)  # Shape: [batch_size, state_dim]
   # next_states = torch.tensor(next_states, dtype=torch.float32)  # Shape: [batch_size, state_dim]
    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # Shape: [batch_size, state_dim]

    actions = torch.tensor(actions, dtype=torch.long)  # Shape: [batch_size, 1]
    rewards = torch.tensor(rewards, dtype=torch.float32)  # Shape: [batch_size]
    dones = torch.tensor(dones, dtype=torch.float32)  # Shape: [batch_size]
    # Get Q-values for all actions in current combined states
    q_values = self.q_network_expected_feedback(states)

    # Select the Q-value for the action taken, using gather
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Get Q-values for all actions in next combined states using target network
    next_q_values = self.q_network_expected_feedback(next_states)
    # Select max Q-value among next actions
    max_next_q_values = torch.max(next_q_values, dim=1)[0]

    # Compute the target Q values
    target_q_values = rewards + self.gamma * max_next_q_values * (1 - dones)

    # Compute loss
    loss = nn.MSELoss()(q_values, target_q_values)

    # Gradient descent
    self.optimizer2.zero_grad()
    loss.backward()
    self.optimizer2.step()



from scipy.optimize import minimize
import numpy as np

def solve_attention_optimization_problem(B_values, alpha_value, max_attention):
  """
  attention_allocation is 
  """
  other_agents = list(B_values.keys())
  
  agent0 = min(other_agents)
  agent1 = max(other_agents)

  B_agent0 = 0
  for bval in B_values[agent0]:
     B_agent0 += bval

  B_agent1 = 0
  for bval in B_values[agent1]:
     B_agent1 += bval

  # Objective function to maximize, hence we minimize its negative
  def objective(x):
      x1, x2 = x # allocation rate
      return -(B_agent0 * np.exp(alpha_value / x1) + B_agent1 * np.exp(alpha_value / x2))

  # Constraints: x1, x2 >= 0 and x1, x2 <= M
  bounds = [(0, M), (0, M)]  # (min, max) pairs for x1 and x2

  # Initial guess
  x0 = [M/2, M/2]

  # Optimize
  result = minimize(objective, x0, bounds=bounds, method='SLSQP')

  # Optimal values of x1 and x2
  optimal_x1, optimal_x2 = result.x

  # Optimal value of the objective function
  optimal_value = -result.fun

  optimal_x1, optimal_x2, optimal_value


    
    