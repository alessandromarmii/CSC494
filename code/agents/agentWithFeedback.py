from .agent import *
from .qNetworks.QNetworks import *
import numpy as np
import random
from scipy.optimize import minimize
import time

class agentWithFeedback(Agent):
  def __init__(self, observation, action_space, location=(0,0), attention_allocation=[], attention_bound=1, learning_rate=0.001, gamma=0.95, feedback_learning_rate=0.001, feedback_importance=0.5):
   # attention_allocation -> has size num_agents in the system - 1 (this agent itself)
    
    super(agentWithFeedback, self).__init__(location)
    self.observation = observation
    self.action_space = action_space

    self.learning_rate = learning_rate

    self.gamma = gamma
    
    self.feedback_learning_rate = feedback_learning_rate  # we use two different rates for selfish Q and expected feedback Q function
    self.feedback_importance = feedback_importance # how important feedback is to agent

    self.attention_allocation = attention_allocation # this is the initialization. 

    self.attention_bound = attention_bound # M in the paper

    self.alpha = 0.01 # delay sensitivity in the paper.

    if sum(attention_allocation) == 0:
       attention_allocation = [attention_bound / len(attention_allocation)] * len(attention_allocation) # same to everyone
    elif sum(attention_allocation) > attention_bound:
       attention_allocation *= (attention_bound / sum(attention_allocation)) # we scale it down
    else:
       attention_allocation *= (sum(attention_allocation) / attention_bound) # we scale up
    
    self.attention_allocation = torch.Tensor(attention_allocation)

    # selfish Q-network
    input_size = np.prod(observation.shape) + 2 # grid info + location
    output_size = action_space.n

    self.q_network_selfish = QNetworkOptimalAgent(input_size, output_size, model_layer_size=600)
    self.optimizer = optim.Adam(self.q_network_selfish.parameters(), lr=learning_rate)

    # expected feedback Q Network
    self.q_network_expected_feedback = QNetworkOptimalAgent(input_size, output_size, model_layer_size=600) # state -> # expected feedback for each action
    self.optimizer2 = optim.Adam(self.q_network_expected_feedback.parameters(), lr=learning_rate)


    # Epsilon-greedy exploration
    self.epsilon = 0.5
    self.epsilon_decay = 0.9999
    self.epsilon_min = 0.03


  def p_of_interest(self, value, normalized=False):
     # if normalized is true, we receive values in the range 0,1 and we use the function -2x+2 (on the interval [0,1]):
     if normalized:
        return - 2 * value + 2

     return np.e**(-value/2)
  
  def q_ability(self, value):
     return np.e**(-value/2)


  def select_action(self, state, test=False):
    if np.random.rand() < self.epsilon and not test: # we explore
      return np.random.choice(self.action_space.n)
    else:
      state = torch.Tensor(state)
      # state is observation with location appended
      q_values = self.q_network_selfish(state).detach() # selfish q values!

      expected_rewards = self.q_network_expected_feedback(state).detach()
    
      # Normalize q_values
      q_values /= q_values.max().clamp(min=1)  # No need for dim=1 in a 1D tensor

      # Normalize expected_rewards
      expected_rewards /= expected_rewards.max().clamp(min=1)

      # Combine and select the action with the highest value.
      # This assumes q_values and expected_rewards are now properly normalized and have the same shape.
      actions = torch.argmax(q_values + self.feedback_importance * expected_rewards).tolist()

      return actions
    

  def provide_feedback(self, states, actions, actors, optimal_models):
      # Assume states are already tensors. If not, convert outside this function to avoid repeated conversion.

      # returns: feedbacks List[Bool] of lenght len(actors) (it corresponds to whether or not we provide feedback to each actor) 
      # returns: B_vals -> list of floats that corresponds to the value that this agent assigns to the given actions

      states_np = np.array(states)  # This concatenates the list of arrays into a single numpy array

      states_tensor = torch.tensor(states_np, dtype=torch.float32)  # Now converting to tensor is efficient

      actions_tensor = torch.tensor(actions, dtype=torch.long)

      # Calculate expected rewards for all states at once and normalize
      expected_rewards = self.q_network_expected_feedback(states_tensor).detach()

      # expected_rewards /= expected_rewards.argmax(dim=1, keepdim=True).clamp(min=1) # normalize

      # Extract the expected reward assigned to the performed action for each state
      action_values = expected_rewards.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

      # Calculate probability of interest for each action
      p_interest = self.p_of_interest((action_values - 1).abs())

      # Vectorize feedback calculation
      delay_values = torch.exp(-self.alpha / self.attention_allocation)
      probs = delay_values * p_interest
      feedbacks = torch.rand(len(actors)) < probs

      # calculate B value
      B_vals = []
      for actor_index, actor in enumerate(actors):
         q_network_expected_feedback = actor.q_network_expected_feedback(states_tensor[actor_index]).detach()
         q_network_expected_feedback.div_(q_network_expected_feedback.abs().max()) # Normalize

         q_values_actor = actor.q_network_selfish(states_tensor[actor_index]).detach() # this needs to be changed once i start training
         q_values_actor.div_(q_values_actor.abs().max()) # Normalize

         optimal_q_values = optimal_models[actor_index](states_tensor[actor_index]).detach()
         optimal_q_values.div_(optimal_q_values.abs().max()) # Normalize

         vals = (q_values_actor + self.feedback_importance * q_network_expected_feedback) / (1 + self.feedback_importance) - optimal_q_values
         # print("vals: ", vals)
         
         s = vals.abs().sum()
         # print("sum: ", s)

         # print("self.q_ability(s): ", self.q_ability(s))
         # print("p_interest: ", p_interest)
         # print("self.q_ability(s) * p_interest: ", self.q_ability(s) * p_interest)

         B_vals.append(self.q_ability(s) * p_interest[actor_index])

      # print("feedbacks: ", feedbacks.tolist())
      # print("B_vals: ", torch.stack(B_vals).tolist())
      # Convert feedbacks and B_vals to desired format if needed
      return feedbacks.tolist(), torch.stack(B_vals).tolist()
   

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



def solve_attention_optimization_problem(B_values, alpha, rate_budget):
   """
   B_values for agent i is dictionary containing [0,1,..,n] except i as keys and the corresponding set of b_values as values 
   """
   """print("B_Values: ", B_values)
   print("alpha: ", alpha)
   print("rate_budget: ", rate_budget)"""

   other_agents = list(B_values.keys())
   
   agent0 = min(other_agents)
   agent1 = max(other_agents)

   B_agent0 = sum(B_values[agent0])
   B_agent1 = sum(B_values[agent1])

   epsilon = 0.01     # we use this for numerical stability

   # Objective function to maximize, hence we minimize its negative
   def objective(x):
    x0, x1 = x
    agent_0_val = B_agent0 * np.exp(alpha / max(x0, epsilon))
    agent_1_val = B_agent1 * np.exp(alpha / max(x1, epsilon))
    return -(agent_0_val + agent_1_val)

   # Constraint: sum of x0 and x1 must equal rate_budget
   def constraint(x):
      return np.sum(x) - rate_budget

   # Constraints: x0, x1 >= 0 and x0, x1 <= M
   bounds = [(epsilon, rate_budget), (epsilon, rate_budget)]  # (min, max) pairs for x1 and x2

   # Define the constraint as a dictionary
   con = {'type': 'eq', 'fun': constraint}

   # Initial guess
   x = [ rate_budget /2, rate_budget /2]

   # Optimize with added constraint
   result = minimize(objective, x, method='SLSQP', bounds=bounds, constraints=con)

   # Optimal values of x0 and x1
   optimal_x0, optimal_x1 = result.x

   # Optimal value of the objective function
   # optimal_value = -result.fun # used for debug

   return torch.Tensor([optimal_x0, optimal_x1])

