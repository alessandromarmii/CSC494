from .qNetworks.QNetworks import *
from .agent import *


class SingleOptimalQLearningAgent(Agent):
  def __init__(self, observation, action_space, location=(0,0), learning_rate=0.001, gamma=0.95, model_layer_size=None):
    super(SingleOptimalQLearningAgent, self).__init__(location)
    self.observation = observation
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.gamma = gamma

    # Q-network
    input_size = np.prod(observation.shape) + 2 # grid info + location
    output_size = action_space.n
    self.q_network = QNetworkOptimalAgent(input_size, output_size, model_layer_size)
   # self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-6)
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    # Epsilon-greedy exploration
    self.epsilon = 1.0
    self.epsilon_decay = 0.99995
    self.epsilon_min = 0.03

  def select_action(self, state, test=False):
    if np.random.rand() < self.epsilon and not test:
      return np.random.choice(self.action_space.n)
    else:
      # state is observation with location appended
      q_values = self.q_network(torch.Tensor(state))
      return torch.argmax(q_values).item()

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
    q_values = self.q_network(states)

    # Select the Q-value for the action taken, using gather
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Get Q-values for all actions in next combined states using target network
    next_q_values = self.q_network(next_states)

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



class QLearningAgent(Agent):
  def __init__(self, observation, action_space, location=(0,0), learning_rate=0.001, gamma=0.95):
    super(QLearningAgent, self).__init__(location)
    self.observation = observation
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.gamma = gamma

    # Q-network
    input_size = np.prod(observation.shape) + 2 # grid info + location
    output_size = action_space.n
    self.q_network = SimpleQNetwork(input_size, output_size)
   # self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-6)
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    # Epsilon-greedy exploration
    self.epsilon = 1.0
    self.epsilon_decay = 0.99996
    self.epsilon_min = 0.03

  def _combine_state(self, observation, location):
        # Flatten the observation and append the agent's location

        location = np.array(location, dtype=np.float32)

        state = np.append(observation, location)
        return state

  def select_action(self, state, test=False):
    if np.random.rand() < self.epsilon and not test:
      return np.random.choice(self.action_space.n)
    else:
      # state is observation with location appended
      q_values = self.q_network(torch.Tensor(state))
      return torch.argmax(q_values).item()

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
    q_values = self.q_network(states)

    # Select the Q-value for the action taken, using gather
    q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

    # Get Q-values for all actions in next combined states using target network
    next_q_values = self.q_network(next_states)

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
