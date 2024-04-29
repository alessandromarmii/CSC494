from .qNetworks.QNetworks import *
from .agent import *

class QLearningAgentCentralized(Agent):
  def __init__(self, observation, action_space, num_agents=3, location=(0,0), learning_rate=0.0001, gamma=0.95):
    super(QLearningAgentCentralized, self).__init__(location)
    self.observation = observation
    self.action_space = action_space
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.num_agents = num_agents

    # Q-network
    input_size = num_agents * (np.prod(observation.shape) + 2) # grid info + location

    output_size = action_space.n ** num_agents

    self.q_network = CentralizedNetwork(input_size, output_size)
   # self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate, weight_decay=1e-6)
    self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
    # Epsilon-greedy exploration
    self.epsilon = 1.0
    self.epsilon_decay = 0.99995
    self.epsilon_min = 0.03


  def select_action(self, state, test=False):
    if np.random.rand() < self.epsilon and not test:
      return np.random.choice(self.action_space.n, self.num_agents)
    else:
      # state is observation with location appended
      q_values = self.q_network(torch.Tensor(state))

      max_q = torch.argmax(q_values).item()

      actions = []

      for i in range(self.num_agents):
          action = max_q // (self.action_space.n ** (self.num_agents - 1 - i))
          actions.append(action)
          max_q = max_q % (self.action_space.n ** (self.num_agents - 1 - i))

      return actions


  def update_q_function(self, states, actions, rewards, next_states, dones):
    # Convert to tensors and ensure correct shapes

    #  states = torch.tensor(states, dtype=torch.float32)  # Shape: [batch_size, num_agents * state_dim]
    states = torch.tensor(np.array(states), dtype=torch.float32)  # Shape: [batch_size, num_agents * state_dim]

    # next_states = torch.tensor(next_states, dtype=torch.float32)  # Shape: [batch_size, num_agents * state_dim]

    next_states = torch.tensor(np.array(next_states), dtype=torch.float32)  # Shape: [batch_size, num_agents * state_dim]

    actions = torch.tensor(actions, dtype=torch.long)  # Shape: [batch_size, 3]
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
