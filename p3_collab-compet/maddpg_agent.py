import torch
import numpy as np
import random
import copy
from collections import namedtuple, deque

from ddpg_agent import DDPG

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 512        # minibatch size
GAMMA = 0.99            # discount factor
UPDATE_EVERY = 2        # how often to learn and update the network weights
UPDATE_AMOUNT = 1       # amount of learning updates to the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MADDPG():
    """Interacts with and learns from the environment."""

    def __init__(self, num_agents, state_size, action_size, random_seed):
        """Initialize an Agent object.
        ipdb.set_trace()  ######### Break Point ###########

        Params
        ======
            num_agents (int) = number of agents initialized in the environment
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.agents = [DDPG(i, num_agents, state_size, action_size, random_seed)
                            for i in range(num_agents)]
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

        # Initialize time step for next learning update
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward into Replay Buffer
        states = states.reshape(1, -1)
        actions = actions.reshape(1, -1)
        next_states = next_states.reshape(1, -1)
        self.memory.add(states, actions, rewards, next_states, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # Learn, if enough samples are available in memory
            if len(self.memory) > BATCH_SIZE:
                # Number of times to learn and update the network
                for _ in range(UPDATE_AMOUNT):
                    for agent_num, _ in enumerate(self.agents):
                        experiences = self.memory.sample()
                        self.learn(agent_num, experiences, GAMMA)

    def act(self, states, add_noise=True):
        """Returns all agents' actions for their given respective states
           as per current policy."""
        actions = []
        for agent, state in zip(self.agents, states):
            action = agent.act(state, add_noise=add_noise)
            actions.append(action)
        return np.array(actions)

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, agent_num, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            agent_num: ID of the given DDPG agent
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        next_actions = []
        actions_pred = []
        states, actions, rewards, next_states, done = experiences
        states = states.reshape(-1, len(self.agents), self.state_size)
        next_states = next_states.reshape(-1, len(self.agents), self.state_size)

        for id, agent in enumerate(self.agents):
            id_tt = torch.tensor([id]).to(device)
            state = states.index_select(1, id_tt).squeeze(1)
            next_state = next_states.index_select(1, id_tt).squeeze(1)
            next_actions.append(agent.actor_target(next_state))
            actions_pred.append(agent.actor_local(state))

        next_actions = torch.cat(next_actions, dim=1).to(device)
        actions_pred = torch.cat(actions_pred, dim=1).to(device)
        agent = self.agents[agent_num]
        agent.learn(experiences, next_actions, actions_pred, gamma)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.standard_normal(len(x))
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
