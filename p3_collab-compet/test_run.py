import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from pyvirtualdisplay import Display
from unityagents import UnityEnvironment

# initialize environment
env = UnityEnvironment(file_name="Tennis_Linux_NoVis/Tennis.x86_64")
# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]
# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)
# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)
# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# Train agents function ########################################################
from maddpg_agent import MADDPG
agents = MADDPG(num_agents=num_agents, state_size=state_size,
                                       action_size=action_size, random_seed=2)

def train(n_episodes=100, max_t=1000):
    """Multi-Agent Deep Deterministic Policy Gradiant.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
    """
    scores_window = deque(maxlen=100)  # last 100 scores
    scores_output = []

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agents.reset()
        scores = np.zeros(num_agents) # initialize all the agents' scores to 0

        for t in range(max_t):
            actions = agents.act(states, add_noise=False)  # select an action
            env_info = env.step(actions)[brain_name]  # send the action to env
            next_states = env_info.vector_observations  # get the next state
            rewards = env_info.rewards  # get the reward
            done = env_info.local_done  # see if episode has finished
            agents.step(states, actions, rewards, next_states, done)
            states = next_states
            scores += rewards
            if np.any(done):
                break

        max_score = np.max(scores)
        scores_window.append(max_score)    # save maximum score among the agents
        scores_output.append(max_score)    # save maximum score among the agents
        avg_score = np.mean(scores_window) # retrieve the agents' avg score

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if avg_score >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            for agent in agents.agents:
                actor = agent.actor_local.state_dict()
                critic = agent.critic_local.state_dict()
                id = agent.agent_id
                torch.save(actor, f'checkpoint_a{id}_actor.pth')
                torch.save(critic, f'checkpoint_a{id}_critic.pth')
            break

    return scores_output


# Train the agent, get the agents' scores ######################################
scores = train(n_episodes=2000)


# Plot the scores ##############################################################
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.plot(np.arange(len(scores)), pd.DataFrame(scores).rolling(20).mean())
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
