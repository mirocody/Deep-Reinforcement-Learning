import numpy as np
from collections import defaultdict
import utils

class Agent:

    def __init__(self, nA=6, alpha=0.005, gamma=1.0):
        """ Initialize agent.

        Params
        ======
        - nA: number of actions available to the agent
        """
        self.nA = nA
        self.alpha = alpha
        self.gamma = gamma
        self.Q = defaultdict(lambda: np.zeros(self.nA))

    def select_action(self, state, i_episode):
        """ Given the state, select an action.

        Params
        ======
        - state: the current state of the environment

        Returns
        =======
        - action: an integer, compatible with the task's action space
        """
        Qs = self.Q[state]
        policy = utils.epsilon_greedy_policy(Qs, self.nA, i_episode)

        return np.random.choice(self.nA, p=policy)

    def step(self, state, action, reward, next_state, done):
        """ Update the agent's knowledge, using the most recently sampled tuple.

        Params
        ======
        - state: the previous state of the environment
        - action: the agent's previous choice of action
        - reward: last reward received
        - next_state: the current state of the environment
        - done: whether the episode is complete (True or False)
        """
        self.Q[state][action] = utils.update_helper(self.Q[state][action], np.max(self.Q[next_state]), self.alpha, reward, self.gamma)
