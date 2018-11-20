import sys
import gym
import numpy as np

def policy_epsilon_greedy(env, Qs, i_episode, epsilon=None):
    """epsilon greedy policy
        env: Gym environment
        s: state
        i_episode: index of current episode -> epsilon should reduce when i increase
        eps: customized epsilon
    """
    epsilon = 1 / i_episode if epsilon is None else epsilon
    policy = np.ones(env.nA) * epsilon / env.nA
    policy[np.argmax(Qs)] = 1 - epsilon + (epsilon /env.nA) #np.argmax(s) means action a maximizes Q(s,a)

    return policy
