import numpy as np


def epsilon_greedy_policy(Qs, nA, i_episode, epsilon=None):
    """epsilon greedy policy
        env: Gym environment
        s: state
        i_episode: index of current episode -> epsilon should reduce when i increase
        eps: customized epsilon
    """
    epsilon = 1 / i_episode if epsilon is None else epsilon
    policy = np.ones(nA) * epsilon / nA
    policy[np.argmax(Qs)] = 1 - epsilon + (epsilon /nA) #np.argmax(s) means action a maximizes Q(s,a)

    return policy

def update_helper(Qsa, Qsa_next, alpha, reward, gamma):
    """helper function to update Q table
    Args:
        Qsa: Q table value at State s, Action a
        Qsa_next: Q table value at State s_next, Action a_next

    Return:
        Updated Qsa
    """
    return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)
