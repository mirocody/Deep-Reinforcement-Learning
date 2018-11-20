import sys
sys.path.append('../')
import gym
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
# %matplotlib inline
import check_test
from plot_utils import plot_values
from Utils.util import policy_epsilon_greedy as policy

env = gym.make('CliffWalking-v0')


def update_helper(Qsa, Qsa_next, alpha, reward, gamma):
    """helper function to update Q table
    Args:
        Qsa: Q table value at State s, Action a
        Qsa_next: Q table value at State s_next, Action a_next

    Return:
        Updated Qsa
    """
    return Qsa + alpha * (reward + gamma * Qsa_next - Qsa)



def sarsa(env=gym.make('CliffWalking-v0'), num_episodes=5000, alpha=0.01, gamma=1.0):
    """Saras Algothrim to calculate Q-table.

    Args:
        env: Gym Cliff Walking environment.
        num_episodes: the number of episodes you want to run. Default value is 1000.
        alpha: step size parameter used to update step. Default value is 0.01.
        gamma: discount rate. Default is 1.

    Return:
        Q: Dictionary Q-Table .
    """
    plot_every = 100
    Q = defaultdict(lambda: np.zeros(env.nA))
    # Q = np.zeros((env.observation_space.n, env.action_space.n))
    # Use scores and tmp_scores to track rewards update.
    scores = deque(maxlen=num_episodes)
    tmp_scores = deque(maxlen=plot_every)
    for i in range(1, num_episodes + 1):
        if i % plot_every == 0:
            print("\rEpisode {}/{}".format(i, num_episodes), end="")
            sys.stdout.flush()

        state = env.reset()
        score = 0
        done = False
        probs = policy(env, Q[state], i) # TODO: change policy e.g. greedy policy -> Done
        action = np.random.choice(env.nA, p=probs)
        while not done: # TODO: This could be relatively slow if there are many states. Can limite time steps here.

            next_state, reward, done, info = env.step(action)
            score += reward
            probs = policy(env, Q[next_state], i) # TODO: change policy e.g. greedy policy -> Done
            next_action = np.random.choice(env.nA, p=probs)
            # Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            # use helper function instaed
            Q[state][action] = update_helper(Q[state][action], Q[next_state][next_action], alpha, reward, gamma)
            action = next_action
            state = next_state
            if (done):
                tmp_scores.append(score)
                Q[state][action] = update_helper(Q[state][action], 0, alpha, reward, gamma)
        if (i % plot_every == 0):
            scores.append(np.mean(tmp_scores))
        plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    return Q

def test_sarsa(): # Credit to Udacity
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsa = sarsa(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsa = np.array([np.argmax(Q_sarsa[key]) if key in Q_sarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_sarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsa)

    # plot the estimated optimal state-value function
    V_sarsa = ([np.max(Q_sarsa[key]) if key in Q_sarsa else 0 for key in np.arange(48)])
    plot_values(V_sarsa)


def q_learning(env=gym.make('CliffWalking-v0'), num_episodes=5000, alpha=0.01, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i in range(1, num_episodes+1):
        # monitor progress
        if i % 100 == 0:
            print("\rEpisode {}/{}".format(i, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        score = 0
        done = False
        while not done:
            probs = policy(env, Q[state], i)
            action = np.random.choice(env.nA, p=probs)
            next_state, reward, done, info = env.step(action)
            score += reward
            Q[state][action] = update_helper(Q[state][action], np.max(Q[next_state]), alpha, reward, gamma)
            state = next_state
            if done:
                tmp_scores.append(score)

        if (i % plot_every == 0):
            scores.append(np.mean(tmp_scores))

        # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q

# test_sarsa()
def test_q_learning():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_sarsamax = q_learning(env, 5000, .01)

    # print the estimated optimal policy
    policy_sarsamax = np.array([np.argmax(Q_sarsamax[key]) if key in Q_sarsamax else -1 for key in np.arange(48)]).reshape((4,12))
    check_test.run_check('td_control_check', policy_sarsamax)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_sarsamax)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_sarsamax[key]) if key in Q_sarsamax else 0 for key in np.arange(48)])

# test_q_learning()

def expected_sarsa(env=gym.make('CliffWalking-v0'), num_episodes=5000, alpha=0.01, gamma=1.0):
    # initialize empty dictionary of arrays
    Q = defaultdict(lambda: np.zeros(env.nA))
    plot_every = 100
    tmp_scores = deque(maxlen=plot_every)
    scores = deque(maxlen=num_episodes)
    # loop over episodes
    for i in range(1, num_episodes+1):
        # monitor progress
        if i % 100 == 0:
            print("\rEpisode {}/{}".format(i, num_episodes), end="")
            sys.stdout.flush()
        state = env.reset()
        score = 0
        done = False
        while not done:
            probs = policy(env, Q[state], i, 0.0005)
            action = np.random.choice(env.nA, p=probs)
            next_state, reward, done, info = env.step(action)
            score += reward
            sum = np.dot(Q[next_state], probs)
            Q[state][action] = update_helper(Q[state][action], sum, alpha, reward, gamma)
            state = next_state
            if done:
                tmp_scores.append(score)

        if (i % plot_every == 0):
            scores.append(np.mean(tmp_scores))

        # plot performance
    plt.plot(np.linspace(0,num_episodes,len(scores),endpoint=False),np.asarray(scores))
    plt.xlabel('Episode Number')
    plt.ylabel('Average Reward (Over Next %d Episodes)' % plot_every)
    plt.show()
    # print best 100-episode performance
    print(('Best Average Reward over %d Episodes: ' % plot_every), np.max(scores))
    return Q


def test_expected_sarsa():
    # obtain the estimated optimal policy and corresponding action-value function
    Q_expsarsa = expected_sarsa(env, 100000, 1)

    # print the estimated optimal policy
    policy_expsarsa = np.array([np.argmax(Q_expsarsa[key]) if key in Q_expsarsa else -1 for key in np.arange(48)]).reshape(4,12)
    check_test.run_check('td_control_check', policy_expsarsa)
    print("\nEstimated Optimal Policy (UP = 0, RIGHT = 1, DOWN = 2, LEFT = 3, N/A = -1):")
    print(policy_expsarsa)

    # plot the estimated optimal state-value function
    plot_values([np.max(Q_expsarsa[key]) if key in Q_expsarsa else 0 for key in np.arange(48)])
test_expected_sarsa()
