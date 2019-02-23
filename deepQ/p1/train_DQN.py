from agent import Agent
from collections import deque


def train_dqn(n_episodes=5, max_t=500, eps_start=0.95, eps_end=0.01, eps_decay=0.999, train=True):
    env_info = env.reset(train_mode=True)[brain_name] # reset the environment
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    step = 0
    agent = Agent(37, 4, 0)

    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=train)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        step = 0
        while step < max_t:
            step +=1
            action = agent.act(state, eps)
            env_info = env.step(action)[brain_name]
            reward = env_info.rewards[0]
            next_state = env_info.vector_observations[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                break
            scores_window.append(scores)
            scores.append(score)
            eps = max(eps_end, eps_decay*eps)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window)>=13.0 and train:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint_dqn.pth')
            break
    return scores

scores = train_dqn()
