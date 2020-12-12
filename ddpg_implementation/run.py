from ddpg_implementation.ddpg import DDPGAgent
import numpy as np
import gym
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def smooth(x):
    n = len(x)
    y = np.zeros(n)
    for i in range(n):
        start = max(0, i - 99)
        y[i] = float(x[start:(i + 1)].sum()) / (i - start + 1)
    return y


def trainer(env, agent, max_episodes, max_steps, batch_size, action_noise, render=False):
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state, action_noise)
            next_state, reward, done, _ = env.step(action)

            if render:
                env.render(mode='human')

            d_store = False if step == max_steps - 1 else done
            agent.replay_buffer.push(state, action, reward, next_state, d_store)
            episode_reward += reward

            if agent.replay_buffer.size > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards


if __name__ == "__main__":
    env = gym.make('gym_adlr.envs:simple-env-v0')

    max_episodes = 20
    max_steps = 500
    batch_size = 32

    gamma = 0.99
    tau = 1e-2
    buffer_maxlen = 100000
    critic_lr = 1e-3
    actor_lr = 1e-3

    agent = DDPGAgent(env, gamma, tau, buffer_maxlen, critic_lr, actor_lr)
    episode_rewards = trainer(env, agent, max_episodes, max_steps, batch_size, action_noise=0.1, render=True)

    with open('./output/episode_rewards_env.txt', 'w') as f:
        for item in episode_rewards:
            f.write("%s\n" % item)
