from sac.sac_agent import SACAgent
import gym

def trainer(env, max_episodes, max_steps, batch_size, render=True):
    # SAC Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 3e-4
    q_lr = 3e-4
    p_lr = 3e-4
    buffer_maxlen = 1000000

    # 2019 agent
    agent = SACAgent(env, gamma, tau, alpha, q_lr, p_lr, a_lr, buffer_maxlen)

    # train
    episode_rewards = []

    for episode in range(max_episodes):
        state = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action, render)
            agent.replay_buffer.push(state, action, reward, next_state, done)
            episode_reward += reward

            if len(agent.replay_buffer) > batch_size:
                agent.update(batch_size)

            if done or step == max_steps - 1:
                episode_rewards.append(episode_reward)
                print("Episode " + str(episode) + ": " + str(episode_reward))
                break

            state = next_state

    return episode_rewards

if __name__ == "__main__":
    env = gym.make('gym_adlr.envs:simple-env-v0')

    # Hyperparameters
    max_episodes = 50
    max_steps = 3000
    batch_size = 64
    render=True

    episode_rewards = trainer(env, max_episodes, max_steps, batch_size, render)

    with open('../output/episode_rewards_sac_env.txt', 'w') as f:
        for item in episode_rewards:
            f.write("%s\n" % item)



