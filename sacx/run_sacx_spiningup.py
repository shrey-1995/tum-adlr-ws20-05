from sacx.sacx_agent_spiningup import SACXAgent
from environments.toy_environment import ToyEnvironment


def main():
    env = ToyEnvironment(render=True) # 2D toy environment. Define if you want to render env.
    tasks = env.get_tasks()

    # SAC Params
    gamma = 0.99
    alpha = 0.2
    lr = 1e-3
    polyak = 0.99 # To update target parameters
    max_episodes = 1000
    max_steps = 1000
    buffer_maxlen = max_steps * 10
    training_batch_size = 60
    schedule_period = 1000
    learn_scheduler = False

    agent = SACXAgent(env=env,
                      tasks=tasks,
                      max_steps=max_steps,
                      max_episodes=max_episodes,
                      schedule_period=schedule_period,
                      learn_scheduler=learn_scheduler,
                      replay_size=buffer_maxlen,
                      gamma=gamma,
                      lr=lr,
                      polyak=polyak,
                      alpha=alpha,
                      batch_size=training_batch_size,
                      update_after=training_batch_size,
                      update_every=1)

    agent.train()


if __name__ == "__main__":
    main()
