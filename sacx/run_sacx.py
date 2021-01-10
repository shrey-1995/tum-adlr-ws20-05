from sacx.sacx_agent import SACXAgent
from environments.mountaincar_cont import MountainCar as MountainCarCont


def main():
    steps_per_episode = 70
    env = MountainCarCont(render=True, max_steps=steps_per_episode)
    tasks = env.get_tasks()

    # SAC Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 3e-4
    q_lr = 3e-4
    p_lr = 3e-4
    buffer_maxlen = 1000000
    max_episodes = 500
    max_steps = 600
    training_batch_size = 64
    schedule_period = 200

    agent = SACXAgent(env=env,
                      gamma=gamma,
                      tau=tau,
                      alpha=alpha,
                      q_lr=q_lr,
                      p_lr=p_lr,
                      a_lr=a_lr,
                      buffer_maxlen=buffer_maxlen,
                      tasks=tasks,
                      max_episodes=max_episodes,
                      max_steps=max_steps,
                      training_batch_size=training_batch_size,
                      schedule_period=schedule_period,
                      storing_frequence=1,
                      store_path="./checkpoints/simple_env/{}_{}.checkpoint",
                      load_from=None)

    agent.train()


if __name__ == "__main__":
    main()
