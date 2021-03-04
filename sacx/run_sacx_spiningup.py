from sacx.sacx_agent_spiningup import SACXAgent
from environments.mountaincar_cont import MountainCar as MountainCarCont


def main():
    env = MountainCarCont(render=False)
    tasks = env.get_tasks()

    # SAC Params
    gamma = 0.99
    alpha = 0.2
    lr = 5e-3
    max_episodes = 1000
    max_steps = 1000
    buffer_maxlen = max_steps * 10
    training_batch_size = 100
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
                      alpha=alpha,
                      batch_size=training_batch_size,
                      update_after=20,
                      update_every=1)

    agent.train()


if __name__ == "__main__":
    main()
