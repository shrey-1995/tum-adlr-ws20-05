from sacx.sacx_agent_3d import SACXAgent
from environments.mountaincar_cont import MountainCar as MountainCarCont
from kukaGymEnv import KukaGymEnv

def main():
    env = KukaGymEnv(renders=True, isDiscrete=False, maxSteps=10000000)
    tasks = MountainCarCont.get_tasks()

    # SAC Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 2e-4
    q_lr = 2e-4
    p_lr = 2e-4
    max_episodes = 100
    max_steps = 1500
    buffer_maxlen = 30000
    training_batch_size = 128
    schedule_period = 500
    learn_scheduler = False

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
                      storing_frequence=10,
                      share_layers=False,
                      learn_scheduler=learn_scheduler,
                      store_path=None,
                      load_from=None)

    rewards = agent.train()
    agent.test(num_episodes=10)
    agent.store_rewards(rewards, max_steps, schedule_period, filename="./results/dense_3_auxiliary.txt")


if __name__ == "__main__":
    main()
