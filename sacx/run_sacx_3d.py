from sacx.sacx_agent_3d import SACXAgent
from environments.toy_environment import ToyEnvironment
from kukaGymEnv import KukaGymEnv

def main():
    env = KukaGymEnv(renders=False, isDiscrete=False, maxSteps=10000000)
    tasks = ToyEnvironment.get_tasks()

    # SAC Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 8e-5
    q_lr = 8e-5
    p_lr = 8e-5
    max_episodes = 100
    max_steps = 750
    buffer_maxlen = 30000
    training_batch_size = 64
    schedule_period = 250 # Sample new task after N episodes
    learn_scheduler = False # True -> SAC-X

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
                      share_layers=False, # If true, all networks will share the first layer
                      learn_scheduler=learn_scheduler,
                      store_path=None, # Path to store networks
                      load_from=None) # If not none, allows to load networks from files

    rewards = agent.train()
    agent.store_rewards(rewards, max_steps, schedule_period, filename="./results/dense_3_auxiliary.txt")


if __name__ == "__main__":
    main()
