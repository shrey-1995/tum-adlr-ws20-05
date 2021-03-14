"""
Run SAC-X in our 2-D toy environment
"""

from sacx.sacx_agent import SACXAgent
from environments.toy_environment import ToyEnvironment


def main():
    env = ToyEnvironment(render=True) # 2D toy environment. Define if you want to render env.
    tasks = env.get_tasks()

    # SAC-X Params
    gamma = 0.99
    tau = 0.01
    alpha = 0.2
    a_lr = 1e-4
    q_lr = 1e-4
    p_lr = 1e-4
    max_episodes = 70
    max_steps = 1200
    buffer_maxlen = 12000
    training_batch_size = 64
    schedule_period = 1200 # Sample new task after N episodes
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
                      storing_frequence=10, # Store networks after N episodes
                      share_layers=False, # If true, all networks will share the first layer
                      learn_scheduler=learn_scheduler,
                      store_path=None, # Path to store networks
                      load_from=None) # If not none, allows to load networks from files

    rewards = agent.train()
    agent.store_rewards(rewards, max_steps, schedule_period, filename="./results/sixth_execution.txt")


if __name__ == "__main__":
    main()
