import random
import numpy as np


# Observation space, according to source:
# state = [
#
# ]

# Auxiliary Rewards:
# Visit circles
#
# Extrinsic Rewards:
# Success: Did we visit all circles

def touch_circles(state):
    """
    Auxiliary reward for touching lander legs on the ground
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    circles_visited = state[8:]  # 1.0 if a circle was visited
    return np.sum(circles_visited)

class TaskScheduler(object):
    """Class defines Scheduler for storing and picking tasks"""

    def __init__(self):
        self.aux_rewards = [touch_circles]

        # Number of tasks is number of auxiliary tasks plus the main task
        self.num_tasks = len(self.aux_rewards) + 1

        # Internal tracking variable for current task, and set of tasks
        self.current_task = 0
        self.current_set = set()

    def reset(self):
        self.current_set = set()

    def sample(self):
        self.current_task = random.randint(0, self.num_tasks-1)
        self.current_set.add(self.current_task)

    def reward(self, state, main_reward):
        reward_vector = []
        for task_reward in self.aux_rewards:
            reward_vector.append(task_reward(state))
        # Append main task reward
        reward_vector.append(main_reward)
        return reward_vector
