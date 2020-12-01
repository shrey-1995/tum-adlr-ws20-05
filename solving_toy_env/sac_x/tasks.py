import random


# Observation space, according to source:
# state = [
#
# ]

# Auxiliary Rewards:
# Touch. Maximizing number of legs touching ground
# Hover Planar. Minimize the planar movement of the lander craft
# Hover Angular. Minimize the rotational movement of ht lander craft
# Upright. Minimize the angle of the lander craft
# Goal Distance. Minimize distance between lander craft and pad
#
# Extrinsic Rewards:
# Success: Did the lander land successfully (1 or 0)

def touch(state):
    """
    Auxiliary reward for touching lander legs on the ground
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    left_contact = state[6]  # 1.0 if self.legs[0].ground_contact else 0.0
    right_contact = state[7]  # 1.0 if self.legs[1].ground_contact else 0.0
    reward = left_contact + right_contact
    return reward


def hover_planar(state):
    """
    Auxiliary reward for hovering the lander (minimal planar movement)
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    x_vel = state[2]  # vel.x * (VIEWPORT_W / SCALE / 2) / FPS
    y_vel = state[3]  # vel.y * (VIEWPORT_H / SCALE / 2) / FPS
    reward = 2.0 - (abs(x_vel) + abs(y_vel))
    return reward


def hover_angular(state):
    """
    Auxiliary reward for hovering the lander (minimal angular movement)
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    ang_vel = state[5]  # 20.0 * self.lander.angularVelocity / FPS
    reward = 2.0 - abs(ang_vel)
    return reward


def upright(state):
    """
    Auxiliary reward for keeping the lander upright
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    angle = state[4]  # self.lander.angle
    reward = 2.0 - abs(angle)
    return reward


def goal_distance(state):
    """
    Auxiliary reward for distance from lander to goal
    :param state: (list) state of lunar lander
    :return: (float) reward
    """
    x_pos = state[2]  # (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2)
    y_pos = state[3]  # (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2)
    reward = 2.0 - (abs(x_pos) + abs(y_pos))
    return reward


class TaskScheduler(object):
    """Class defines Scheduler for storing and picking tasks"""

    def __init__(self):
        self.aux_rewards = [touch,
                            hover_planar,
                            hover_angular,
                            upright,
                            goal_distance]

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
