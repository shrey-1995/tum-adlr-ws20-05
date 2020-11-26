import random
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point


# Observation space, according to source:
# state = [
#     (pos.x - VIEWPORT_W / SCALE / 2) / (VIEWPORT_W / SCALE / 2),
#     (pos.y - (self.helipad_y + LEG_DOWN / SCALE)) / (VIEWPORT_W / SCALE / 2),
#     vel.x * (VIEWPORT_W / SCALE / 2) / FPS,
#     vel.y * (VIEWPORT_H / SCALE / 2) / FPS,
#     self.lander.angle,
#     20.0 * self.lander.angularVelocity / FPS,
#     1.0 if self.legs[0].ground_contact else 0.0,
#     1.0 if self.legs[1].ground_contact else 0.0
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

def _check_collision(state, circle_to_visit, circles):
    """
    Check if a line intersects with any of the circles in a dictionary and returns its id (key in dictionary)
    :param circles: dictionary with circles
    :param line: line shape from shapely library
    :return: key of circle if intersection or None if no intersection found
    """
    trajectory = LineString([state[0], state[1]])
    circle = circles[circles.keys()[circle_to_visit]]
    circle_shapely = Point(*circle[0]).buffer(circle[1])

    if not trajectory.disjoint(circle_shapely): # If there is an intersection
        return True

    return False

def reach_circle(state, visit_sequence):
    """
    Auxiliary reward for getting to the correct circle
    :param state: (list) state of the environment
    :return: (list) reward for each circle to visit
    """
    to_visit = np.sum(visit_sequence) # Visited circles in binary array of shape (1,0,0...). The sum returns the next index to visit
    if (_check_collision(state, to_visit)):
        visit_sequence[to_visit] = 1
    return list(visit_sequence)


class TaskScheduler(object):
    """Class defines Scheduler for storing and picking tasks
    Observation space is defined as:
    [agent_position: tuple(x, y),
     speed: tuple(x, y),
     circles: {key: ((x,y), radius)},
     visited_circles: list]
    """

    def __init__(self, state, circles_):

        self.circles = circles_

        self.aux_rewards = [reach_circle for i in range(len(self.circles.keys()))]

        # Number of tasks is number of circles plus the main task
        self.num_tasks = len(self.circles.keys()) + 1

        # Establish sequence of circles to be visited
        self.visit_sequence = [k for k in self.circles.keys()]

        # Internal tracking variable for current task, and set of tasks
        self.circles_keys = self.circles.keys()
        self.current_task = 0
        self.current_set = []

    def reset(self):
        self.current_set = []

    def sample(self):
        self.current_task+=1
        self.current_set.append(self.current_task)

    def reward(self, state, main_reward):
        # Will contain a value for each circle returned from task_reward and another for the final reward
        reward_vector = []
        reward_vector += reach_circle(state)

        # Append main task reward
        reward_vector.append(main_reward)
        return reward_vector