import gym
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
from gym import spaces
import numpy as np
import math
import tensorflow as tf

class SparseToyEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    self.seed()

    # Information about our reward
    self.reward = 0
    self.visited=[]

    # Size of the screen
    self.xmax=1000
    self.ymax=1000
    self.viewer = None

    # Initialize speed for each axis and limitations for them.
    self.speed = (0, 0)
    self.max_speed = (0.7, 0.7)

    # Set initial position for the agent
    self.init_pos = self.get_random_position(clearance= 10)
    self.state = [self.init_pos, self.speed]

    # Set random location for dots and random radius
    self.loc1 = self.get_random_position(clearance= 100)
    self.radius1 = random.randint(5, 40)
    self.loc2 = self.get_random_position(clearance= 100)
    self.radius2 = random.randint(5, 40)
    self.loc3 = self.get_random_position(clearance= 100)
    self.radius3 = random.randint(5, 40)

  def get_random_position(self, clearance=20) -> tuple:
    """
    Returns a random (x, y) position with a given padding to the screen
    :param clearance: minimum distance to the border
    :return: tuple
    """
    return random.randint(clearance, self.xmax - clearance), random.randint(clearance, self.ymax - clearance)

  def seed(self, seed=None):
    """
    Creates random seed
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def step(self, action):
    position_x = self.state[0][0]
    position_y = self.state[0][1]

    speed_x = self.state[1][0]
    speed_y = self.state[1][1]

    #TODO: change speed for both axis and therefore position.
    #TODO: check if agent is in some circle or went through it. Then, include it to visited and compute reward
    #TODO: define
    reward = None
    done = None

    return self.state, reward, done, {}

  def reset(self):
    return None

  def render(self, mode='human'):
    """
    Renders the environment to the screen.
    """
    # If the view was not declared, show the window with given size
    if self.viewer is None:
      self.viewer = rendering.Viewer(self.xmax, self.ymax)
      # Draw the circles on the screen
      self.render_space()

    #TODO: include agent (square?)

    # Actual rendering
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def render_space(self, mode='human'):
    dot1 = rendering.make_circle(self.radius1, res=40, filled=True)
    dot1.set_color(0,1,0)
    dot1.add_attr(rendering.Transform(translation=self.loc1))
    self.viewer.add_geom(dot1)

    dot2 = rendering.make_circle(self.radius2, res=40, filled=True)
    dot2.set_color(1, 0, 0)
    dot2.add_attr(rendering.Transform(translation=self.loc2))
    self.viewer.add_geom(dot2)

    dot3 = rendering.make_circle(self.radius3, res=40, filled=True)
    dot3.set_color(0, 0, 1)
    dot3.add_attr(rendering.Transform(translation=self.loc3))
    self.viewer.add_geom(dot3)

    # Actual rendering
    return self.viewer.render(return_rgb_array= mode == 'rgb_array')

  def close(self):
    if self.viewer:
      self.viewer.close()

if __name__ == '__main__':
  env = SparseToyEnvironment()

  while True:
    env.render()
  env.close()
