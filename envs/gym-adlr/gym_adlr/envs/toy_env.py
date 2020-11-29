import gym
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
from gym import spaces
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import torch
from gym_adlr.components.car import Car
import Box2D
import pyglet
from pyglet import gl

STATE_W = 96
STATE_H = 96
VIDEO_W = 600
VIDEO_H = 400
WINDOW_W = 500
WINDOW_H = 500

class ToyEnv(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    self._seed()

    # Information about our reward
    self.reward = 500
    self.visited=[]
    self.done = False # True if we have visited all circles

    # Size of the screen
    self.xmax=WINDOW_W
    self.ymax=WINDOW_H
    self.viewer = None

    # Set time for each step
    self.fps = 50
    self.t = 0.

    # Set initial position for the agent
    self.init_position = self._get_random_position(clearance= 10)
    self.init_angle = random.randint(0, 360)

    # Init car and world
    self.world = Box2D.b2World((0, 0))
    self.car = Car(self.world, self.init_angle, *self.init_position)

    # Generate circles as tuples ((x, y), radius, (R,G,B))
    self.circles, self.circles_shapely = self._generate_circles(n_circles=3, clearance=100)

    # Define sequence in which circles should be visited
    self.sequence = [k for k in self.circles.keys()]

    # Action space is defined for steering, accelarating and breaking
    self.action_space = spaces.Box(
      np.array([-1, 0, 0]), np.array([+1, +1, +1]), dtype=np.float32
    )

    # Observation space
    self.observation_space = spaces.Box(
      low=0, high=255, shape=(self.ymax, self.xmax, 3), dtype=np.uint8
    )

  def _is_circle_valid(self, circle):
    if len(self.circles.keys())==0: # First circle is always valid
      return True
    else: # Check intersection with previous circles
      shapely_circle = Point(*circle[0]).buffer(circle[1]) # represent circle as shapely shape
      for k in self.circles_shapely:
        if not shapely_circle.disjoint(self.circles_shapely[k]): # If there is overlap return false
          return False
      return True

  def _generate_circles(self, n_circles = 3, clearance=100):
    """
    Function to generate n circles for our environment
    :param clearance: clearance from the screen edges
    :return: dictionary with all the circles
    """
    self.circles = {}
    self.circles_shapely = {}
    done = 0

    while done<n_circles:
      circle = [self._get_random_position(clearance=clearance), random.randint(15, 40), (random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))]
      if self._is_circle_valid(circle):
        self.circles[done] = circle
        self.circles_shapely[done] = Point(*circle[0]).buffer(circle[1])
        done+=1

    return self.circles, self.circles_shapely


  def _get_random_position(self, clearance=20) -> tuple:
    """
    Returns a random (x, y) position with a given padding to the screen
    :param clearance: minimum distance to the border
    :return: tuple
    """
    return random.randint(clearance, self.xmax - clearance), random.randint(clearance, self.ymax - clearance)

  def _seed(self, seed=None):
    """
    Creates random seed
    """
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def _check_collision(self, circles, line):
    """
    Check if a line intersects with any of the circles in a dictionary and returns its id (key in dictionary)
    :param circles: dictionary with circles
    :param line: line shape from shapely library
    :return: key of circle if intersection or None if no intersection found
    """
    for c in circles:
      if not line.disjoint(circles[c]):
        return c
    return None

  def step(self, action):
    """
    Given a position in a 2D space and the speed component for each dimension, the agent will accelerate in both dimensions in each step
    :param action: tuple with two elements (accelaration_x, acceleration_y)
    :return: state after action, reward collected, if the task is finished
    """
    # Store previous position to compute trajectory
    x_prev, y_prev = self.car.hull.position

    # Update information in our car class
    if action is not None:
      self.car.steer(action[0])
      self.car.gas(action[1])
      self.car.brake(action[2])

    self.car.step(1.0 / self.fps)
    self.world.Step(1.0 / self.fps, 6 * 30, 2 * 30)
    self.t += 1.0 / self.fps

    # Render new position
    self.state = self.render("state_pixels")

    # Get new position
    x, y = self.car.hull.position

    # Init variables
    step_reward = 0
    done = False

    if action is not None:  # First step without action, called from reset()
      # We discount reward for not reaching objectives
      self.reward -= 0.1

      # We don't care about fuel so just set it to 0
      self.car.fuel_spent = 0.0

      # Compute trajectory in this step and check intersection with circles
      trajectory = LineString([(x_prev, y_prev), (x, y)])
      intersection = self._check_collision(self.circles_shapely, trajectory)

      # If there is a new intersection, include it and reward agent
      if intersection is not None and intersection not in self.visited:
        self.visited.append(intersection)
        self.reward += 1.1 # .1 to compensate reward for not reaching objectives

      # Check if we finished visiting all circles
      if len(self.visited) == len(self.circles.keys()):
        self.done = True
        self.reward += 100

      # Compute how much reward we achieved in this step
      step_reward = self.reward - self.prev_reward

      # Update previous reward with current
      self.prev_reward = self.reward

    return self.state, step_reward, self.done, self.visited

  def _destroy(self):
    # Reset the view
    self.viewer = None

    # Reset car
    self.car.destroy()

  def reset(self):
    self._destroy()

    # Information about our reward
    self.reward = 0
    self.prev_reward = 0
    self.visited = []
    self.done = False  # True if we have visited all circles

    # Define circles as tuples ((x, y), radius)
    self.circles, self.circles_shapely = self._generate_circles(n_circles=3, clearance=100)

    # Define sequence in which circles should be visited
    self.sequence = [k for k in self.circles.keys()]

    # Create car
    self.car = Car(self.world, self.init_angle, *self.init_position)

    return self.step(None)[0]

  def render(self, mode='human'):
    """
    Renders the environment to the screen.
    """
    # If the view was not declared, show the window with given size
    if self.viewer is None:
      self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)

    # Render agent
    self.car.draw(self.viewer, mode != "state_pixels")

    if mode == "rgb_array":
      VP_W = VIDEO_W
      VP_H = VIDEO_H
    elif mode == "state_pixels":
      VP_W = STATE_W
      VP_H = STATE_H
    else:
      pixel_scale = 2
      VP_W = int(pixel_scale * WINDOW_W)
      VP_H = int(pixel_scale * WINDOW_H)

    gl.glViewport(0, 0, VP_W, VP_H)

    if mode == "human":
      # Draw the circles on the screen
      self._render_circles()
      return self.viewer.isopen

    image_data = (
      pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
    )
    arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
    arr = arr.reshape(VP_H, VP_W, 4)
    arr = arr[::-1, :, 0:3]

    return arr

  def _render_circles(self, mode='human'):
    """
    Renders all the circles that were created
    """
    for c in self.circles:
      dot = rendering.make_circle(self.circles[c][1], res=40, filled=True)
      dot.set_color(*self.circles[c][2])
      dot.add_attr(rendering.Transform(translation=self.circles[c][0]))
      self.viewer.add_geom(dot)

    # Actual rendering
    return self.viewer.render(return_rgb_array = mode == 'rgb_array')

  def close(self):
    """
    Close the view
    :return:
    """
    if self.viewer:
      self.viewer.close()

if __name__ == '__main__':
  env = ToyEnv()
  env.reset()

  while True:
    env.render()
  env.close()
