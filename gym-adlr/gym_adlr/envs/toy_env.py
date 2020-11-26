import gym
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
from gym import spaces
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
import torch

class SparseToyEnvironment(gym.Env):
  metadata = {'render.modes': ['human']}
  def __init__(self):
    self._seed()

    # Information about our reward
    self.reward = 0
    self.visited=[]
    self.done = False # True if we have visited all circles

    # Size of the screen
    self.xmax=1000
    self.ymax=1000
    self.viewer = None

    # Size of the agent
    self.agent_width = 30
    self.agent_height = 30

    # Set time for each step
    self.fps = 50
    self.t = 1./self.fps

    # Set initial position for the agent
    self.position = self._get_random_position(clearance= 10)

    # Initialize speed for each axis and limitations for them.
    self.speed = (0, 0)
    self.speed_range = (0, 1)
    self.acceleration = (0,0)

    # Define circles as tuples ((x, y), radius)
    self.circles = self._generate_circles(n_circles=3, clearance=100)

    # Express the circles with shapely library for quite intersection check
    self.circles_shapely = self._circles_to_shapely(self.circles)

    # Define sequence in which circles should be visited
    self.sequence = [k for k in self.circles.keys()]

    # Actions in 2D world from -1 to 1(x y) 
    self.action_space = spaces.Box(-1, +1, (2,), dtype=np.float32)

    """
    Observation space is defined as:
    [position_x: float,
     position_y: float,
     speed_x: float,
     speed_y: float,
     acceleration_x: float,
     acceleration_y: float,
     prev_position_x: float,
     prev_position_y: float]
    """
    self.observation_space = [self.position[0], self.position[1], self.speed[0], self.speed[1],
                              self.acceleration[0], self.acceleration[1], self.position[0], self.position[1]]

  def _generate_circles(self, n_circles = 3, clearance=100):
    """
    Function to generate n circles for our environment
    :param clearance: clearance from the screen edges
    :return: dictionary with all the circles
    """
    circles = {}
    for i in range(n_circles):
      circles[i] = (self._get_random_position(clearance=clearance), random.randint(20, 60))
    return circles

  def _circles_to_shapely(self, circles: dict):
    """
    Transforms our circles into shapely shapes to compute faster intersections
    :param circles: dictionary of circles
    :return: dictionary of shapely circles
    """
    shapely = {}
    for c in circles:
      shapely[c] = Point(*circles[c][0]).buffer(circles[c][1])
    return shapely

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

    prev_position_x = self.observation_space[6]
    prev_position_y = self.observation_space[7]

    speed_x = self.observation_space[2]
    speed_y = self.observation_space[3]

    # Action clipped to the range -1, 1
    action = action.numpy()
    print("The action is: {}".format(action))

    # Compute new position
    position_x = prev_position_x + speed_x * self.t + 0.5*(action[0]*self.t*self.t)
    position_y = prev_position_y + speed_y * self.t + 0.5 * (action[1] * self.t * self.t)

    # Ensure the position is within our window
    self.observation_space[0] = max(0, min(position_x, self.xmax))
    self.observation_space[1] = max(0, min(position_y, self.ymax))

    # Compute new speed for both axis
    self.observation_space[2] = speed_x+action[0]*self.t
    self.observation_space[3] = speed_y+action[1]*self.t

    # Update acceleration
    self.observation_space[4] = action[0]
    self.observation_space[5] = action[1]

    trajectory = LineString([(prev_position_x, prev_position_y), self.agent_state[0]])
    intersection = self._check_collision(self.circles_shapely, trajectory)
    if intersection:
      self.visited.append(intersection)
      self.reward += 1

    self.done = True if len(self.visited)==len(self.circles.keys()) else False

    if self.done:
      if self.visited==self.sequence:
        self.reward += 10

    return self.observation_space, self.reward, self.done, {}

  def _destroy(self):
    # Information about our reward
    self.reward = 0
    self.visited = []
    self.done = False  # True if we have visited all circles

    # Reset the view
    self.viewer = None

    # Initialize speed for each axis and limitations for them.
    self.speed = (0, 0)
    self.speed_range = (0, 1)

    # Set initial position for the agent
    self.init_pos = self._get_random_position(clearance=10)
    self.agent_state = [self.init_pos, self.speed]

  def reset(self):
    self._destroy()

    # Define circles as tuples ((x, y), radius)
    self.circles = self._generate_circles(n_circles=3, clearance=100)

    # Express the circles with shapely library for quite intersection check
    self.circles_shapely = self._circles_to_shapely(self.circles)

    # Define sequence in which circles should be visited
    self.sequence = [k for k in self.circles.keys()]

    return self.step(torch.tensor([0, 0]))[0]

  def render(self, mode='human'):
    """
    Renders the environment to the screen.
    """
    # If the view was not declared, show the window with given size
    if self.viewer is None:
      self.viewer = rendering.Viewer(self.xmax, self.ymax)

      # Draw the circles on the screen
      self._render.circles()

    # Render agent
    l, r, t, b = -self.agent_width / 2, self.agent_width / 2, self.agent_height, 0
    agent = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
    agent.add_attr(rendering.Transform(translation=(self.agent_state[0])))
    self.viewer.add_geom(agent)

    # Actual rendering
    return self.viewer.render(return_rgb_array=mode == 'rgb_array')

  def _render_circles(self, mode='human'):
    """
    Renders all the circles that were created
    """
    for c in self.circles:
      dot = rendering.make_circle(self.circles[c][1], res=40, filled=True)
      dot.set_color(random.uniform(0,1), random.uniform(0,1), random.uniform(0,1))
      dot.add_attr(rendering.Transform(translation=self.circles[c][0]))
      self.viewer.add_geom(dot)

    # Actual rendering
    return self.viewer.render(return_rgb_array= mode == 'rgb_array')

  def close(self):
    """
    Close the view
    :return:
    """
    if self.viewer:
      self.viewer.close()

if __name__ == '__main__':
  env = SparseToyEnvironment()

  while True:
    env.render()
  env.close()
