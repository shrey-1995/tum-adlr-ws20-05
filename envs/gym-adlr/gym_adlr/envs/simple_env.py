import gym
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
from gym import spaces
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from gym_adlr.components.simple_agent import SimpleAgent

STATE_W = 96
STATE_H = 96
WINDOW_W = 500
WINDOW_H = WINDOW_W

N_CIRCLES = 3

SPARSE = True

if SPARSE:
    #### SPARSE SETTING
    INIT_REWARD = 0
    STEP_REWARD = 0
    VISITING_CIRCLE_REWARD = 0
    FINISHING_REWARD = 500
else:
    #### NON SPARSE SETTING
    INIT_REWARD = 300
    STEP_REWARD = 0.1  # this value will be substracted during each step
    VISITING_CIRCLE_REWARD = 30
    FINISHING_REWARD = 500


class SimpleEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self._seed()

        # Information about our reward
        self.reward = INIT_REWARD
        self.visited = np.zeros(N_CIRCLES)
        self.done = False  # True if we have visited all circles

        # Size of the screen
        self.xmax = WINDOW_W
        self.ymax = WINDOW_H
        self.viewer = None

        # Set time for each step
        self.fps = 10
        self.t = 0.
        self.dt = 1./self.fps

        # Set initial position for the agent
        self.init_position = self._get_random_position(clearance=100)

        # agent
        self.agent = SimpleAgent(*self.init_position, self.xmax, self.ymax)

        # Generate circles as tuples ((x, y), radius, (R,G,B))
        self.circles, self.circles_shapely, self.circles_positions = self._generate_circles(n_circles=N_CIRCLES, clearance=100)

        # Define sequence in which circles should be visited
        self.sequence = [k for k in self.circles.keys()]

        # Action defined as acceleration in both axis
        self.action_space = spaces.Box(
            np.array([-3, -3]), np.array([+3, +3]), dtype=np.float32
        )

        # Observation space
        # s (list): The state. Attributes:
        #                   s[0] is the horizontal position
        #                   s[1] is the vertical position
        #                   s[2] is the horizontal speed
        #                   s[3] is the vertical speed
        #                   s[4:4+N_CIRCLES] X coordinates for each circle
        #                   s[4+N_CIRCLES:4+N_CIRCLES*2] Y coordinates for each circle
        #                   s[4+N_CIRCLES*2:4+N_CIRCLES*2+N_CIRCLES] boolean determining if circles were visited

        lower_bound_obs_space = np.array([0, 0, -1., -1.] + [0]*N_CIRCLES*2 + [0]*N_CIRCLES)
        upper_bound_obs_space = np.array([WINDOW_W, WINDOW_H, 1., 1.] + [WINDOW_W]*2*N_CIRCLES + [1]*N_CIRCLES)

        self.observation_space = spaces.Box(
            lower_bound_obs_space, upper_bound_obs_space, dtype=np.float32
        )

    def _is_circle_valid(self, circle):
        """
        Returns True if circle doesn't overlap with any other circle, else False
        :param circle:
        :return:
        """
        if len(self.circles.keys()) == 0:  # First circle is always valid
            return True
        else:  # Check intersection with previous circles
            shapely_circle = Point(*circle[0]).buffer(circle[1])  # represent circle as shapely shape
            for k in self.circles_shapely:
                if not shapely_circle.disjoint(self.circles_shapely[k]):  # If there is overlap return false
                    return False
            return True

    def _generate_circles(self, n_circles=3, clearance=100):
        """
        Function to generate n circles for our environment
        :param clearance: clearance from the screen edges
        :return: dictionary with all the circles
        """
        self.circles = {}
        self.circles_shapely = {}
        self.positions = [] # To precompute positions for observation space
        done = 0

        while done < n_circles:
            circle = [self._get_random_position(clearance=clearance), random.randint(15, 40),
                      (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1))]
            if self._is_circle_valid(circle):
                self.circles[done] = circle
                self.circles_shapely[done] = Point(*circle[0]).buffer(circle[1])
                self.positions.append(circle[0][0])
                self.positions.append(circle[0][1])
                done += 1

        return self.circles, self.circles_shapely, self.positions

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

    def step(self, action, render=True):
        """
        Performs a step in our world

        """
        # Store previous position to compute trajectory
        x_prev, y_prev = self.agent.get_position()

        # End if agent achieved nothing and return negative reward
        if self.t > 120:
            return self.observation_space, -200, True, self.visited

        if action is not None:
            pos, speed = self.agent.step(action, self.dt)
        else:
            speed = (0., 0.)

        # Get new position
        x, y = self.agent.get_position()

        # Init variables
        step_reward = 0
        done = False

        if action is not None:  # First step without action, called from reset()
            # We discount reward for not reaching objectives
            self.reward -= STEP_REWARD

            # Compute trajectory in this step and check intersection with circles
            trajectory = LineString([(x_prev, y_prev), (x, y)])
            intersection = self._check_collision(self.circles_shapely, trajectory)

            # If there is a new intersection, include it and reward agent
            if intersection is not None and self.visited[intersection] == 0:
                self.visited[intersection] = 1
                self.reward += VISITING_CIRCLE_REWARD

            # Check if we finished visiting all circles
            if np.sum(self.visited) == len(self.circles.keys()):
                self.done = True
                self.reward += FINISHING_REWARD

            # Compute how much reward we achieved in this step
            step_reward = self.reward - self.prev_reward

            # Update previous reward with current
            self.prev_reward = self.reward

            self.t += self.dt

        # Update obsetvation space
        state = [x, y, speed[0], speed[1]] + self.circles_positions + list(self.visited)

        self.observation_space = np.array(state, dtype=np.float32)

        if render:
            self.render()

        return self.observation_space, step_reward, self.done, self.visited

    def _destroy(self):
        # Reset the view
        self.viewer = None

        self.circles = {}
        self.circles_shapely = {}

        self.t = 0.

    def reset(self):
        self._destroy()

        # Information about our reward
        self.reward = INIT_REWARD
        self.prev_reward = 0
        self.visited = np.zeros(N_CIRCLES)
        self.done = False  # True if we have visited all circles

        # Define circles as tuples ((x, y), radius)
        self.circles, self.circles_shapely, self.circles_positions = self._generate_circles(n_circles=3, clearance=100)

        # Define sequence in which circles should be visited
        self.sequence = [k for k in self.circles.keys()]

        # Set initial position for the agent
        self.init_position = self._get_random_position(clearance=100)

        # Create car
        self.agent = SimpleAgent(*self.init_position, self.xmax, self.ymax)

        self.render()

        return self.step(None)[0]

    def render(self, mode='human'):
        """
    Renders the environment to the screen.
    """
        # If the view was not declared, show the window with given size
        if self.viewer is None:
            self.viewer = rendering.Viewer(WINDOW_W, WINDOW_H)
            self._render_circles()

        # Render agent
        self.agent.draw(self.viewer)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

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
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        """
    Close the view
    :return:
    """
        if self.viewer:
            self.viewer.close()


if __name__ == '__main__':
    env = SimpleEnv()
    env.reset()

    while True:
        env.render()
    env.close()
