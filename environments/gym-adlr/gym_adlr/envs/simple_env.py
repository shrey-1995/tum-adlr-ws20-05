import gym
from gym.utils import seeding
import random
from gym.envs.classic_control import rendering
from gym import spaces
import numpy as np
from shapely.geometry import LineString
from shapely.geometry import Point
from gym_adlr.components.simple_agent import SimpleAgent
import math
from copy import copy

STATE_W = 96
STATE_H = 96
WINDOW_W = 500
WINDOW_H = WINDOW_W

N_CIRCLES = 3

FIXED_POSITIONS = [(350, 150), (300, 400), (100, 200)]
#INIT_POS = WINDOW_W, 0
INIT_POS = (WINDOW_W / 2, WINDOW_H / 2)
SPARSE = False

EPISODE_LENGTH = 100  # Maximum number of actions that can be taken

if SPARSE:
    #### SPARSE SETTING
    INIT_REWARD = 0
    STEP_REWARD = 0
    VISITING_CIRCLE_REWARD = [40,90,160]
    FINISHING_REWARD = 160
else:
    #### NON SPARSE SETTING
    INIT_REWARD = 0
    STEP_REWARD = 0  # this value will be substracted during each step
    VISITING_CIRCLE_REWARD = [40,90,160]
    FINISHING_REWARD = 160


class SimpleEnvClean(gym.Env):
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

        # Set initial position for the agent
        self.init_position = INIT_POS

        # agent
        self.agent = SimpleAgent(*self.init_position, self.xmax, self.ymax)

        # Generate circles as tuples ((x, y), radius, (R,G,B))
        self.circles, self.circles_shapely, self.circles_positions = self._generate_fixed_circles(n_circles=N_CIRCLES)

        # Compute mindist
        self.prev_dist = np.zeros(4)
        self.prev_dist[0] = math.sqrt(math.pow(self.init_position[0] - self.circles[0][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[0][0][1], 2))
        self.prev_dist[1] = math.sqrt(math.pow(self.init_position[0] - self.circles[1][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[1][0][1], 2))
        self.prev_dist[2] = math.sqrt(math.pow(self.init_position[0] - self.circles[2][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[2][0][1], 2))

        # Define sequence in which circles should be visited
        self.sequence = [k for k in self.circles.keys()]
        self.visit_next = 0
        self.visit_sequence = np.zeros(3)

        # Action defined as acceleration in both axis
        self.action_space = spaces.Box(
            np.array([-3, -3]), np.array([+3, +3]), dtype=np.float32
        )

        print("SPARSE = {}".format(SPARSE))

        # Observation space
        # s (list): The state. Attributes:
        #                   s[0] is the horizontal position
        #                   s[1] is the vertical position
        #                   s[2:2+N_CIRCLES] X coordinates for each circle
        #                   s[2+N_CIRCLES:2+N_CIRCLES*2] Y coordinates for each circle
        #                   s[2+N_CIRCLES*2:2+N_CIRCLES*2+N_CIRCLES] boolean determining if circles were visited

        lower_bound_obs_space = np.array([0, 0] + [0] * N_CIRCLES * 2 + [0] * N_CIRCLES + [0] * N_CIRCLES)
        upper_bound_obs_space = np.array([WINDOW_W, WINDOW_H] + [WINDOW_W] * 2 * N_CIRCLES + [1] * N_CIRCLES + [1] * N_CIRCLES)

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

    def _generate_random_circles(self, n_circles=3, clearance=40):
        """
        Function to generate n circles for our environment
        :param clearance: clearance from the screen edges
        :return: dictionary with all the circles
        """
        self.circles = {}
        self.circles_shapely = {}
        self.positions = []  # To precompute positions for observation space
        done = 0

        while done < n_circles:
            color = [0,0,0]
            color[done] = 1
            color = tuple(color)
            pos = self._get_random_position(clearance=clearance)
            circle = [pos, random.randint(15, 40), color]
            if self._is_circle_valid(circle) is True:
                self.circles[done] = circle
                self.circles_shapely[done] = Point(*circle[0]) # Include this to consider radius: .buffer(circle[1])
                self.positions.append(circle[0][0])
                self.positions.append(circle[0][1])
                done += 1

        return self.circles, self.circles_shapely, self.positions

    def _generate_fixed_circles(self, n_circles=3):
        """
        Function to generate n fixed circles for our environment
        :param clearance: clearance from the screen edges
        :return: dictionary with all the circles
        """
        self.circles = {}
        self.circles_shapely = {}
        self.positions = []  # To precompute positions for observation space
        color = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
        for i in range(n_circles):
            circle = [FIXED_POSITIONS[i], 40,
                      color[i]]
            self.circles[i] = circle
            self.circles_shapely[i] = Point(*circle[0]).buffer(circle[1])
            self.positions.append(circle[0][0])
            self.positions.append(circle[0][1])

        return self.circles, self.circles_shapely, self.positions

    def _get_random_position(self, clearance=20) -> tuple:
        """
        Returns a random (x, y) position with a given padding to the screen
        :param clearance: minimum distance to the border
        :return: tuple
        """
        random.seed(a=None)
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
        current_visit = np.zeros(3)

        # Store previous position to compute trajectory
        x_prev, y_prev = self.agent.get_position()

        if action is not None:
            pos = self.agent.step(action)

        # Get new position
        x, y = self.agent.get_position()

        # Init variables
        step_reward = np.zeros(4)
        done = False
        visit = np.zeros(3)

        if action is not None:  # First step without action, called from reset()
            # We discount reward for not reaching objectives
            self.reward -= STEP_REWARD

            # Compute trajectory in this step and check intersection with circles
            trajectory = LineString([(x_prev, y_prev), (x, y)])
            intersection = self._check_collision(self.circles_shapely, trajectory)

            # Compute reward for auxiliary tasks
            curr_dist = np.zeros(4)
            curr_dist[0] = math.sqrt(math.pow(x - self.circles[0][0][0], 2) + math.pow(y - self.circles[0][0][1], 2))
            curr_dist[1] = math.sqrt(math.pow(x - self.circles[1][0][0], 2) + math.pow(y - self.circles[1][0][1], 2))
            curr_dist[2] = math.sqrt(math.pow(x - self.circles[2][0][0], 2) + math.pow(y - self.circles[2][0][1], 2))

            diff = self.prev_dist - curr_dist
            #diff = -curr_dist
            self.prev_dist = curr_dist

            if not SPARSE:
                #step_reward += np.minimum(diff, [3, 3, 3, 0])
                step_reward += diff
            if intersection is not None:
                step_reward[intersection] += 20
                visit[intersection] = 3
                current_visit[intersection] = 3

                if intersection == self.visit_next:
                    # Preempt task on reaching the circle
                    self.visit_sequence[intersection]=1
                    self.visited[intersection] = 3
                    self.reward += VISITING_CIRCLE_REWARD[intersection]
                    step_reward[3] += VISITING_CIRCLE_REWARD[intersection]
                    self.visit_next+=1
                    #if intersection==1:
                    #   self.done=True
                    if np.sum(self.visit_sequence) == len(self.visit_sequence):
                        self.done = True
                        step_reward[3] = FINISHING_REWARD
                        print("Done with reward: ", self.reward)

                    print("Circle reached: ", step_reward)

                elif intersection == self.visit_next-1:
                    pass

                else:
                    '''self.visit_next=0
                    self.reward-=300
                    #step_reward[3] = -300

                    for i in range(len(self.visit_sequence)):
                        self.visit_sequence[i] = 0
                        self.visited[i] = 0'''

                    if intersection==0:
                        self.visit_next += 1
                        self.reward+=VISITING_CIRCLE_REWARD[intersection]
                        self.visited[0] = 3
                        self.visit_sequence[0] = 1
                        step_reward[3] += VISITING_CIRCLE_REWARD[intersection]

        # Update obsetvation space
        state = [x, y] + self.circles_positions + list(self.visited) + list(current_visit)

        self.observation_space = np.array(state, dtype=np.float32)
        self.observation_space[:8] = (self.observation_space[:8]/100)-2.5
        if render:
            self.render()

        return self.observation_space, step_reward, self.done, visit

    def _destroy(self):

        self.circles = {}
        self.circles_shapely = {}

    def reset(self, r=True):
        self._destroy()

        # Information about our reward
        self.reward = INIT_REWARD
        self.prev_reward = 0
        self.visit_next = 0
        self.visited = np.zeros(N_CIRCLES)
        self.done = False  # True if we have visited all circles

        # Define circles as tuples ((x, y), radius)
        self.circles, self.circles_shapely, self.circles_positions = self._generate_fixed_circles(n_circles=N_CIRCLES)

        # Define sequence in which circles should be visited
        self.sequence = [k for k in self.circles.keys()]
        self.visit_sequence = np.zeros(3)

        # Set initial position for the agent
        if r:
            self.init_position = self._get_random_position(clearance=20)
        else:
            self.init_position = INIT_POS

        # Reset minimum distance
        self.prev_dist[0] = math.sqrt(math.pow(self.init_position[0] - self.circles[0][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[0][0][1], 2))
        self.prev_dist[1] = math.sqrt(math.pow(self.init_position[0] - self.circles[1][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[1][0][1], 2))
        self.prev_dist[2] = math.sqrt(math.pow(self.init_position[0] - self.circles[2][0][0], 2) + math.pow(
            self.init_position[1] - self.circles[2][0][1], 2))
        self.prev_dist[3] = 0
        # Create car
        self.agent = SimpleAgent(*self.init_position, self.xmax, self.ymax)

        """if self.viewer:
            self.viewer.close()
            self.viewer = None
            """

        self.render()

        return self.step(None)[0]

    def render(self, mode='human', reset_circles=False):
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
