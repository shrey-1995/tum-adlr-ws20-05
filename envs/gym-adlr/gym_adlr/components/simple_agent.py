from gym.envs.classic_control import rendering
import numpy as np

WIDTH = 20
HEIGHT = 20

class SimpleAgent:
    def __init__(self, init_x, init_y, xmax, ymax):
        self.position = (init_x, init_y)
        self.speed = (0., 0.)
        self.xmax = xmax - WIDTH/2
        self.ymax = ymax - WIDTH/2
        self.xmin = WIDTH/2
        self.ymin = HEIGHT/2

    def step(self, action, dt):
        # Action is defined as the acceleration in both axis (acc_x, acc_y)
        # dt is time for the step

        assert len(action)==2, "Action must be a 2-dimensional tuple"

        # Compute new position
        new_pos_x = self.position[0] + self.speed[0]*dt + (action[0]*dt*dt)/2
        new_pos_y = self.position[1] + self.speed[1]*dt + (action[1]*dt*dt)/2

        new_pos_x = np.clip(new_pos_x, self.xmin, self.xmax)
        new_pos_y = np.clip(new_pos_y, self.ymin, self.ymax)

        self.position = (new_pos_x, new_pos_y)
        self.speed = (self.speed[0]+action[0]*dt, self.speed[1]+action[1]*dt)

        return self.position, self.speed

    def draw(self, viewer):
        x = self.position[0]
        y = self.position[1]
        viewer.draw_polygon([(x-WIDTH/2, y-HEIGHT/2), (x+WIDTH/2, y-HEIGHT/2), (x+WIDTH/2, y+HEIGHT/2), (x-WIDTH/2, y+HEIGHT/2)], filled=True, color=(0,0,0))
        return viewer

    def get_position(self):
        return self.position[0], self.position[1]