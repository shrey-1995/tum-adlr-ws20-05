import gym

from environments.core import State, Action
from environments.extcore import TaskEnvironment, Task

"""
    Environment wrapper for OpenAI Gym's MountainCar
"""

gym.envs.register(
    id='MountainCarLongCont-v0',
    entry_point='gym.envs.classic_control:Continuous_MountainCarEnv'
)


class MountainCarState(State):
    """
        MountainCarState
    """

    def __init__(self, state, terminal: bool):
        """
        Create a new MountainCar State
        :param state: An state obtained from the OpenAI environment
        :param terminal: A boolean indicating whether the environment state is terminal
        """
        super().__init__(terminal)
        self.state = state

    def __str__(self) -> str:
        return str(self.state)

#PEND: change task def
RED = Task("Red circle")
GREEN = Task("Green circle")
BLUE = Task("Blue circle")
MAIN_TASK = Task("Main Task")
TASKS = [RED, GREEN, BLUE, MAIN_TASK]


class MountainCar(TaskEnvironment):
    """
        MountainCar environment class
    """

    def __init__(self, render=False, max_steps=1000):
        """
        Create a new MountainCarEnvironment
        :param render: A boolean indicating whether the environment should be rendered
        """
        super().__init__()
        #PEND:change env
        self.env = gym.make('gym_adlr.envs:simple-env-clean-v0')
        self.render = render
        self.terminal = False
        self.step_v = 0
        self.max_steps = max_steps
        self.reset()

    def action_space(self) -> list:
        return self.env.action_space

    def step(self, action) -> tuple:
        """
        Perform an action on the current environment state
        :param action: The action to be performed
        :return: A two-tuple of (state, reward)
        """
        if self.terminal:
            raise Exception('Cannot perform action on terminal state!')
        if self.render:
            self.env.render()
        state, reward, self.terminal, info = self.env.step(action)
        self.step_v += 1

        return state, reward, self.terminal, info

    def reset(self):
        """
        Reset the environment state
        :return: A state containing the initial state
        """
        self.terminal = False
        self.step_v = 0
        return self.env.reset()

    @staticmethod
    def auxiliary_tasks() -> list:
        return TASKS

    @staticmethod
    def get_tasks():
        #PEND: change task
        return TASKS

    def get_state_size(self):
        return self.env.reset().shape


if __name__ == '__main__':

    _e = MountainCar(render=True)
    _s = _e.reset()

    for _ in range(1000):
        while not _s.is_terminal():
            _s, _r = _e.step(_e.sample())
        _s = _e.reset()
