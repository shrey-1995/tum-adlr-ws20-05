import numpy as np

from common.q_table import QTable

class Scheduler:
    def __init__(self, temperature, tasks, schedule_period, gamma=1.0):

        self.q_table = QTable()
        self.temperature = temperature
        self.auxiliary_tasks = tasks[:-1]
        self.main_task = tasks[-1]
        self.sa_map = lambda x: self.auxiliary_tasks
        self.schedule_period = schedule_period
        self.gamma = gamma

    def _actions_values_from(self, s):
        return self.q_table.Qs(s, self.sa_map(s))

    def distribution(self, state):
        dist = dict()
        total = 0
        for a, v in self._actions_values_from(state).items():
            exp_v = np.exp(v/self.temperature) # Temperature by default is 1
            total += exp_v
            dist[a] = exp_v

        if total==0:
            return {a: 1/len(self.auxiliary_tasks) for a, v in dist.items()}
        else:
            return {a: v/total for a, v in dist.items()}

    def sample(self, state):
        state = tuple(state)
        dist = self.distribution(state)

        print(dist)
        choice = np.random.random()
        cumulative_p = 0

        for a, p in dist.items():
            cumulative_p += p
            if cumulative_p > choice:
                task = self.auxiliary_tasks.index(a)
                if task is None:
                    print('Stop here')
                return task

    def train_scheduler(self, trajectories, scheduled_tasks):
        xi = self.schedule_period
        main_rewards = [r[-1] for _, _, r, _, _ in trajectories]
        for h in range(len(scheduled_tasks)):
            R = sum([r * self.gamma**k for k, r in enumerate(main_rewards[h*xi:])])

            # We used a Q-Table with 0.1 learning rate to update the values in the table.
            # Change 0.1 to the desired learning rate
            self.q_table[tuple(scheduled_tasks[:h]), scheduled_tasks[h]] += 0.1 * (R - self.q_table[tuple(scheduled_tasks[:h]), scheduled_tasks[h]])