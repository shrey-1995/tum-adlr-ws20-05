from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import time
import common.sp_sac_core as core


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, num_tasks):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(core.combined_shape(size, num_tasks), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in batch.items()}


class SACXAgent():
    def __init__(self,
                 env,
                 tasks,
                 actor_critic=core.MLPActorCritic,
                 ac_kwargs=dict(),
                 seed=0,
                 max_steps=4000,
                 max_episodes=100,
                 schedule_period=100,
                 learn_scheduler=False,
                 replay_size=int(1e6),
                 gamma=0.99,
                 polyak=0.01,
                 lr=1e-3,
                 alpha=0.2,
                 batch_size=100,
                 update_after=1000,
                 update_every=50):
        """
        Soft Actor-Critic (SAC)
        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.
            actor_critic: The constructor method for a PyTorch Module with an ``act``
                method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
                The ``act`` method and ``pi`` module should accept batches of
                observations as inputs, and ``q1`` and ``q2`` should accept a batch
                of observations and a batch of actions as inputs. When called,
                ``act``, ``q1``, and ``q2`` should return:
                ===========  ================  ======================================
                Call         Output Shape      Description
                ===========  ================  ======================================
                ``act``      (batch, act_dim)  | Numpy array of actions for each
                                               | observation.
                ``q1``       (batch,)          | Tensor containing one current estimate
                                               | of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ``q2``       (batch,)          | Tensor containing the other current
                                               | estimate of Q* for the provided observations
                                               | and actions. (Critical: make sure to
                                               | flatten this!)
                ===========  ================  ======================================
                Calling ``pi`` should return:
                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                               | given observations.
                ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                               | actions in ``a``. Importantly: gradients
                                               | should be able to flow back into ``a``.
                ===========  ================  ======================================
            ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object
                you provided to SAC.
            seed (int): Seed for random number generators.
            steps_per_epoch (int): Number of steps of interaction (state-action pairs)
                for the agent and the environment in each epoch.
            epochs (int): Number of epochs to run and train agent.
            replay_size (int): Maximum length of replay buffer.
            gamma (float): Discount factor. (Always between 0 and 1.)
            polyak (float): Interpolation factor in polyak averaging for target
                networks. Target networks are updated towards main networks
                according to:
                .. math:: \\theta_{\\text{targ}} \\leftarrow
                    \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta
                where :math:`\\rho` is polyak. (Always between 0 and 1, usually
                close to 1.)
            lr (float): Learning rate (used for both policy and value learning).
            alpha (float): Entropy regularization coefficient. (Equivalent to
                inverse of reward scale in the original SAC paper.)
            batch_size (int): Minibatch size for SGD.
            start_steps (int): Number of steps for uniform-random action selection,
                before running real policy. Helps exploration.
            update_after (int): Number of env interactions to collect before
                starting to do gradient descent updates. Ensures replay buffer
                is full enough for useful updates.
            update_every (int): Number of env interactions that should elapse
                between gradient descent updates. Note: Regardless of how long
                you wait between updates, the ratio of env steps to gradient steps
                is locked to 1.
            num_test_episodes (int): Number of episodes to test the deterministic
                policy at the end of each epoch.
            max_ep_len (int): Maximum length of trajectory / episode / rollout.
            logger_kwargs (dict): Keyword args for EpochLogger.
            save_freq (int): How often (in terms of gap between epochs) to save
                the current policy and value function.
        """
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env, self.test_env = env, env
        self.tasks = tasks
        obs_dim = env.get_state_size()[0]
        act_dim = env.action_space().shape[0]

        # Create actor-critic module and target networks for each task
        self.actor_critics = []
        self.actor_critics_target = []
        self.q_params = []
        self.pi_optimizers = []
        self.q_optimizers = []

        for i in range(len(self.tasks)):
            ac_cr = actor_critic(env.get_state_size()[0], env.action_space, **ac_kwargs)
            self.actor_critics.append(ac_cr)
            self.actor_critics_target.append(deepcopy(ac_cr))

            # Freeze target networks with respect to optimizers (only update via polyak averaging)
            for p in self.actor_critics_target[i].parameters():
                p.requires_grad = False

            # List of parameters for both Q-networks (save this for convenience)
            self.q_params.append(
                itertools.chain(self.actor_critics[i].q1.parameters(), self.actor_critics[i].q2.parameters()))

            # Set up optimizers for policy and q-function
            self.pi_optimizers.append(Adam(self.actor_critics[i].pi.parameters(), lr=lr))
            self.q_optimizers.append(Adam(self.q_params[i], lr=lr))

        # Experience buffer
        self.replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, num_tasks = 4)

        self.alpha = alpha
        self.gamma = gamma
        self.polyak = polyak

        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.schedule_period = schedule_period
        self.learn_scheduler = learn_scheduler

        self.update_after = update_after
        self.update_every = update_every

        self.batch_size = batch_size

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data, task):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        # Select reward only for task
        r = r[:, task:task + 1]

        q1 = self.actor_critics[task].q1(o, a)
        q2 = self.actor_critics[task].q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.actor_critics[task].pi(o2)

            # Target Q-values
            q1_pi_targ = self.actor_critics_target[task].q1(o2, a2)
            q2_pi_targ = self.actor_critics_target[task].q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup) ** 2).mean()
        loss_q2 = ((q2 - backup) ** 2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data, task):
        o = data['obs']
        pi, logp_pi = self.actor_critics[task].pi(o)
        q1_pi = self.actor_critics[task].q1(o, pi)
        q2_pi = self.actor_critics[task].q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def update_tasks(self, data, auxiliary=False, main=False):
        if auxiliary:
            self.update(data, 0)
            #for task in range(len(self.tasks) - 1):
                #self.update(data, task)
        if main:
            self.update(data, len(self.tasks) - 1)

    def update(self, data, task):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizers[task].zero_grad()
        loss_q, q_info = self.compute_loss_q(data, task)
        loss_q.backward()
        self.q_optimizers[task].step()

        # Freeze Q-networks so you don't waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params[task]:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizers[task].zero_grad()
        loss_pi, pi_info = self.compute_loss_pi(data, task)
        loss_pi.backward()
        self.pi_optimizers[task].step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params[task]:
            p.requires_grad = True

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.actor_critics[task].parameters(), self.actor_critics_target[task].parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

    def get_action(self, o, task, deterministic=False):
        return self.actor_critics[task].act(torch.as_tensor(o, dtype=torch.float32),
                                            deterministic)

    def test_agent(self, num_test_episodes, episode_len):
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            for i in range(episode_len):
                # Take deterministic actions at test time
                o, r, d, _ = self.test_env.step(self.get_action(o, len(self.tasks) - 1, True))  # Always use main task

    def schedule_task(self, scheduled_tasks, learn_scheduler):
        if learn_scheduler is True:
            return self.scheduler.sample(scheduled_tasks)
        else:
            if len(scheduled_tasks) == 0:
                return 0
            else:
                return (self.tasks.index(scheduled_tasks[-1]) + 1) % 3
            # return random.choice([i for i in range(len(self.tasks)

    def train(self):
        for episode in range(self.max_episodes):
            o, ep_ret, ep_len = self.env.reset(), 0, 0

            scheduled_task_step = 0
            scheduled_tasks = []
            scheduled_tasks_steps = []
            task = None

            # Main loop: collect experience in env and update/log each epoch
            for t in range(self.max_steps):
                if (t - scheduled_task_step) % self.schedule_period == 0:
                    task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler)
                    print("Switching to ", self.tasks[task])
                    scheduled_tasks_steps.append(t)
                    scheduled_tasks.append(self.tasks[task])
                    scheduled_task_step = t

                a = self.get_action(o, task)

                # Step the env
                o2, r, d, visited_circles = self.env.step(a)
                ep_ret += r
                ep_len += 1

                # Store experience to replay buffer
                self.replay_buffer.store(o, a, r, o2, d)

                # Super critical, easy to overlook step: make sure to update
                # most recent observation!
                o = o2

                if d is True:
                    break

                # Update handling
                if t >= self.update_after and t % self.update_every == 0:
                    for j in range(self.update_every):
                        batch = self.replay_buffer.sample_batch(self.batch_size)
                        self.update_tasks(data=batch, auxiliary=True, main=False)

                if visited_circles[task] == 1:
                    task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler)
                    print("Switching to ", self.tasks[task])
                    scheduled_tasks_steps.append(t)
                    scheduled_tasks.append(self.tasks[task])
                    scheduled_task_step = t

            if episode + 1 % 10 == 0:
                print("=== TESTING EPISODE ===")
                # Test the performance of the deterministic version of the agent.
                self.test_agent(1, 1000)
                print("=== END TEST EPISODE ===")

            print("Finished episode {}".format(episode+1))
