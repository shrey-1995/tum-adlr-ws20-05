import math
import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pickle
import torch.nn as nn
import numpy as np
from sacx.scheduler import Scheduler
from collections import deque
from common.models import SoftQNetwork, PolicyNetwork
from common.buffer import BasicBuffer

class SACXAgent():
    def __init__(self,
                 env,
                 gamma,
                 tau,
                 alpha,
                 q_lr,
                 p_lr,
                 a_lr,
                 buffer_maxlen,
                 tasks,
                 max_episodes,
                 max_steps,
                 training_batch_size,
                 share_layers=False,
                 schedule_period=100,
                 storing_frequence=10,
                 learn_scheduler=True,
                 store_path="./checkpoints/simple_env/{}_{}.checkpoint",
                 load_from=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.action_range = [env.action_space.low, env.action_space.high]
        self.obs_dim = env.get_state_size()
        self.action_dim = 3

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        self.schedule_period = schedule_period
        self.learn_scheduler = learn_scheduler

        self.storing_frequence = storing_frequence
        self.store_path = store_path

        self.tasks = tasks
        self.max_episodes = max_episodes
        self.max_steps = max_steps
        self.training_batch_size = training_batch_size

        self.p_nets = self.init_p_models(load_from, share_layer=share_layers)
        self.q_nets1, self.q_nets2, self.target_q_nets1, self.target_q_nets2 = self.init_q_models(load_from, share_layer=share_layers)
        self.q1_optimizers, self.q2_optimizers, self.policy_optimizers = self.init_optimizers(q_lr, p_lr, load_from)
        self.entropy_temperatures = self.init_temperatures(a_lr, alpha, load_from)

        self.replay_buffer = BasicBuffer(buffer_maxlen)
        #self.main_replay_buffer = deque(maxlen=5) # Used for retrace. Not converging.
        #self.non_zero_main_rewards = deque(maxlen=5) # Used for retrace. Not converging.
        self.main_replay_buffer_q = BasicBuffer(3000)
        self.non_zero_main_rewards_q = deque(maxlen=1000)
        self.non_zero_rewards_q = deque(maxlen=1000)

        # SAC-X
        self.scheduler = Scheduler(temperature=1, tasks=tasks, schedule_period=schedule_period)

    def init_p_models(self, load_path, share_layer):
        policy_nets = []

        if share_layer is True:
            shared_layer = nn.Linear(self.obs_dim, 256)
        else:
            shared_layer = None

        for i, task in enumerate(self.tasks):
            if load_path is None:
                obs_dim = self.obs_dim if i==3 else 6 # For auxiliary tasks, we only consider agent position and target position as obs. space
                policy_nets.append(PolicyNetwork(obs_dim, self.action_dim, shared_layer=shared_layer).to(self.device))
            else:
                policy_nets.append(torch.load(load_path.format('p_net', i)))
        return policy_nets

    def init_q_models(self, load_path, share_layer):
        q_nets1 = []
        q_nets2 = []
        target_q_nets1 = []
        target_q_nets2 = []

        if share_layer is True:
            shared_layer_1 = nn.Linear(self.obs_dim + self.action_dim, 256)
            shared_layer_2 = nn.Linear(self.obs_dim + self.action_dim, 256)
            shared_layer_t_1 = nn.Linear(self.obs_dim + self.action_dim, 256)
            shared_layer_t_2 = nn.Linear(self.obs_dim + self.action_dim, 256)
        else:
            shared_layer_1 = None
            shared_layer_2 = None
            shared_layer_t_1 = None
            shared_layer_t_2 = None

        for i, task in enumerate(self.tasks):
            if load_path is None:
                obs_dim = self.obs_dim if i == 3 else 6
                q_nets1.append(SoftQNetwork(obs_dim, self.action_dim, shared_layer=shared_layer_1).to(self.device))
                q_nets2.append(SoftQNetwork(obs_dim, self.action_dim, shared_layer=shared_layer_2).to(self.device))
                target_q_nets1.append(SoftQNetwork(obs_dim, self.action_dim, shared_layer=shared_layer_t_1).to(self.device))
                target_q_nets2.append(SoftQNetwork(obs_dim, self.action_dim, shared_layer=shared_layer_t_2).to(self.device))

                # copy params to target param
                for target_param, param in zip(target_q_nets1[i].parameters(), q_nets1[i].parameters()):
                    target_param.data.copy_(param)

                for target_param, param in zip(target_q_nets2[i].parameters(), q_nets2[i].parameters()):
                    target_param.data.copy_(param)
            else:
                q_nets1.append(torch.load(load_path.format('q_net1', i)))
                q_nets2.append(torch.load(load_path.format('q_net2', i)))
                target_q_nets1.append(torch.load(load_path.format('target_q_net1', i)))
                target_q_nets2.append(torch.load(load_path.format('target_q_net2', i)))

        return q_nets1, q_nets2, target_q_nets1, target_q_nets2

    def init_optimizers(self, q_lr, policy_lr, load_path):
        q1_optimizers = []
        q2_optimizers = []
        policy_optimizers = []

        for i, task in enumerate(self.tasks):
            if load_path is None:
                q1_optimizers.append(optim.Adam(self.q_nets1[i].parameters(), lr=q_lr))
                q2_optimizers.append(optim.Adam(self.q_nets2[i].parameters(), lr=q_lr))
                policy_optimizers.append(optim.Adam(self.p_nets[i].parameters(), lr=policy_lr))
            else:
                q1_optimizers.append(torch.load(load_path.format('q1_optimizer', i)))
                q2_optimizers.append(torch.load(load_path.format('q2_optimizer', i)))
                policy_optimizers.append(torch.load(load_path.format('policy_optimizer', i)))

        return q1_optimizers, q2_optimizers, policy_optimizers

    def init_temperatures(self, a_lr, alpha, load_path):
        temperatures = []

        for i in range(len(self.tasks)):
            if load_path is None:
                target_entropy = -torch.prod(torch.Tensor(self.action_dim).to(self.device)).item()
                log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                alpha_optim = optim.Adam([log_alpha], lr=a_lr)
                temperatures.append((alpha, target_entropy, log_alpha, alpha_optim))
            else:
                with open(load_path.format('temperature', i), 'rb') as f:
                    temperatures.append(pickle.load(f))

        return temperatures

    def rescale_action(self, action):
        """
        Rescale actions to desired range
        :param action: Original action
        :return: Rescaled action
        """
        return action * 0.05

    def get_action(self, state, task, deterministic=False):
        """
        Sample action using a given task.
        :param state: observation space
        :param task: task used to sample actions
        :param deterministic: If True, standard deviation will not be considered for sampling
        :return: z, rescaled action, probability of action
        """

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if task<3: # For auxiliary tasks, we only consider agent position and target position as obs. space
            agent_pos = state[:, :3]
            goal_pos = state[:, 3*(task+1):3*(task+1)+3]
            state = torch.cat((agent_pos, goal_pos), 1)

        mean, log_std = self.p_nets[task].forward(state)

        std = log_std.exp()

        normal = Normal(mean, std)

        if not deterministic:
            z = normal.sample()
            action = torch.tanh(z)
            prob = normal.log_prob(z)
        else:
            action = torch.tanh(mean)
            prob = normal.log_prob(mean)

        action = action.cpu().detach().squeeze(0).numpy()

        prob = torch.exp(prob.sum(1, keepdim=True))
        prob = prob.cpu().detach().squeeze(0).numpy()

        return z if not deterministic else mean, self.rescale_action(action), prob

    def get_probability(self, state, task, z):
        """
        Get the probability of a given z for a task
        :param state: observation state
        :param task: task to compute probability
        :param z: value sampled from distribution
        :return: probability
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if task<3:
            agent_pos = state[:, :3]
            goal_pos = state[:, 3*(task+1):3*(task+1)+3]
            state = torch.cat((agent_pos, goal_pos), 1)

        mean, log_std = self.p_nets[task](state)
        std = log_std.exp()

        normal = Normal(mean, std)
        prob = normal.log_prob(z)

        prob = torch.exp(prob.sum(1, keepdim=True))
        prob = prob.cpu().detach().squeeze(0).numpy()
        return prob

    def train(self):
        """
        Train agent
        """
        episode_rewards = []

        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0

            # Information used by the scheduler
            scheduled_tasks = []
            main_reward_list = []
            scheduled_tasks_steps = []
            scheduled_task_step = 0

            # Store trajectories
            trajectory = []
            #self.main_replay_buffer.append(trajectory)
            #self.non_zero_main_rewards.append(main_reward_list)

            task = None

            for step in range(self.max_steps):

                # Schedule new task after schedule_period steps
                if (step-scheduled_task_step) % self.schedule_period == 0:
                    prev_task = task
                    t = 0
                    while task == prev_task:
                        t+=1
                        task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler, episode>9)
                        if t > 10:  # Avoid the scheduler selecting same task forever
                            task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler, episode>9)
                            break

                    scheduled_tasks_steps.append(step)
                    scheduled_tasks.append(self.tasks[task])
                    scheduled_task_step = step
                    print("Switching to ", self.tasks[task])

                z, action, prob = self.get_action(state, task) # Sample new action using the task policy network
                next_state, reward, done, visited_circles = self.env.step(action) # Interact with environment

                if reward[3]>0: # If main task received non-zero reward, store information in buffer for selective sampling
                    main_reward_list.append(step)
                    self.non_zero_rewards_q.append((state, action, np.array([reward]), next_state, done))

                # Store trajectory
                self.replay_buffer.push(state, action, reward, next_state, done)
                trajectory.append((state, action, reward, next_state, done, z, prob))

                episode_reward += reward[task] # Update episode reward

                # Update auxiliary tasks if we have enough training data
                if len(self.replay_buffer) > self.training_batch_size:
                    self.update(self.training_batch_size, auxiliary=True, main=False, epochs=1)

                if done or step == self.max_steps - 1:
                    print("Finished episode " + str(episode) + ": " + str(episode_reward))
                    break

                # Schedule new task if previous one was finished
                if task < 3:
                    if visited_circles[task] == 3:
                        prev_task = task
                        t = 0
                        while task==prev_task:
                            t+=1
                            task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler, episode>9)
                            if t>10: # Avoid the scheduler selecting same task forever
                                task = self.schedule_task(scheduled_tasks[-2:], self.learn_scheduler, episode>9)
                                break
                        scheduled_tasks.append(self.tasks[task])
                        scheduled_tasks_steps.append(step)
                        scheduled_task_step = step
                        print("Switching to ", self.tasks[task])

                state = next_state

                # Retrace implementation
                '''if (step+1)%10==0:
                    trajectories = self.sample_trajectories(10, 8, False)
                    self.update_q_main(trajectories)
                    self.update_p_main(trajectories)'''

            # Train scheduler after 7 episodes to ensure random exploration at first
            if self.learn_scheduler is True and episode+1>7:
                self.scheduler.train_scheduler(trajectories=trajectory, scheduled_tasks=scheduled_tasks, scheduled_tasks_steps=scheduled_tasks_steps)

            # Update SAC implementation of Main Task
            self.update(self.training_batch_size, auxiliary=False, main=True, epochs=200)

            # Testing episode and train using main task
            if (episode+1) % 3 == 0:
                print("=== TESTING EPISODE ===")
                for j in range(1):
                    test_rewards = self.test(1, 800)
                    episode_rewards.append(test_rewards)
                print("=== END TEST EPISODE ===")


        return episode_rewards

    def update(self, batch_size, auxiliary=True, main=False, epochs=1):
        """
        Used to update networks in a simple way
        :param batch_size: training batch size
        :param auxiliary: boolean to decide if auxiliary tasks must be trained
        :param main: boolean to decide if main task must be trained
        :param epochs: number of epochs for training
        """
        for e in range(epochs):
            if auxiliary is True:
                for i in range(len(self.tasks)-1):
                    self.update_task(batch_size, i)
            if main is True:
                self.update_task(batch_size, len(self.tasks)-1)

    def update_task(self, batch_size, index, f_t = True):
        """
        Uses SAC to update the networks for a given task
        :param batch_size: training batch size
        :param index: task index to be updated
        :param f_t: if true, we use auxiliary tasks experience for training, else we use main task experience generated in test function
        """
        i = index

        if f_t:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size, index == 3, self.non_zero_rewards_q)
        else:
            states, actions, rewards, next_states, dones = self.main_replay_buffer_q.sample(batch_size, index == 3, self.non_zero_main_rewards_q)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        rewards = rewards.reshape(batch_size, rewards.shape[-1])  # Reshape
        rewards = rewards[:, i:i + 1]  # Select only rewards for this task
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        dones = dones.view(dones.size(0), -1)

        next_actions, next_log_pi = self.p_nets[i].sample(next_states)
        next_q1 = self.target_q_nets1[i](next_states, next_actions)
        next_q2 = self.target_q_nets2[i](next_states, next_actions)
        alpha = self.entropy_temperatures[i][0]
        next_q_target = torch.min(next_q1, next_q2) - alpha * next_log_pi
        expected_q = rewards + (1 - dones) * self.gamma * next_q_target

        # q loss
        curr_q1 = self.q_nets1[i].forward(states, actions)
        curr_q2 = self.q_nets2[i].forward(states, actions)
        q1_loss = F.mse_loss(curr_q1, expected_q.detach())
        q2_loss = F.mse_loss(curr_q2, expected_q.detach())

        # update q networks
        self.q1_optimizers[i].zero_grad()
        q1_loss.backward()
        self.q1_optimizers[i].step()

        self.q2_optimizers[i].zero_grad()
        q2_loss.backward()
        self.q2_optimizers[i].step()

        # Update policy and target Q-networks
        new_actions, log_pi = self.p_nets[i].sample(states)
        if self.update_step % self.delay_step == 0:
            min_q = torch.min(
                self.q_nets1[i].forward(states, new_actions),
                self.q_nets2[i].forward(states, new_actions)
            )
            policy_loss = (self.entropy_temperatures[i][0] * log_pi - min_q).mean()

            self.policy_optimizers[i].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[i].step()

            # target networks
            for target_param, param in zip(self.target_q_nets1[i].parameters(), self.q_nets1[i].parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

            for target_param, param in zip(self.target_q_nets2[i].parameters(), self.q_nets2[i].parameters()):
                target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

        # Update entropy. Recall info is stored as (alpha, target_entropy, log_alpha, alpha_optim)
        log_alpha = self.entropy_temperatures[i][2]
        target_entropy = self.entropy_temperatures[i][1]
        alpha_loss = (log_alpha * (-log_pi - target_entropy).detach()).mean()

        self.entropy_temperatures[i][3].zero_grad()  # Alpha optim
        alpha_loss.backward()
        self.entropy_temperatures[i][3].step()

        self.entropy_temperatures[i] = (
            log_alpha.exp(), self.entropy_temperatures[i][1], self.entropy_temperatures[i][2],
            self.entropy_temperatures[i][3])

        self.update_step += 1

    def test(self, num_episodes=3, max_steps=500, train=True):
        """
        Test agent behavior. Also allows to train main task using this experience
        :param num_episodes: number of testing episodes
        :param max_steps: maximum steps in one episode
        :param train: boolean to determine if we want to also train the main task
        """
        episode_rewards = []

        for episode in range(num_episodes):
            print("Testing episode {}\n".format(episode))
            state = self.env.reset(False)
            episode_reward = 0

            for step in range(max_steps):
                _, action, _ = self.get_action(state, 3)  # Sample new action using the main task policy network
                next_state, reward, done, visited_circles = self.env.step(action)

                if reward[3]>0: # If we found positive reward
                    self.non_zero_main_rewards_q.append((state, action, np.array([reward]), next_state, done))

                self.main_replay_buffer_q.push(state, action, reward, next_state, done)
                episode_reward += reward[3]

                # Train main task using its own experience
                if len(self.main_replay_buffer_q)>64 and train:
                    self.update_task(64, len(self.tasks) - 1, False)

                if done or step == max_steps - 1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    break

                state = next_state
            print("Finished with reward: ", episode_reward)
        return episode_rewards

    def store_models(self):
        for i, p_net in enumerate(self.p_nets):
            torch.save(p_net.state_dict(), self.store_path.format('p_net', i))

        for i, q_net in enumerate(self.q_nets1):
            torch.save(q_net.state_dict(), self.store_path.format('q_net1', i))

        for i, q_net in enumerate(self.q_nets2):
            torch.save(q_net.state_dict(), self.store_path.format('q_net2', i))

        for i, target_q_net in enumerate(self.target_q_nets1):
            torch.save(target_q_net.state_dict(), self.store_path.format('target_q_net1', i))

        for i, target_q_net in enumerate(self.target_q_nets2):
            torch.save(target_q_net.state_dict(), self.store_path.format('target_q_net2', i))

        for i, temperature in enumerate(self.entropy_temperatures):
            with open(self.store_path.format('temperature', i), 'wb') as f:
                pickle.dump(temperature, f)

        for i, p_opt in enumerate(self.policy_optimizers):
            torch.save(p_opt, self.store_path.format('policy_optimizer', i))

        for i, q_opt in enumerate(self.q1_optimizers):
            torch.save(q_opt, self.store_path.format('q1_optimizer', i))

        for i, q_opt in enumerate(self.q2_optimizers):
            torch.save(q_opt, self.store_path.format('q2_optimizer', i))

    def schedule_task(self, scheduled_tasks, learn_scheduler, oracle=True):
        """
        Scheduler
        :param scheduled_tasks: previously scheduled tasks
        :param learn_scheduler: if true -> SAC-X, else SAC-U or oracle
        :param oracle: use oracle scheduler instead of SAC-U
        :return: new task
        """
        if learn_scheduler is True: # SAC-X
            return self.scheduler.sample(scheduled_tasks)
        elif oracle: # Use oracle with room for random exploration
            choice = np.random.random()
            if choice < 0.7: # Oracle 70% of the time
                if len(scheduled_tasks) == 0:
                    return 0
                else:
                    return (self.tasks.index(scheduled_tasks[-1]) + 1) % 3
            else: # Random 30% of the time
                return random.choice([i for i in range(len(self.tasks) - 1)])
        else: #SAC-U
            return random.choice([i for i in range(len(self.tasks) - 1)])

    def store_rewards(self, episode_rewards, max_steps, scheduler_period, filename):
        with open(filename, 'w') as f:
            f.write("{}\n".format(max_steps))
            f.write("{}\n".format(scheduler_period))
            for item in episode_rewards:
                f.write("{}\n".format(item))

    def sample_trajectories(self, training_sequence_len=20, selective_sampling=True):
        """
        Samples trajectories from the replay memory. Used for retrace implementation
        :return: A minibatch (list) of random-length trajectories
        """
        minibatch = []
        threshold = 0.45
        for i in range(50):
            j = random.randint(0, len(self.main_replay_buffer) - 1)
            trajectory = self.main_replay_buffer[j]
            choice = np.random.random()
            initial_step = -1
            if choice < threshold and selective_sampling:
                non_zero_steps = self.non_zero_main_rewards[j]
                if len(non_zero_steps) > 0:
                    initial_step = max(0, non_zero_steps[random.randint(0, len(non_zero_steps)-1)]-training_sequence_len+1)
            if initial_step == -1:
                initial_step = random.randint(0, len(trajectory) - training_sequence_len)

            trajectory = trajectory[initial_step : initial_step+training_sequence_len]
            minibatch.append(trajectory)
        return minibatch

    def update_q_main(self, trajectories, gamma=0.99):
        """
        Retrace update of Q network for main task. Not converging.
        """

        # Calculation of retrace Q
        q_ret = torch.zeros(len(trajectories))

        curr_q = torch.zeros(len(trajectories))

        for i, trajectory in enumerate(trajectories):
            num_steps = len(trajectory)
            states = torch.FloatTensor([step[0] for step in trajectory])
            rewards = torch.FloatTensor([step[2] for step in trajectory])
            actions = torch.FloatTensor([step[1] for step in trajectory])
            z = torch.FloatTensor([step[5].squeeze(0).numpy() for step in trajectory])
            probs = torch.FloatTensor([step[6] for step in trajectory])
            # actions (for each task) for every state action pair in trajectory
            task_actions, _ = self.p_nets[3].sample(states)
            task_actions = task_actions*3.0
            task_probs = self.p_nets[3].get_probability(states, z)
            # Q-values (for each task) for every state and task-action pair in trajectory
            task_q = self.target_q_nets1[3].forward(states, task_actions)
            # Q-values (for each task) for every state and action pair in trajectory
            traj_q = self.target_q_nets1[3].forward(states, actions)
            # Q-value with current Q net
            curr_q_i = self.q_nets1[3].forward(states[0].unsqueeze(0), actions[0].unsqueeze(0))

            q_ret_i = 0

            for j in range(num_steps):
                # Discount factor
                discount = gamma ** (j)
                cj = 1.0
                for k in range(j):
                    ck = min(task_probs.data[j] / float(probs.data[j]), 1.0)
                    cj *= ck
                # Difference between the two q values
                del_q = task_q.data[0] - traj_q[j]
                # Retrace Q value is sum of discounted weighted rewards
                q_ret_i += discount * cj * (rewards[j, 3] + del_q)

            # Append retrace Q value to float tensor using index_fill
            q_ret[i] = q_ret_i[0]
            curr_q[i] = curr_q_i[0][0]

        # Critic loss uses retrace Q
        critic_loss = F.mse_loss(curr_q, q_ret.detach())
        # Use Huber Loss for critic
        #critic_loss = torch.nn.SmoothL1Loss()(task_q, torch.autograd.Variable(q_ret, requires_grad=False))
        self.q1_optimizers[3].zero_grad()
        critic_loss.backward()
        self.q1_optimizers[3].step()
        # target networks
        for target_param, param in zip(self.target_q_nets1[3].parameters(), self.q_nets1[3].parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)

    def update_p_main(self, trajectories, gamma=0.95):
        """
        Retrace update of policy network for main task. Not converging.
        """
        # Extract information out of trajectory
        states = torch.FloatTensor([trajectory[0][0] for trajectory in trajectories])
        # actions (for each task) for every state action pair in trajectory
        task_actions, task_log_prob = self.p_nets[3].sample(states)
        # Q-values (for each task) for every state and task-action pair in trajectory
        task_q = self.q_nets1[3].forward(states, task_actions)
        # delayed update for policy network and target q networks
        policy_loss = (self.entropy_temperatures[3][0] * task_log_prob - task_q).mean()
        self.policy_optimizers[3].zero_grad()
        policy_loss.backward()
        self.policy_optimizers[3].step()
        # update temperature. Recall info is stored as (alpha, target_entropy, log_alpha, alpha_optim)
        log_alpha = self.entropy_temperatures[3][2]
        target_entropy = self.entropy_temperatures[3][1]
        alpha_loss = (log_alpha * (-task_log_prob - target_entropy).detach()).mean()

        self.entropy_temperatures[3][3].zero_grad()  # Alpha optim
        alpha_loss.backward()
        self.entropy_temperatures[3][3].step()

        self.entropy_temperatures[3] = (
            log_alpha.exp(), self.entropy_temperatures[3][1], self.entropy_temperatures[3][2],
            self.entropy_temperatures[3][3])