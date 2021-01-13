import random
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import pickle
import torch.nn as nn
import numpy as np

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
                 store_path="./checkpoints/simple_env/{}_{}.checkpoint",
                 load_from=None):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.env = env
        self.action_range = [env.action_space().low, env.action_space().high]
        self.obs_dim = env.get_state_size()[0]
        self.action_dim = env.action_space().shape[0]

        # hyperparameters
        self.gamma = gamma
        self.tau = tau
        self.update_step = 0
        self.delay_step = 2
        self.schedule_period = schedule_period

        self.storing_frequence=storing_frequence
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

    def init_p_models(self, load_path, share_layer):
        policy_nets = []

        if share_layer is True:
            shared_layer = nn.Linear(self.obs_dim, 256)
        else:
            shared_layer = None

        for i, task in enumerate(self.tasks):
            if load_path is None:
                policy_nets.append(PolicyNetwork(self.obs_dim, self.action_dim, shared_layer=shared_layer).to(self.device))
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
                q_nets1.append(SoftQNetwork(self.obs_dim, self.action_dim, shared_layer=shared_layer_1).to(self.device))
                q_nets2.append(SoftQNetwork(self.obs_dim, self.action_dim, shared_layer=shared_layer_2).to(self.device))
                target_q_nets1.append(SoftQNetwork(self.obs_dim, self.action_dim, shared_layer=shared_layer_t_1).to(self.device))
                target_q_nets2.append(SoftQNetwork(self.obs_dim, self.action_dim, shared_layer=shared_layer_t_2).to(self.device))

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
        return action * (self.action_range[1] - self.action_range[0]) / 2.0 + \
               (self.action_range[1] + self.action_range[0]) / 2.0

    def get_action(self, state, task):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        mean, log_std = self.p_nets[task].forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        action = action.cpu().detach().squeeze(0).numpy()

        return self.rescale_action(action)

    def train(self):
        episode_rewards = []

        for episode in range(self.max_episodes):
            task = -1
            state = self.env.reset()
            episode_reward = 0
            scheduled_tasks = []
            scheduled_task_step = 0
            for step in range(self.max_steps):

                if (step-scheduled_task_step) % self.schedule_period == 0:
                    task = self.schedule_task(task)
                    scheduled_tasks.append(task)
                    scheduled_task_step = step
                    print("Switching to ", self.tasks[task])

                action = self.get_action(state, task) # Sample new action using the task policy network
                next_state, reward, done, visited_circles = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                episode_reward += reward[task]

                if len(self.replay_buffer) > self.training_batch_size:
                    #print("Training")
                    self.update(self.training_batch_size)

                if done or step == self.max_steps - 1:
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    break

                state = next_state

                #Schedule new task
                if visited_circles[task] == 1:
                    task = self.schedule_task(task)
                    scheduled_tasks.append(task)
                    scheduled_task_step = step
                    print("Switching to ", self.tasks[task])
            if (episode+1) % self.storing_frequence == 0:
                self.store_models()

        return episode_rewards

    def update(self, batch_size):
        for i, task in enumerate(self.tasks):
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
            states = torch.FloatTensor(states).to(self.device)
            actions = torch.FloatTensor(actions).to(self.device)
            rewards = torch.FloatTensor(rewards).to(self.device)
            rewards = rewards.reshape(batch_size, rewards.shape[-1]) # Reshape
            rewards = rewards[:, i:i+1] # Select only rewards for this task
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

            # delayed update for policy network and target q networks
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

            # update temperature. Recall info is stored as (alpha, target_entropy, log_alpha, alpha_optim)
            log_alpha = self.entropy_temperatures[i][2]
            target_entropy = self.entropy_temperatures[i][1]
            alpha_loss = (log_alpha * (-log_pi - target_entropy).detach()).mean()

            self.entropy_temperatures[i][3].zero_grad() # Alpha optim
            alpha_loss.backward()
            self.entropy_temperatures[i][3].step()

            self.entropy_temperatures[i] = (log_alpha.exp(), self.entropy_temperatures[i][1], self.entropy_temperatures[i][2], self.entropy_temperatures[i][3])

        self.update_step += 1

    def test(self, num_episodes=3):
        episode_rewards = []

        for episode in range(num_episodes):
            print("Testing episode {}\n".format(episode))
            state = self.env.reset()
            episode_reward = 0
            for step in range(500):
                action = self.get_action(state, 3)  # Sample new action using the main task policy network
                next_state, reward, done, visited_circles = self.env.step(action)
                episode_reward += reward[3]

                if done or step == self.max_steps - 1:
                    if done:
                         print("Task completed")
                    episode_rewards.append(episode_reward)
                    print("Episode " + str(episode) + ": " + str(episode_reward))
                    break

                state = next_state

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

    def schedule_task(self, last_task=-1):
        return (last_task+1)%3
        #return random.choice([i for i in range(len(self.tasks)-1)])

    def store_rewards(self, episode_rewards, max_steps, scheduler_period, filename):
        with open(filename, 'w') as f:
            f.write("{}\n".format(max_steps))
            f.write("{}\n".format(scheduler_period))
            for item in episode_rewards:
                f.write("{}\n".format(item))
