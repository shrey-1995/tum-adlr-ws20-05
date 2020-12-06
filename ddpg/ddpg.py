import numpy as np
import tensorflow as tf
from datetime import datetime
from ddpg.models import Critic_gen, Actor_gen
from collections import deque
from sys import exit
import random

class Buffer:

    def __init__(self, size, obs_dim=None, act_dim=None):
        self.max_size = size
        self.buffer = deque(maxlen=size)
        self.size = 0

    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.size = min(self.size + 1, self.max_size)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)
        # np.random.seed(0)
        batch = np.random.randint(0, len(self.buffer), size=batch_size)
        for experience in batch:
            state, action, reward, next_state, done = self.buffer[experience]
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)

        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)


class DDPGAgent:

    def __init__(self, env, gamma, tau, buffer_maxlen, critic_learning_rate, actor_learning_rate):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_max = env.action_space.high[0]
        # self.action_max = 1

        # hyperparameters
        self.env = env
        self.gamma = gamma
        self.tau = tau

        # Network layers
        actor_layer = [512, 200, 128]
        critic_layer = [1024, 512, 300, 1]

        # Main network outputs
        self.mu = Actor_gen((self.obs_dim), (self.action_dim), actor_layer, self.action_max)
        self.q_mu = Critic_gen((self.obs_dim), (self.action_dim), critic_layer)

        # Target networks
        self.mu_target = Actor_gen((self.obs_dim), (self.action_dim), actor_layer, self.action_max)
        self.q_mu_target = Critic_gen((self.obs_dim), (self.action_dim), critic_layer)

        # Copying weights in,
        self.mu_target.set_weights(self.mu.get_weights())
        self.q_mu_target.set_weights(self.q_mu.get_weights())

        # optimizers
        self.mu_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_learning_rate)
        self.q_mu_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_learning_rate)

        self.replay_buffer = Buffer(size=buffer_maxlen, obs_dim=self.obs_dim, act_dim=self.action_dim)

        self.q_losses = []

        self.mu_losses = []

    def get_action(self, s, noise_scale):
        a = self.mu.predict(s.reshape(1, -1))[0]
        a += noise_scale * np.random.randn(self.action_dim)
        return np.clip(a, -self.action_max, self.action_max)

    def test_agent(self, num_episodes=5, max_episode_length=500):
        test_returns = []

        for j in range(num_episodes):
            s, episode_return, episode_length, d = self.env.reset(), 0, 0, False

            while not (d or (episode_length == max_episode_length)):
                # Take deterministic actions at test time (noise_scale=0)
                self.env.render()
                s, r, d, _ = self.env.step(self.get_action(s, 0))
                episode_return += r
                episode_length += 1

            print('Testing episode {}/{} ===> Episode return = {} | Episode length = {}:'.format(j, num_episodes, episode_return, episode_length))
            test_returns.append(episode_return)

        return test_returns

    def update(self, batch_size):
        X, A, R, X2, D = self.replay_buffer.sample(batch_size)
        X = np.asarray(X, dtype=np.float32)
        A = np.asarray(A, dtype=np.float32)
        R = np.asarray(R, dtype=np.float32)
        X2 = np.asarray(X2, dtype=np.float32)

        Xten = tf.convert_to_tensor(X)

        # Updating Ze Critic
        with tf.GradientTape() as tape:
            A2 = self.mu_target(X2)
            q_target = R + self.gamma * self.q_mu_target([X2, A2])
            qvals = self.q_mu([X, A])
            q_loss = tf.reduce_mean((qvals - q_target) ** 2)
            grads_q = tape.gradient(q_loss, self.q_mu.trainable_variables)
        self.q_mu_optimizer.apply_gradients(zip(grads_q, self.q_mu.trainable_variables))
        self.q_losses.append(q_loss)

        # Updating ZE Actor
        with tf.GradientTape() as tape2:
            A_mu = self.mu(X)
            Q_mu = self.q_mu([X, A_mu])
            mu_loss = -tf.reduce_mean(Q_mu)
            grads_mu = tape2.gradient(mu_loss, self.mu.trainable_variables)
        self.mu_losses.append(mu_loss)
        self.mu_optimizer.apply_gradients(zip(grads_mu, self.mu.trainable_variables))

        # update target networks
        ## Updating both netwokrs
        # # updating q_mu network

        temp1 = np.array(self.q_mu_target.get_weights())
        temp2 = np.array(self.q_mu.get_weights())
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.q_mu_target.set_weights(temp3)

        # updating mu network
        temp1 = np.array(self.mu_target.get_weights())
        temp2 = np.array(self.mu.get_weights())
        temp3 = self.tau * temp2 + (1 - self.tau) * temp1
        self.mu_target.set_weights(temp3)

        self.mu_target.save('./models/mu_target.h5')
        self.q_mu_target.save('./models/q_mu_target.h5')
        self.mu.save('./models/mu.h5')
        self.q_mu.save('./models/q_mu.h5')
