import gym
import itertools
import numpy as np
import tensorflow as tf
from collections import Counter
import matplotlib.pyplot as plt

from dqn_cartpole.utils.linear_scheduler import linear_scheduler
from dqn_cartpole.utils.replay_buffer import replay_buffer
from dqn_cartpole.utils.eager_setup import eager_setup

eager_setup()  # soft allocation of GPU, otherwise this script uses up all GPU resource....


class DQN:
    def __init__(self,
                 env,
                 optimizer,
                 num_episode=1000,
                 sync_freq=100,
                 train_freq=1,
                 gamma=0.99,
                 decay_step=3000,
                 ep_end=0.02,
                 batch_size=32,
                 warm_start=1000,
                 memory_size=10000):
        self.global_ts = 0
        self.env = env
        self.num_episode = num_episode
        self.sync_freq = sync_freq
        self.train_freq = train_freq
        self.gamma = gamma
        self.ep_end = ep_end
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.total_rewards = []
        self.optimizer = optimizer  # we don't assign an optimiser to main/target individually
        self.main_model = self._build_net()
        self.target_model = self._build_net()  # we just deeply copy the model
        self.target_model.set_weights(self.main_model.get_weights())  # make sure to start with the same params.
        self.memory = replay_buffer(memory_size=memory_size)  # see utils/replay_buffer.py
        self.scheduler = linear_scheduler(total_timesteps=decay_step, final_ratio=ep_end)

    def _build_net(self):
        """ Instantiate the Keras model """
        inputs = tf.keras.layers.Input(shape=self.env.observation_space.shape)
        x = tf.keras.layers.Dense(16, activation='relu')(inputs)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        x = tf.keras.layers.Dense(16, activation='relu')(x)
        q_vals = tf.keras.layers.Dense(self.env.action_space.n, activation='linear')(x)
        model = tf.keras.models.Model(inputs=inputs, outputs=q_vals)
        model.compile(loss='mse', optimizer=self.optimizer)
        return model

    def choose_action(self, state):
        """ Epsilon-greedy policy """
        if np.random.uniform() < self.scheduler.get_value(timestep=self.global_ts):
            action = self.env.action_space.sample()
        else:
            state_action = self.main_model.predict(state[None, :])  # main model defines the behaviour of the agent
            action = np.argmax(state_action)
        return action

    def store(self, state, action, reward, next_state, done):
        """ Stores data at time-step t into the replay buffer """
        self.memory.add(state, action, reward, next_state, done)

    def plot(self, reward):
        plt.plot(reward)
        plt.show()

    def learn(self):
        batch = self.memory.sample(batch_size=self.batch_size)
        states, actions, rewards, next_states, dones = batch
        next_q = np.max(self.target_model.predict(next_states), axis=-1)
        current_q = self.main_model.predict(states)
        td_target = rewards + self.gamma * next_q * (1 - dones)
        td_target = tf.stop_gradient(td_target)  # just in case to avoid the back flow of gradient computation
        current_q[np.arange(self.batch_size, dtype='int8'), actions] = td_target

        # log_dir = "logs\\DQN\\"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.main_model.fit(states, current_q, epochs=1, verbose=0, callbacks=None)

    def rl(self):
        for i in range(self.num_episode):
            state = self.env.reset()
            episode_reward = 0
            taken_actions = list()
            for t in itertools.count():
                action = self.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.store(state, action, reward, next_state, done)
                episode_reward += reward
                taken_actions.append(action)

                if len(self.memory) >= self.warm_start:
                    if (t + 1) % self.train_freq == 0:
                        # update the main model
                        self.learn()

                        if (self.global_ts + 1) % self.sync_freq == 0:
                            # we do Hard update, but you can do soft-update as well
                            # soft-update: target_w = target_w * (alpha) + current_w * (1-alpha)
                            self.target_model.set_weights(self.main_model.get_weights())

                if done:
                    break

                state = next_state
                self.global_ts += 1

            """ === After One episode === """
            # Logging on console is really important to understand what's going on during execution
            self.total_rewards.append(episode_reward)
            print("| Ep {} | Time-step {} | Reward {} | Mean Reward {:.3f} | Action {} | Epsilon {:.4f}|".format(
                i, self.global_ts, episode_reward, np.mean(self.total_rewards), dict(Counter(taken_actions)),
                self.scheduler.get_value(timestep=self.global_ts))
            )

        """ === After all episodes === """
        self.env.close()
        self.plot(self.total_rewards)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    agent = DQN(env=env,
                optimizer=tf.optimizers.RMSprop(learning_rate=0.00025,
                                                decay=0.95,
                                                momentum=0.0,
                                                epsilon=0.00001,
                                                centered=True))
    agent.rl()
