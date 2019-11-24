import gym
import itertools
import datetime
import numpy as np
import tensorflow as tf
from collections import Counter, deque

from dqn_cartpole.utils.linear_scheduler import linear_scheduler
from dqn_cartpole.utils.replay_buffer import replay_buffer
from dqn_cartpole.utils.eager_setup import eager_setup

eager_setup()  # soft allocation of GPU, otherwise this script uses up all GPU resource....


class Network(tf.keras.Model):
    def __init__(self, num_action):
        super(Network, self).__init__()
        self.dense1 = tf.keras.layers.Dense(16, activation='relu')
        self.dense2 = tf.keras.layers.Dense(16, activation='relu')
        self.dense3 = tf.keras.layers.Dense(16, activation='relu')
        self.pred = tf.keras.layers.Dense(num_action, activation='linear')

    @tf.function
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.pred(x)


class DQN:
    def __init__(self,
                 env,
                 optimizer,
                 loss_fn,
                 random_seed=123,
                 log_dir="Test",
                 num_episode=1000,
                 sync_freq=1000,
                 train_freq=1,
                 gamma=0.99,
                 decay_step=3000,
                 ep_end=0.02,
                 batch_size=32,
                 warm_start=1000,
                 memory_size=10000,
                 reward_buffer_len=10,
                 goal=190):
        tf.compat.v1.set_random_seed(random_seed)
        np.random.seed(seed=random_seed)
        self.global_ts = tf.compat.v1.train.create_global_step()
        self.env = env
        self.num_episode = num_episode
        self.sync_freq = sync_freq
        self.train_freq = train_freq
        self.gamma = gamma
        self.goal = goal
        self.ep_end = ep_end
        self.warm_start = warm_start
        self.batch_size = batch_size
        self.total_rewards = deque(maxlen=reward_buffer_len)
        self.loss_fn = loss_fn
        self.optimizer = optimizer  # we don't assign an optimiser to main/target individually
        self.main_model = Network(num_action=self.env.action_space.n)
        self.target_model = Network(num_action=self.env.action_space.n)  # we just deeply copy the model
        self.target_model.set_weights(self.main_model.get_weights())  # make sure to start with the same params
        self.memory = replay_buffer(memory_size=memory_size)  # see utils/replay_buffer.py
        self.scheduler = linear_scheduler(total_timesteps=decay_step, final_ratio=ep_end)
        self.summary_writer = tf.compat.v2.summary.create_file_writer("./logs/{}".format(log_dir))

    def choose_action(self, state):
        """ Epsilon-greedy policy """
        if np.random.random() < self.scheduler.get_value(timestep=self.global_ts.numpy()):
            action = self.env.action_space.sample()
        else:
            # main model defines the behaviour of the agent
            q_values = self.main_model(np.expand_dims(state, 0).astype(np.float32))
            action = np.argmax(q_values)
        return action

    def store(self, state, action, reward, next_state, done):
        """ Stores data at time-step t into the replay buffer """
        self.memory.add(state, action, reward, next_state, done)

    def learn(self):
        # randomly sample from the replay buffer
        batch = self.memory.sample(batch_size=self.batch_size)
        states, actions, rewards, next_states, dones = batch

        # format the sampled data before feeding to the network
        states = np.array(states, dtype=np.float32)  # batch_size x w x h x c
        next_states = np.array(next_states, dtype=np.float32)  # batch_size x w x h x c
        actions = np.array(actions, dtype=np.int32)  # (batch_size,)
        rewards = np.array(rewards, dtype=np.float32)  # (batch_size,)
        dones = np.array(dones, dtype=np.float32)  # (batch_size,)
        self._learn(states, actions, rewards, next_states, dones)

    @tf.function
    def _learn(self, states, actions, rewards, next_states, dones):
        # ===== make sure to fit all process to compute gradients within this Tape context!! =====
        with tf.GradientTape() as tape:
            q_tp1 = self.target_model(next_states)  # batch_size x num_action
            q_t = self.main_model(states)  # batch_size x num_action
            td_target = rewards + self.gamma * tf.math.reduce_max(q_tp1, axis=-1) * (1. - dones)  # (batch_size,)
            td_target = tf.stop_gradient(td_target)

            # get the q-values which is associated with actually taken actions in a game
            idx = tf.concat([tf.expand_dims(tf.range(0, actions.shape[0]), 1), tf.expand_dims(actions, 1)], axis=-1)
            chosen_q = tf.gather_nd(q_t, idx)  # (batch_size,)
            td_error = self.loss_fn(td_target, chosen_q)  # scalar

        # get gradients
        grads = tape.gradient(td_error, self.main_model.trainable_variables)

        # apply processed gradients to the network
        self.optimizer.apply_gradients(zip(grads, self.main_model.trainable_variables))

        ts = tf.compat.v1.train.get_global_step()
        tf.compat.v2.summary.scalar("agent/loss_td_error", td_error, step=ts)
        tf.compat.v2.summary.scalar("agent/mean_diff_q_tp1_q_t", tf.math.reduce_mean(q_tp1 - q_t), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_td_target", tf.math.reduce_mean(td_target), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_q_tp1", tf.math.reduce_mean(q_tp1), step=ts)
        tf.compat.v2.summary.scalar("agent/mean_q_t", tf.math.reduce_mean(q_t), step=ts)

    def rl(self):
        with self.summary_writer.as_default():
            for i in range(self.num_episode):
                state = self.env.reset()
                episode_reward = 0
                taken_actions = list()
                for _ in itertools.count():
                    action = self.choose_action(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.store(state, action, reward, next_state, done)
                    episode_reward += reward
                    taken_actions.append(action)
                    state = next_state
                    self.global_ts.assign_add(1)  # increment the global step

                    if self.global_ts.numpy() >= self.warm_start:
                        if self.global_ts.numpy() % self.train_freq == 0:
                            # update the main model
                            self.learn()

                        if self.global_ts.numpy() % self.sync_freq == 0:
                            # Hard update, but you can do soft-update as well
                            # soft-update: target_w = target_w * (alpha) + current_w * (1-alpha)
                            self.target_model.set_weights(self.main_model.get_weights())

                    if done:
                        break

                """ === After One episode === """
                # Logging on console is really important to understand what's going on during execution
                self.total_rewards.append(episode_reward)
                print("| Ep {} | Time-step {} | Reward {} | Mean Reward {:.3f} | Action {} | Epsilon {:.4f}|".format(
                    i, self.global_ts.numpy(), episode_reward, np.mean(self.total_rewards),
                    dict(Counter(taken_actions)), self.scheduler.get_value(timestep=self.global_ts.numpy()))
                )
                tf.compat.v2.summary.scalar("train/reward", episode_reward, step=self.global_ts.numpy())
                if self.global_ts.numpy() > self.warm_start:
                    tf.compat.v2.summary.scalar("train/MAR", np.mean(self.total_rewards), step=self.global_ts.numpy())
                tf.compat.v2.summary.histogram("train/taken actions", taken_actions, step=self.global_ts.numpy())

                # Early Stopping Condition
                if np.mean(self.total_rewards) >= self.goal:
                    break

            """ === After all episodes === """
            self.env.close()


if __name__ == '__main__':
    env = gym.make('CartPole-v0')

    # define log dir name
    suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    log_dir = "Test-{}".format(suffix)
    # log_dir = "tf.compat.v1.train.AdamOptimizer"

    # === Play with some possible combinations ===
    loss_fn = tf.compat.v1.losses.mean_squared_error
    # loss_fn = tf.compat.v1.losses.huber_loss
    # optimizer = tf.optimizers.RMSprop(learning_rate=0.00025,
    #                                   decay=0.95,
    #                                   momentum=0.0,
    #                                   epsilon=0.00001,
    #                                   centered=True)
    # optimizer = tf.optimizers.Adam(learning_rate=1e-5)

    optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=0.00025,
                                                    decay=0.95,
                                                    momentum=0.0,
                                                    epsilon=0.00001,
                                                    centered=True)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=1e-5)

    agent = DQN(env=env,
                loss_fn=loss_fn,
                optimizer=optimizer,
                log_dir=log_dir)
    agent.rl()
