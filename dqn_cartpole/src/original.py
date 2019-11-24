import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from random import sample


class q:
    def __init__(self, env=None, lr=0.0001, gamma=0.9, epsilon=1, decay=0.995):
        self.env = env
        self.action = env.action_space
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.reward = []
        self.eval_model = self._build_net('eval')
        self.target_model = self._build_net('target')
        self.memory = deque(maxlen=10000)

    def _build_net(self, name):
        eval_inputs = tf.keras.layers.Input(shape=(4,))
        x = tf.keras.layers.Dense(64, activation='relu')(eval_inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        q_eval = tf.keras.layers.Dense(2, activation='linear')(x)
        model = tf.keras.models.Model(eval_inputs, q_eval, name=name)
        model.compile(loss='mse', optimizer=tf.optimizers.Adam(lr=self.lr))
        return model

    def choose_action(self, state):
        if np.random.uniform() < self.epsilon:
            action = self.action.sample()
        else:
            state_action = self.eval_model.predict(state[None, :])
            action = np.argmax(state_action)
        return action

    def store(self, state, action, reward, done, next_state):
        sets = np.hstack([state, action, reward, done, next_state])
        self.memory.append(sets)
        return None

    def plot(self, reward):
        plt.plot(reward)
        plt.show()

    def learn(self):
        if len(self.memory) < 1000:
            return None
        minibatch = np.array(sample(self.memory, 32))
        next_state = minibatch[:, -4:]
        max_future_q = np.max(self.target_model.predict(next_state), axis=1)
        states = minibatch[:, :4]
        current_q = self.eval_model.predict(states)
        actions = minibatch[:, 4].astype('int8')
        rewards = minibatch[:, 5]
        done = minibatch[:, 6]
        current_q[np.arange(32, dtype='int8'), actions] = rewards + self.gamma * max_future_q * np.logical_not(done)

        # log_dir = "logs\\DQN\\"
        # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
        self.eval_model.fit(states, current_q, epochs=1, verbose=0, callbacks=None)
        if self.epsilon > 0.01:
            self.epsilon *= self.decay

    def rl(self):
        for i in range(500):
            state = self.env.reset()
            reward = 0
            done = False
            step = 0
            while not done:
                # self.env.render()
                action = self.choose_action(state)
                next_state, r, done, _ = self.env.step(action)
                self.store(state, action, r if not done else -5, done, next_state)
                reward += r
                step += 1
                self.learn()
                if step % 5 == 0 or done:
                    self.target_model.set_weights(self.eval_model.get_weights())
                next_state = state
            self.reward.append(reward)
            print(f'In episode {i}, reward {reward}, mean reward {np.mean(self.reward)}')
        self.plot(self.reward)


if __name__ == '__main__':
    env = gym.make('CartPole-v0')
    a = q(env)
    a.rl()
