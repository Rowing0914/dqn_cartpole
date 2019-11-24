"""
borrowed the code from
https://github.com/TianhongDai/reinforcement-learning-algorithms/blob/a3d88c0489a9c42e47d7cdfc39d0ebe6f33dbfe4/rl_utils/experience_replay/experience_replay.py#L9
"""

import numpy as np
import random


class replay_buffer:
    def __init__(self, memory_size):
        self.storge = []
        self.memory_size = memory_size
        self.next_idx = 0

    def __len__(self):
        return len(self.storge)

    # add the samples
    def add(self, obs, action, reward, obs_, done):
        data = (obs, action, reward, obs_, done)
        if self.next_idx >= len(self.storge):
            self.storge.append(data)
        else:
            self.storge[self.next_idx] = data
        # get the next idx
        self.next_idx = (self.next_idx + 1) % self.memory_size

    # encode samples
    def _encode_sample(self, idx):
        obses, actions, rewards, obses_, dones = [], [], [], [], []
        for i in idx:
            data = self.storge[i]
            obs, action, reward, obs_, done = data
            obses.append(np.array(obs, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_.append(np.array(obs_, copy=False))
            dones.append(done)
        return np.array(obses), np.array(actions), np.array(rewards), np.array(obses_), np.array(dones)

    # sample from the memory
    def sample(self, batch_size):
        idxes = [random.randint(0, len(self.storge) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes)
