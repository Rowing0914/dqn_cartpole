import gym
import time
import itertools
from dqn_cartpole.utils.replay_buffer import replay_buffer

MEMORY_SIZE = 10000
NUM_EPISODE = 1000


def _test():
    env = gym.make('CartPole-v0')
    buffer = replay_buffer(memory_size=MEMORY_SIZE)
    for ep in range(NUM_EPISODE):
        state = env.reset()
        for i in itertools.count():
            action = env.action_space.sample()
            next_state, reward, done, info = env.step(action=action)
            buffer.add(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

    env.close()
    assert len(buffer) == MEMORY_SIZE
    print("Len of buffer: {}".format(len(buffer)))

    print("=== Performance Test ===")
    begin = time.time()
    for _ in range(1000): buffer.sample(batch_size=32)
    exec_time = time.time() - begin
    print("[buffer.sample] Took: {:.4f}".format(exec_time))
    # [buffer.sample] Took: 0.0790
    print("Ave Exec time for sample: {}".format(exec_time / 1000))
    # Ave Exec time for sample: 7.993650436401368e-05


if __name__ == '__main__':
    _test()
