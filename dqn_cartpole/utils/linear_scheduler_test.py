import matplotlib.pyplot as plt
import numpy as np
from dqn_cartpole.utils.linear_scheduler import linear_scheduler

scheduler = linear_scheduler(total_timesteps=3000, final_ratio=0.02, init_ratio=1.0)

res = [scheduler.get_value(timestep=ts) for ts in range(4000)]

plt.plot(np.asarray(res))
plt.grid()
plt.title("Linear Scheduler Test")
plt.show()
