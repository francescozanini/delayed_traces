import matplotlib.pyplot as plt
import pickle
import numpy as np


with open('rewards.pickle', 'rb') as file:
    rewards = pickle.load(file)
with open('delayed_rewards.pickle', 'rb') as file:
    del_rewards = pickle.load(file)

window = 100

avg_over_episodes = np.mean(rewards, axis=0)
moving_avg_over_steps = np.convolve(avg_over_episodes, np.ones(window)/window, mode='valid')  # with 'valid' starts from step=window
del_avg_over_episodes = np.mean(del_rewards, axis=0)
del_moving_avg_over_steps = np.convolve(del_avg_over_episodes, np.ones(window)/window, mode='valid')  # with 'valid' starts from step=window

plt.plot([i for i in range(len(moving_avg_over_steps))], moving_avg_over_steps, label='standard')
plt.plot([i for i in range(len(del_moving_avg_over_steps))], del_moving_avg_over_steps, label='delayed')
plt.xlabel('Step')
plt.ylabel('Reward')
plt.legend(loc='best')
plt.savefig('comparison.pdf')
#plt.show()
