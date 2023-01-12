import matplotlib.pyplot as plt
import pickle
import numpy as np
from IPython import embed


with open('delayed_rewards.pickle', 'rb') as file:
    rewards = pickle.load(file)

window = 10000
moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')  # with 'valid' starts from step=window

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.show()
