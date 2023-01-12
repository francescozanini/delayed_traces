from utils import *
from tqdm import trange
import pickle


xdim = 5
ydim = 5
env = Grid(xdim, ydim)
num_actions = env.get_action_space()
epsilon = 0.05
learning_rate = 0.05
lamb = 0.9
gamma = 0.95
num_episodes = 1_000


# Standard approach

q = np.zeros((xdim, ydim, num_actions))
for x in range(xdim):
    for y in range(ydim):
        q[x, y] = [np.random.uniform(0, 1) for i in range(num_actions)]  # induce exploration

traces = np.zeros((xdim, ydim, num_actions))

storing_rewards = []
for episode in trange(num_episodes):
    state = env.reset()
    # eps-greedy action
    draw = np.random.uniform(0, 1)
    if draw < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q[state])
    #
    done = False
    while not done:
        state_next, reward, done = env.step(action)
        storing_rewards.append(reward)
        # replacing trace
        traces[(*state, action)] = 1
        #
        # eps-greedy action
        draw = np.random.uniform(0, 1)
        if draw < epsilon:
            action_next = np.random.randint(num_actions)
        else:
            action_next = np.argmax(q[state_next])
        #
        # update q
        q = q + traces*learning_rate*(reward + gamma*q[(*state_next, action_next)] - q[(*state, action)])
        #
        # replacing trace
        traces *= lamb*gamma
        #
        state = state_next
        action = action_next



# DELAYED TRACES

del_q = np.zeros((xdim, ydim, num_actions))
for x in range(xdim):
    for y in range(ydim):
        del_q[x, y] = [np.random.uniform(0, 1) for i in range(num_actions)]  # induce exploration

del_traces = np.zeros((xdim, ydim, num_actions))

del_storing_rewards = []
for episode in trange(num_episodes):
    state_buffer = deque()
    action_buffer = deque()
    state = env.reset()
    # eps-greedy action
    draw = np.random.uniform(0, 1)
    if draw < epsilon:
        action = np.random.randint(num_actions)
    else:
        action = np.argmax(q[state])
    #
    state_buffer.append(state)
    action_buffer.append(action)
    done = False
    while not done:
        state_next, reward, done = env.step(action)
        del_storing_rewards.append(reward)
        # replacing trace
        del_traces[(*state, action)] = 1
        #
        # eps-greedy action
        draw = np.random.uniform(0, 1)
        if draw < epsilon:
            action_next = np.random.randint(num_actions)
        else:
            action_next = np.argmax(q[state_next])
        #
        state_buffer.append(state_next)
        action_buffer.append(action_next)
        if env.get_step() > 4:
            state_delayed = state_buffer.popleft()
            action_delayed = action_buffer.popleft()
            state_next_delayed = state_buffer[0]
            action_next_delayed = action_buffer[0]
            # update q
            q = q + del_traces*learning_rate*(reward + gamma*q[(*state_next_delayed, action_next_delayed)] -
                                          q[(*state_delayed, action_delayed)])
            #
            # replacing trace
            del_traces *= lamb*gamma
            #
        state = state_next
        action = action_next

# Storing
with open('rewards.pickle', 'wb') as door:
    pickle.dump(storing_rewards, door)
with open('delayed_rewards.pickle', 'wb') as door2:
    pickle.dump(del_storing_rewards, door2)
