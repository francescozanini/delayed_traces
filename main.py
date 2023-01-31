from utils import *
from tqdm import trange
import pickle


xdim = 5
ydim = 5
env = Grid(xdim, ydim, max_steps=20_000)
num_actions = env.get_action_space()
epsilon_0 = 3
learning_rate = 0.15
lamb = 0.9
gamma = 0.95
num_episodes = 1_000


# Standard approach
seed = 42
np.random.seed(seed)

storing_rewards = np.empty((num_episodes, env.get_max_steps()))
for episode in trange(num_episodes):

    # Build q_table
    q = np.zeros((xdim, ydim, num_actions))
    for x in range(xdim):
        for y in range(ydim):
            q[x, y] = [np.random.uniform(0, 1) for i in range(num_actions)]  # induce exploration

    # Re-initialise traces every episode
    traces = np.zeros((xdim, ydim, num_actions))

    state = env.reset()
    # random action (epsilon>1 in I step)
    draw = np.random.uniform(0, 1)
    action = np.random.randint(num_actions)
    #
    done = False
    while not done:
        state_next, reward, done = env.step(action)
        storing_rewards[episode, env.get_step()-1] = reward
        # replacing trace
        traces[(*state, action)] = 1
        #
        # eps-greedy action
        epsilon = epsilon_0 / np.sqrt(env.get_step())  # decaying exploration (already updated step)
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
seed = 42
np.random.seed(seed)

del_storing_rewards = np.empty((num_episodes, env.get_max_steps()))
for episode in trange(num_episodes):

    # Build delayed q_table
    del_q = np.zeros((xdim, ydim, num_actions))
    for x in range(xdim):
        for y in range(ydim):
            del_q[x, y] = [np.random.uniform(0, 1) for i in range(num_actions)]  # induce exploration

    del_traces = np.zeros((xdim, ydim, num_actions))

    state_buffer = deque()
    action_buffer = deque()
    state = env.reset()
    # random action (epsilon>1 in I step)
    action = np.random.randint(num_actions)
    #
    state_buffer.append(state)
    action_buffer.append(action)
    done = False
    while not done:
        state_next, reward, done = env.step(action)
        del_storing_rewards[episode, env.get_step()-1] = reward
        # eps-greedy action
        epsilon = epsilon_0 / np.sqrt(env.get_step())  # decaying exploration (already updated step)
        draw = np.random.uniform(0, 1)
        if draw < epsilon:
            action_next = np.random.randint(num_actions)
        else:
            action_next = np.argmax(del_q[state_next])
        #
        state_buffer.append(state_next)
        action_buffer.append(action_next)
        if env.get_step() > env.get_delay():
            state_delayed = state_buffer.popleft()
            action_delayed = action_buffer.popleft()
            state_next_delayed = state_buffer[0]
            action_next_delayed = action_buffer[0]
            # replacing trace
            del_traces[(*state_delayed, action_delayed)] = 1
            #
            # update del_q
            del_q = del_q + del_traces*learning_rate*(reward + gamma*del_q[(*state_next_delayed, action_next_delayed)] -
                                          del_q[(*state_delayed, action_delayed)])
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
