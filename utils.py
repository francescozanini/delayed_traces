import numpy as np
from collections import deque


class Grid:

    def __init__(self, xlim, ylim, pacman_effect=True, delay=5, max_steps=10_000, delayed_actions=False):
        self.xlim = xlim
        self.ylim = ylim
        self.goal = None
        self.agent = None
        self.pacman_effect = pacman_effect
        self.delay = delay
        self.delayed_actions = delayed_actions
        if delayed_actions:
            self.actions_buffer = deque()
            for i in range(self.delay):  # reward of actions must be the same as delay of reward
                self.actions_buffer.append(4)  # filling the buffer with 'idle' action
        else:
            self.rewards_buffer = deque()
            for i in range(self.delay):
                self.rewards_buffer.append(0)  # reward for the initial steps in which no actual rewards are given
        if max_steps is not None:
            self.max_steps = max_steps
            self.steps = 0
        self.flag = False

    def reset(self, allow_start_goal=False):
        goal = [np.random.randint(self.xlim), np.random.randint(self.ylim)]
        if allow_start_goal:
            agent = [np.random.randint(self.xlim), np.random.randint(self.ylim)]
        else:
            agent = goal
            while agent == goal:
                agent = [np.random.randint(self.xlim), np.random.randint(self.ylim)]
        self.goal = goal
        self.agent = agent
        self.steps = 0
        self.rewards_buffer = deque()
        for i in range(self.delay):
            self.rewards_buffer.append(0)
        self.flag = False
        return tuple(self.get_agent())

    def get_agent(self):
        return self.agent.copy()

    def get_goal(self):
        return self.goal.copy()

    def get_action_space(self):
        return 5

    def get_max_steps(self):
        return self.max_steps

    def get_step(self):
        return self.steps

    def get_delay(self):
        return self.delay

    def step(self, action):
        self.steps += 1
        if self.delayed_actions:
            self.actions_buffer.append(action)
            action = self.actions_buffer.popleft()
        if action == 0:  # LEFT
            self.agent[1] -= 1
            if self.pacman_effect:
                if self.agent[1] < 0:
                    self.agent[1] += self.ylim
            else:
                if self.agent[1] < 0:
                    self.agent[1] = 0
        elif action == 2:  # RIGHT
            self.agent[1] += 1
            if self.pacman_effect:
                if self.agent[1] >= self.ylim:
                    self.agent[1] -= self.ylim
            else:
                if self.agent[1] >= self.ylim:
                    self.agent[1] = self.ylim - 1
        elif action == 1:  # UP
            self.agent[0] -= 1
            if self.pacman_effect:
                if self.agent[0] < 0:
                    self.agent[0] += self.xlim
            else:
                if self.agent[0] < 0:
                    self.agent[0] = 0
        elif action == 3:  # DOWN
            self.agent[0] += 1
            if self.pacman_effect:
                if self.agent[0] >= self.xlim:
                    self.agent[0] -= self.xlim
            else:
                if self.agent[0] >= self.xlim:
                    self.agent[0] = self.xlim - 1
        elif action == 4:  # IDLE
            pass
        reward = 0
        if self.agent == self.goal:
            reward = 10
        if not self.delayed_actions:
            self.rewards_buffer.append(reward)
            reward = self.rewards_buffer.popleft()
        if self.steps >= self.max_steps:
            self.flag = True
        return tuple(self.get_agent()), reward, self.flag
