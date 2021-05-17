import sys

import random
import numpy as np
import cv2 as cv
import gym
import gym.spaces as spaces
import concurrent.futures
DO_NO_THING = 0
UP = 1
DOWN = 2
LEFT = 3
RIGHT = 4
class EscapeEnv(gym.Env):


    _ACTION = {DO_NO_THING, UP, DOWN, LEFT, RIGHT}

    def __init__(self, width=1366, high=768, boom_count=1000, size_r=5, observation_range=100):
        self.boom_count = boom_count
        self.width = width
        self.high = high
        self.observation_range = observation_range
        self.x = int(width/2)
        self.y = int(high/2)
        self.size_r = size_r
        self.action_space = spaces.Discrete(len(self._ACTION))
        # self.action_space = spaces.Box(1, 10, shape=(1,))
        self.observation_space = spaces.Box(0, 255, shape=(self.observation_range*2, self.observation_range*2, 1), dtype=np.uint8)
        self.map = None
        self.boom_list = []

    def step(self, action):
        if action == DO_NO_THING:
            pass
        elif action == UP:
            self.y -= 1
            if self.y < self.observation_range:
                self.y = self.observation_range
        elif action == DOWN:
            self.y += 1
            if self.y > self.high-self.observation_range-1:
                self.y = self.high-self.observation_range-1
        elif action == LEFT:
            self.x -= 1
            if self.x < self.observation_range:
                self.x = self.observation_range
        elif action == RIGHT:
            self.x += 1
            if self.x > self.width-self.observation_range-1:
                self.x = self.width-self.observation_range-1
        #
        crash = self.boom_run()

        #
        self.map = np.zeros((self.high, self.width, 1), dtype=np.uint8)
        for boom in self.boom_list:
            self.map[int(boom['y'])][int(boom['x'])][0] = 255

        #
        reward = 0.01
        done = False
        if crash:
            reward = -10
            done = True

        observation = self.map[self.y-self.observation_range:self.y+self.observation_range, self.x-self.observation_range:self.x+self.observation_range][:]
        return observation, reward, done, {}

    def reset(self):
        self.x = int(self.width / 2)
        self.y = int(self.high / 2)
        self.make_boom(self.boom_count)
        self.map = np.zeros((self.high, self.width, 1), dtype=np.uint8)
        for boom in self.boom_list:
            self.map[int(boom['y'])][int(boom['x'])][0] = 255
        observation = self.map[self.y - self.observation_range:self.y + self.observation_range, self.x - self.observation_range:self.x + self.observation_range][:]

        return observation

    def render(self, mode='human'):
        if mode == 'human':
            cv.circle(self.map, (self.x, self.y), self.size_r, (255,))
            cv.imshow("", self.map)
            cv.waitKey(1)

    def make_boom(self, boom_count):
        self.boom_list = []
        for i in range(boom_count):
            self.boom_list.append({'x': random.randint(0, self.width-1), 'y': 0, 'vx': random.random()-0.5, 'vy': random.random()+0.5})

    def boom_run(self):
        crash = False
        for boom in self.boom_list:
            boom['x'] += boom['vx']
            boom['y'] += boom['vy']

            if(boom['x']-self.x)**2 + (boom['y']-self.y)**2 <= self.size_r**2:
                crash = True

            if boom['x'] <= 0:
                boom['x'] = 0
                boom['vx'] = random.random()+0.5
            if boom['x'] >= self.width-1:
                boom['x'] = self.width-1
                boom['vx'] = -random.random()-0.5
            if boom['y'] <= 0:
                boom['y'] = 0
                boom['vy'] = random.random()+0.5
            if boom['y'] >= self.high-1:
                boom['y'] = self.high-1
                boom['y'] = -random.random()-0.5
        return crash


if __name__ == "__main__":
    env = EscapeEnv()
    env.reset()
    while True:
        observation, reward, done, _ = env.step(0)
        # cv.imshow('observation', observation)
        # cv.waitKey(1)
        env.render()