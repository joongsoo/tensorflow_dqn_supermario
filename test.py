# -*- coding: utf-8 -*-
import numpy as np
import random
import dqn
from collections import deque
from data import env


env = env.Env()


dis = 0.9
REPLAY_MEMORY = 50000


def main():

    max_episodes = 1500
    for episode in range(max_episodes):
        done = False
        step_count = 0
        env.reset()

        while not done:
            state, reward, done = env.step(env.get_random_actions()[0])

            step_count += 1



if __name__ == "__main__":
    main()
