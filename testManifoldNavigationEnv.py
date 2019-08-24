#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  1 23:55:51 2019

@author: yangyutu123
"""

from ManifoldNavigationEnv import ManifoldNavigationEnv
import numpy as np
import math
import random



env = ManifoldNavigationEnv('config.json', 1)

step = 10000

state = env.reset()
print(state)

for i in range(step):
    state = env.currentState
    action = np.random.randn(3)
    nextState, reward, done, info = env.step(action)
    #print(done)
    #print(nextState)
    #print(info)
    #if i%2 == 0 and i < 10:
    #    env.step(100, np.array([u, v, 1.0]))
    #else:
    #    env.step(100, np.array([u, v, 0]))
        
