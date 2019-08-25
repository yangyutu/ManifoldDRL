

import json
import numpy as np
from ManifoldNavigationEnv import ManifoldDirectedScheduler



configName = 'config.json'
with open(configName,'r') as f:
    config = json.load(f)

originTrainData = np.genfromtxt('originTrainData.txt')
lowDEmbedding = np.genfromtxt('lowDEmbedding.txt')

scheduler = ManifoldDirectedScheduler(config, 'originTrainData.txt', 'lowDEmbedding.txt')


numEpisode = 1000
targetState = config['targetState']
output = []
output.append(targetState)
for i in range(numEpisode):
    state = scheduler.getNextStartState(i)
    distanctVec = state - targetState
    distance = np.linalg.norm(distanctVec, ord=2)
    print(state, distance)
    output.append(state)

np.savetxt('schduleStateDirected.txt', output)
