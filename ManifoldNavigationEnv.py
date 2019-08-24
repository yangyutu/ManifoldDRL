from ManifoldNavigationModelPython import ManifoldNavigationModelPython
import numpy as np
import random
import json
import os
from sklearn.metrics.pairwise import euclidean_distances
import math
import sys
from abc import ABC
from abc import abstractmethod

class ManifoldScheduler(ABC):
    def __init__(self, config, originTrainDataFile, lowDEmbeddingFile):

        self.config = config

        self.originTrainData = np.genfromtxt(originTrainDataFile)
        self.lowDEmbedding = np.genfromtxt(lowDEmbeddingFile)
        self.startConfig = self.config['currentState']
        self.dim = self.lowDEmbedding.shape[1]
        self.targetConfig = self.config['targetState']
        self.methodName = 'ManifoldScheduler'
        dist = euclidean_distances([self.targetConfig], self.originTrainData)
        idx = np.argmin(dist)
        self.targetConfigEmbedding = self.lowDEmbedding[idx]

        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

    @abstractmethod
    def getNextStartState(self, episodeIdx):
        pass
    def findClosest(self, state):

        dist = euclidean_distances(state, self.lowDEmbedding)
        idx = np.argmin(dist)
        return self.originTrainData[idx]


class ManifoldUniformScheduler(ManifoldScheduler):
    def __init__(self, config, originTrainData, lowDEmbedding):
        super(ManifoldUniformScheduler, self).__init__(config, originTrainData, lowDEmbedding)
        self.methodName = 'ManifoldUniformScheduler'

    def getNextStartState(self, episodeIdx):
        targetThresh = self.thresh_by_episode(episodeIdx) * 200
        print('targetThresh', targetThresh)
        while True:
            state = np.random.randn(self.dim) * targetThresh + self.targetConfigEmbedding
            distanctVec = state - self.targetConfigEmbedding
            distance = np.linalg.norm(distanctVec, ord=2)
            if distance < targetThresh:
                break

        originalState = self.findClosest(state)

        return originalState

    def findClosest(self, state):

        dist = euclidean_distances([state], self.lowDEmbedding)
        idx = np.argmin(dist)
        return self.originTrainData[idx]

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

class ManifoldDirectedScheduler(ManifoldScheduler):
    def __init__(self, config, originTrainData, lowDEmbedding):
        super(ManifoldDirectedScheduler, self).__init__(config, originTrainData, lowDEmbedding)

    def getNextStartState(self, episodeIdx):
        targetThresh = self.thresh_by_episode(episodeIdx) * 200

        while True:
            state = np.random.randn(self.dim) * targetThresh + self.startConfig
            distanctVec = state - self.startConfig
            distance = np.linalg.norm(distanctVec, ord=np.inf)
            if distance < targetThresh:
                break

        originalState = self.findClosest(state)

        return originalState




class ManifoldNavigationEnv:
    def __init__(self, configName, randomSeed = 1):

        with open(configName) as f:
            self.config = json.load(f)
        self.randomSeed = randomSeed
        self.model = ManifoldNavigationModelPython(configName, randomSeed)
        self.read_config()
        self.initilize()
        self.nbActions = 3
        #self.padding = self.config['']

    def initilize(self):
        if not os.path.exists('Traj'):
            os.makedirs('Traj')
        # import parameter for vector env
        self.stepCount = 0

        self.info = {}

        random.seed(self.randomSeed)
        np.random.seed(self.randomSeed)

        self.epiCount = -1

    def read_config(self):

        self.stateDim = 3.0

        self.episodeEndStep = 500
        if 'episodeLength' in self.config:
            self.episodeEndStep = self.config['episodeLength']

        self.startThresh = 1
        self.endThresh = 1
        self.distanceThreshDecay = 10000

        self.targetThreshFlag = False

        if 'targetThreshFlag' in self.config:
            self.targetThreshFlag = self.config['targetThreshFlag']

        if 'target_start_thresh' in self.config:
            self.startThresh = self.config['target_start_thresh']
        if 'target_end_thresh' in self.config:
            self.endThresh = self.config['target_end_thresh']
        if 'distance_thresh_decay' in self.config:
            self.distanceThreshDecay = self.config['distance_thresh_decay']

        self.distanceScale = 20
        if 'distanceScale' in self.config:
            self.distanceScale = self.config['distanceScale']

        self.actionPenalty = 0.0
        if 'actionPenalty' in self.config:
            self.actionPenalty = self.config['actionPenalty']

        self.finishThresh = 1.0
        if 'finishThresh' in self.config:
            self.finishThresh = self.config['finishThresh']

        self.nStep = 100
        if 'modelNStep' in self.config:
            self.nStep = self.config['modelNStep']

        self.manifoldSchedule = None
        if 'schedulerMethod' in self.config:
            if self.config['schedulerMethod'] == 'ManifoldUniform':
                self.manifoldSchedule = ManifoldUniformScheduler(self.config, self.config['originTrainData'], self.config['lowDEmbedding'])

    def thresh_by_episode(self, step):
        return self.endThresh + (
                self.startThresh - self.endThresh) * math.exp(-1. * step / self.distanceThreshDecay)

    def getHindSightExperience(self, state, action, nextState, info):

        raise NotImplementedError

    def actionPenaltyCal(self, action):
        actionNorm = np.linalg.norm(action, ord=2)
        return -self.actionPenalty * actionNorm ** 2

    def step(self, action):
        reward = 0.0
        #if self.customExploreFlag and self.epiCount < self.customExploreEpisode:
        #    action = self.getCustomAction()
        self.model.step(self.nStep, action[0], action[1], action[2])
        self.currentState = self.model.getCurrentState()
        distance = self.targetState - self.currentState

        # update step count
        self.stepCount += 1

        done = False

        if self.is_terminal(distance):
            reward = 1.0
            done = True

        # penalty for actions
        reward += self.actionPenaltyCal(action)

        state = distance / self.distanceScale

        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()

        return state, reward, done, self.info.copy()

    def is_terminal(self, distance):
        return np.linalg.norm(distance, ord=np.inf) < self.finishThresh

    def reset_helper(self):

        self.targetState = self.config['targetState']


        if self.config['dynamicInitialStateFlag']:

            if self.manifoldSchedule is None:
                targetThresh = float('inf')
                if self.targetThreshFlag:
                    targetThresh = self.thresh_by_episode(self.epiCount) * 200
                    print('target Thresh', targetThresh)

                while True:
                    meshIdx = self.model.findClosestFace(self.targetState[0], self.targetState[1], self.targetState[2], targetThresh)
                    self.model.setInitialState(meshIdx, 0.25, 0.25)
                    meshCenter = self.model.getCurrentState()

                    distanctVec = meshCenter - self.targetState
                    distance = np.linalg.norm(distanctVec, ord=np.inf)
                    if distance < targetThresh and not self.is_terminal(distanctVec):
                        break
                # set initial state

                print('target distance', distance)
                self.currentState = meshCenter
            else:
                candidateState = self.manifoldSchedule.getNextStartState(self.epiCount)
                meshIdx = self.model.findClosestFace(candidateState[0], candidateState[1], candidateState[2], 3.0)
                self.model.setInitialState(meshIdx, 0.25, 0.25)
                meshCenter = self.model.getCurrentState()

                distanctVec = meshCenter - self.targetState
                distance = np.linalg.norm(distanctVec, ord=np.inf)
                # set initial state
                print('target distance', distance, self.manifoldSchedule.methodName)
                self.currentState = meshCenter
        else:
            self.currentState = self.config['currentState']
            meshIdx = self.model.findClosestFace(self.currentState[0], self.currentState[1], self.currentState[2], 10.0)
            self.model.setInitialState(meshIdx, 0.25, 0.25)
            meshCenter = self.model.getCurrentState()
            # set initial state
            self.currentState = meshCenter

        print("initial state ", self.currentState)

    def reset(self):
        self.stepCount = 0

        self.info = {}

        self.info['scaleFactor'] = self.distanceScale
        self.epiCount += 1

        self.currentState = np.array(self.config['currentState'], dtype=np.float32)
        self.targetState = np.array(self.config['targetState'], dtype=np.int32)

        self.model.reset()
        self.reset_helper()

        self.info['currentState'] = self.currentState.copy()
        self.info['targetState'] = self.targetState.copy()

        distance = self.targetState - self.currentState
        return distance / self.distanceScale
