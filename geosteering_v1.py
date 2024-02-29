from state_last import SubsurfaceState, eval_along_y
import numpy as np
import pandas as pd
import scipy.stats as stats
import copy
import math
import time

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Geosteering:
    def __init__(self, inputs="PF", bestrl=1, bestpf=1, dec=32, thickness=20, nParticles=128, resample=1):
        self.nSteps = 320
        self.DecisionFreq = dec
        self.nDecisions = int(self.nSteps / (self.DecisionFreq))
        self.nDiscretizations = self.nSteps + self.DecisionFreq

        self.nPlots = self.nDecisions + 1

        self.inputs = inputs
        if self.inputs == "PF":
            self.nParticles = nParticles
            self.nParameters = 2
            self.bestrl = bestrl
            self.bestpf = bestpf

            self.posRange = [-1, 1]
            self.angleRange = [np.pi / 2, np.pi / 2]
            self.parRanges = [self.posRange, self.angleRange]

            self.resampleType = resample
            if self.resampleType == 1:
                self.resampleFreq = np.copy(self.DecisionFreq)
            else:
                self.resampleFreq = int(self.DecisionFreq / 2)

        self.h = thickness * 2
        self.scale = 2
        self.calc_adj = self.h / self.scale
        self.dogleg = 3
        self.dx = 10
        self.bd = 0
        self.faults_num = 0

        self.refData = pd.read_csv("gr.csv")["GR"].to_numpy()[10350:11850]
        self.normalized = (self.refData - np.min(self.refData)) / \
            (np.max(self.refData) - np.min(self.refData))

    def GenerateState(self, pos, angle):
        state = SubsurfaceState(
            pos, angle, decfreq=self.DecisionFreq, discrete=len(self.boundary) + 1 
        )
        return state

    def GenerateGammaObs(self, state):
        if (self.UB_idx + self.h - state * self.scale >= len(self.normalized) - 100).any():
            self.error = True
        gammaObs = eval_along_y(
            self.UB_idx + self.h - state * self.scale, self.normalized, noise=False
        )
        return gammaObs

    def TrueBoundaryFunc(self):
        trueStatePars = [0, np.pi / 2]
        trueState = self.GenerateState(*trueStatePars)
        for iter in range(len(self.boundary)):
            trueState.get_next_state_random_true()
        self.faults_num = trueState.fault_num
        self.fault_positive = trueState.fault_positive
        self.boundary = trueState.state_array[: len(self.boundary), 0]

    def GenerateInitialParticles_Uniform(self):
        initParam = np.zeros((self.nParameters))
        initStates = []
        for i in range(0, self.nParticles):
            for j in range(0, self.nParameters):
                initParam[j] = np.random.uniform(
                    self.parRanges[j][0], self.parRanges[j][1])

            initStates.append(self.GenerateState(initParam[0], initParam[1]))

        return [initStates]

    def getGammaRayRMSE(self, predictedState, actualState):
        yActualState = np.array(actualState)
        yPredictedState = np.array(predictedState)
        return np.sqrt(np.mean(np.square(np.subtract(yActualState, yPredictedState))))

    def ParticleFilter(self, states, prevFrequencyVec, observation, bitdepth, likelihoodStd=0.5):
        nonZeroFrequencyIndex = np.nonzero(prevFrequencyVec)[0]

        state_next = bitdepth - \
            np.array([states[i].get_next_state_random()
                     for i in nonZeroFrequencyIndex])
        obs_vec = self.GenerateGammaObs(state_next)

        rmseVec = np.sqrt(
            np.mean(np.square(np.subtract(obs_vec, observation)), axis=1))
        rmseVec /= np.max(rmseVec)
        likelihoodVec = stats.norm(0, likelihoodStd).pdf(
            rmseVec) * prevFrequencyVec
        return likelihoodVec

    def StochasticModelResampling(self, states, priorVec, likelihood):
        posteriorVec = priorVec * likelihood
        posteriorVec /= np.sum(posteriorVec)
        if not np.isclose(np.sum(posteriorVec), 1.0):
            print("error 1 in resampling")

        cumProb = np.cumsum(posteriorVec)
        tempRandomU = np.random.uniform(size=self.nParticles)
        resampledStates = [copy.deepcopy(
            states[np.argmax(tempRandomU[j] <= cumProb)]) for j in range(self.nParticles)]
        newPosteriorVec = np.ones(posteriorVec.size) / self.nParticles
        return [resampledStates, newPosteriorVec]

    def InitPF(self):
        self.particleStates = self.GenerateInitialParticles_Uniform()

        self.particlesFrequencyMat[0, ...] = np.ones((1, self.nParticles))
        self.likelihoodMat[0, ...] = np.ones(
            (1, self.nParticles)) / self.nParticles
        self.posteriorMat[0, ...] = self.likelihoodMat[0, ...]

    def MainPF(self, idxs):
        for i in idxs:
            self.likelihoodMat[i, ...] = self.ParticleFilter(
                self.particleStates[-1], self.particlesFrequencyMat[i - 1, ...], self.gamma[i], self.bitdepth[i], likelihoodStd=0.5,)

            if i % self.resampleFreq != 0:
                self.posteriorMat[i, ...] = (
                    self.posteriorMat[i - 1, ...] * self.likelihoodMat[i, ...])
                self.posteriorMat[i, ...] = self.posteriorMat[i, ...] / \
                    np.sum(self.posteriorMat[i, ...])
                self.particlesFrequencyMat[i, ...] = self.particlesFrequencyMat[i - 1, ...]
            else:
                [newParticleStates, self.posteriorMat[i, ...],] = self.StochasticModelResampling(
                    self.particleStates[-1], self.posteriorMat[i - 1, ...], self.likelihoodMat[i, ...],)
                self.particleStates.append(newParticleStates)
                self.particlesFrequencyMat[i, ...] = self.particlesFrequencyMat[i - 1, ...]

    def reset(self):
        self.bitdepth = np.zeros(self.nDiscretizations + 1)
        self.boundary = np.zeros(self.nDiscretizations + 1)
        self.SVD = np.zeros(self.nDiscretizations + 1)
        self.gamma = np.zeros(self.nDiscretizations + 1)

        if self.inputs == "PF":
            self.particlesFrequencyMat = np.zeros(
                (self.nDiscretizations + 1, self.nParticles))
            self.likelihoodMat = np.zeros(
                (self.nDiscretizations + 1, self.nParticles))
            self.posteriorMat = np.zeros(
                (self.nDiscretizations + 1, self.nParticles))
            self.particleStates = []

        self.Inc = np.zeros(self.nDiscretizations + 1)
        self.bitdepth[0] = self.calc_adj + np.random.randint(200, 250)
        self.Inc[0] = 110
        self.bd = 0
        self.contact = 0
        self.error = False
        self.pf_time = 0

        self.training_num = 0
        distances = [900, 1350, 450]
        self.UB_idx = distances[self.training_num]
        self.mean_log = np.mean(self.normalized[self.UB_idx:self.UB_idx+self.h])

        self.TrueBoundaryFunc()
        self.SVD[0] = self.bitdepth[0] - self.boundary[0]
        self.gamma[0] = self.GenerateGammaObs(self.SVD[0])

        if self.inputs == "PF":
            self.best_state = np.zeros(self.nDiscretizations + 1)
            self.mean_state = np.zeros(self.nDiscretizations + 1)
            return self.step(5)
        elif self.inputs == "Sensor":
            return self.step_sensor(5)
        elif self.inputs == "Gamma":
            return self.step_nonPF(5)

    def min_curvature(self, action):
        if self.bd < self.nSteps // 3:
            angle = [70, 110]
        else:
            angle = [86, 94]

        Inc_f = (self.Inc[self.bd] + (action - 5) / 5 *
                 (self.DecisionFreq * self.dx) * self.dogleg / 100)
        Inc_step = (Inc_f - self.Inc[self.bd]) / self.DecisionFreq

        reward = 0
        for jdx in range(self.DecisionFreq):
            if self.bd == self.nDiscretizations:
                break
            self.Inc[self.bd +
                     1] = np.clip(self.Inc[self.bd] + Inc_step, angle[0], angle[1])
            beta = Inc_step * math.pi / 180
            if beta == 0:
                deltavert = (
                    self.dx / 2 * (2 * math.cos(self.Inc[self.bd] * math.pi / 180)))
            else:
                RF = 2 / beta * math.tan(beta / 2)
                deltavert = (self.dx / 2 * (math.cos(self.Inc[self.bd + 1] * math.pi / 180) + math.cos(
                    self.Inc[self.bd] * math.pi / 180)) * RF)
            self.bitdepth[self.bd + 1] = self.bitdepth[self.bd] + deltavert
            self.bd += 1

            norm_DTLB = (self.bitdepth[self.bd] -
                         self.boundary[self.bd]) / self.calc_adj
            norm_DTUB = (
                self.boundary[self.bd] + self.calc_adj - self.bitdepth[self.bd]) / self.calc_adj
            alpha = min(norm_DTUB, norm_DTLB)

            if 0 <= alpha <= 0.5:
                reward += 0
            else:
                reward += -1

        return reward, Inc_step, deltavert

    def step(self, action):
        reward, Inc_step, deltavert = self.min_curvature(action)

        if self.bd == self.DecisionFreq:
            self.InitPF()

            self.Inc[self.bd + 1] = self.Inc[self.bd] + Inc_step
            self.bitdepth[self.bd + 1] = self.bitdepth[self.bd] + deltavert

            self.bd += 1
            x = np.arange(1, self.bd)
        elif self.bd == self.nDiscretizations:
            self.bd += 1
            x = np.arange(self.bd - self.DecisionFreq, self.bd)
        else:
            x = np.arange(self.bd - self.DecisionFreq, self.bd)

        self.SVD[x] = self.bitdepth[x] - self.boundary[x]
        self.gamma[x] = np.apply_along_axis(
            self.GenerateGammaObs, 0, self.SVD[x])

        start = time.time()
        self.MainPF(x)
        self.pf_time += time.time()-start

        indices = np.argsort(
            self.posteriorMat[self.bd - 2, :])[::-1]
        sample = self.bd // self.resampleFreq - 1
        sum_prob_best = np.sum(self.posteriorMat[self.bd - 2, indices[ :self.bestpf]])

        state = []
        naive = []
        info = []
        if self.bd == self.nDiscretizations + 1 or self.error == True:
            done = True
            for jdx in range(self.nParticles):
                self.particleStates[-1][jdx].get_next_state_random()

            for jdx in range(int(self.nSteps * 3 / 10) + 1, self.nSteps + 1):
                if (self.boundary[jdx] < self.bitdepth[jdx] < self.boundary[jdx] + self.calc_adj):
                    self.contact += 1
            self.contact /= self.nSteps * (1 - 3 / 10)

            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd - 1] - 90) / 4)
            best_num = 0
            #==================================================================#
            self.mean_state[self.bd - 1 - self.DecisionFreq] = 0
            self.best_state[self.bd - 1 - self.DecisionFreq] = 0
            for i in range(self.nParticles):
                for idx in range(self.DecisionFreq + 1):
                    ax = self.bd - 1 - self.DecisionFreq + idx
                    self.mean_state[ax] += self.posteriorMat[self.bd - 2, i] \
                        * self.particleStates[sample][i].state_array[ax, 0]
                    if i in indices[: self.bestpf]:
                        self.best_state[ax] += self.posteriorMat[self.bd - 2, i]/sum_prob_best \
                        * self.particleStates[sample][i].state_array[ax, 0]
                best_num += 1
            #==================================================================#
            for jdx in indices[: self.bestrl]:
                state_array = self.particleStates[sample][jdx].state_array
                state.append(self.posteriorMat[self.bd - 2, jdx])
                for idx in range(self.DecisionFreq + 1):
                    ax = self.bd - 1 - self.DecisionFreq + idx
                    state.append(
                        (state_array[ax, 0] + self.calc_adj - self.bitdepth[ax]) / self.calc_adj)
                    state.append(
                        (self.bitdepth[ax] - state_array[ax, 0]) / self.calc_adj)
            #==================================================================#
        else:
            done = False
            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd] - 90) / 4)
            best_num = 0
            #==================================================================#
            info.append(self.posteriorMat[self.bd - 2, :])
            
            self.mean_state[self.bd - 1 - self.DecisionFreq] = 0
            self.best_state[self.bd - 1 - self.DecisionFreq] = 0
            for i in range(self.nParticles):
                info.append(self.particleStates[sample][i].state_array[self.bd - 2, 0] 
                            - self.boundary[self.bd - 2]) 
                for idx in range(self.DecisionFreq + 1):
                    ax = self.bd - 1 - self.DecisionFreq + idx
                    self.mean_state[ax] += self.posteriorMat[self.bd - 2, i] \
                        * self.particleStates[sample][i].state_array[ax, 0]
                    if i in indices[: self.bestpf]:
                        self.best_state[ax] += self.posteriorMat[self.bd - 2, i]/sum_prob_best \
                        * self.particleStates[sample][i].state_array[ax, 0]
                best_num += 1
            #==================================================================#
            for jdx in indices[: self.bestrl]:
                state_array = self.particleStates[sample][jdx].state_array
                state.append(self.posteriorMat[self.bd - 2, jdx])
                for idx in range(self.DecisionFreq + 1):
                    ax = self.bd - 1 - self.DecisionFreq + idx
                    state.append(
                        (state_array[ax, 0] + self.calc_adj - self.bitdepth[ax]) / self.calc_adj)
                    state.append(
                        (self.bitdepth[ax] - state_array[ax, 0]) / self.calc_adj)
            #==================================================================#
                naive_state = copy.deepcopy(self.particleStates[sample][jdx])
                naive.append([naive_state, self.posteriorMat[self.bd - 2, jdx]])
            naive.append(self.boundary)
            naive.append(self.bitdepth[ax])
            naive.append(self.Inc[ax])
            #==================================================================#

        return np.array(state, dtype=np.float32), reward, done, info

    def step_nonPF(self, action):
        reward, Inc_step, deltavert = self.min_curvature(action)

        if self.bd == self.DecisionFreq:
            self.Inc[self.bd + 1] = self.Inc[self.bd] + Inc_step
            self.bitdepth[self.bd + 1] = self.bitdepth[self.bd] + deltavert

            self.bd += 1
            x = np.arange(1, self.bd)
        elif self.bd == self.nDiscretizations:
            self.bd += 1
            x = np.arange(self.bd - self.DecisionFreq, self.bd)
        else:
            x = np.arange(self.bd - self.DecisionFreq, self.bd)

        self.SVD[x] = self.bitdepth[x] - self.boundary[x]
        self.gamma[x] = np.apply_along_axis(
            self.GenerateGammaObs, 0, self.SVD[x])

        state = []
        if self.bd == self.nDiscretizations + 1 or self.error == True:
            done = True
            for jdx in range(int(self.nSteps * 3 / 10) + 1, self.nSteps + 1):
                if (self.boundary[jdx] < self.bitdepth[jdx] < self.boundary[jdx] + self.calc_adj):
                    self.contact += 1
            self.contact /= self.nSteps * (1 - 3 / 10)

            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd - 1] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 - self.DecisionFreq + idx
                state.append(self.gamma[ax])
        else:
            done = False
            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 - self.DecisionFreq + idx
                state.append(self.gamma[ax])

        return np.array(state, dtype=np.float32), reward, done, {}

    def step_sensor(self, action):
        reward, Inc_step, deltavert = self.min_curvature(action)

        if self.bd == self.DecisionFreq:
            self.Inc[self.bd + 1] = self.Inc[self.bd] + Inc_step
            self.bitdepth[self.bd + 1] = self.bitdepth[self.bd] + deltavert

            self.bd += 1
        elif self.bd == self.nDiscretizations:
            self.bd += 1

        state = []
        if self.bd == self.nDiscretizations + 1:
            done = True
            for jdx in range(int(self.nSteps * 3 / 10) + 1, self.nSteps + 1):
                if (self.boundary[jdx] < self.bitdepth[jdx] < self.boundary[jdx] + self.calc_adj):
                    self.contact += 1
            self.contact /= self.nSteps - self.nSteps * 3 / 10

            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd - 1] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 - self.DecisionFreq + idx
                state.append(
                    (self.boundary[ax] + self.calc_adj - self.bitdepth[ax]) / self.calc_adj)
                state.append(
                    (self.bitdepth[ax] - self.boundary[ax]) / self.calc_adj)

        else:
            done = False
            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 - self.DecisionFreq + idx
                state.append(
                    (self.boundary[ax] + self.calc_adj - self.bitdepth[ax]) / self.calc_adj)
                state.append(
                    (self.bitdepth[ax] - self.boundary[ax]) / self.calc_adj)

        return np.array(state, dtype=np.float32), reward, done, {}
    
    def step_sensorahead(self, action):
        reward, Inc_step, deltavert = self.min_curvature(action)

        if self.bd == self.DecisionFreq:
            self.Inc[self.bd + 1] = self.Inc[self.bd] + Inc_step
            self.bitdepth[self.bd + 1] = self.bitdepth[self.bd] + deltavert

            self.bd += 1
        elif self.bd == self.nDiscretizations:
            self.bd += 1

        state = []
        if self.bd == self.nDiscretizations + 1:
            done = True
            for jdx in range(int(self.nSteps * 3 / 10) + 1, self.nSteps + 1):
                if (self.boundary[jdx] < self.bitdepth[jdx] < self.boundary[jdx] + self.calc_adj):
                    self.contact += 1
            self.contact /= self.nSteps - self.nSteps * 3 / 10

            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd - 1] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 + idx
                state.append(
                    (self.boundary[ax] + self.calc_adj - self.bitdepth[self.bd - 1]) / self.calc_adj)
                state.append(
                    (self.bitdepth[self.bd - 1] - self.boundary[ax]) / self.calc_adj)

        else:
            done = False
            state.append((self.bd - self.DecisionFreq - 1 -
                         self.nSteps / 2) / (self.nSteps / 2))
            state.append((self.Inc[self.bd] - 90) / 4)
            for idx in range(self.DecisionFreq + 1):
                ax = self.bd - 1 + idx
                state.append(
                    (self.boundary[ax] + self.calc_adj - self.bitdepth[self.bd - 1]) / self.calc_adj)
                state.append(
                    (self.bitdepth[self.bd - 1] - self.boundary[ax]) / self.calc_adj)

        return np.array(state, dtype=np.float32), reward, done, {}
    
    def LogPlot(self):
        fig, ax2 = plt.subplots(figsize=(5, 10))
        y1 = np.arange(len(self.normalized))
        y2 = np.arange(self.UB_idx, self.UB_idx + self.h)
        x = self.normalized
        ax2.plot(x[y1], y1 / self.scale, linewidth=0.7)
        # ax2.plot(x[y2], y2 / self.scale, linewidth=1.0, color="red")
        ax2.set_title("Gamma-ray log")
        ax2.set_ylabel("TVD (shifted, ft)")
        ax2.set_xlabel("Scaled to 0..1")
        ax2.set_ylim(0, len(self.normalized) // self.scale)
        ax2.set_xlim(0, 1)
        plt.gca().invert_yaxis()
        plt.show()

    def InitPlot(self):
        fig, self.ax = plt.subplots(nrows=2, ncols=3, figsize=(13, 5))

        self.ax[0, 0].set_position([0.1, 0.35, 0.3, 0.55])
        self.ax[0, 1].set_position([0.48, 0.35, 0.3, 0.55])
        self.ax[0, 2].set_position([0.8, 0.35, 0.1, 0.55])
        self.ax[1, 0].set_position([0.1, 0.17, 0.3, 0.1])
        self.ax[1, 1].set_position([0.48, 0.17, 0.3, 0.1])

        for r in range(2):
            for c in range(3):
                if c == 0:
                    self.ax[r, c].set_xlim(0, self.nDiscretizations * self.dx)
                elif c == 2:
                    self.ax[r, c].set_xlim(0, 1.0)

        x_data = np.arange(0, self.nDiscretizations + 1, 1)
        y_data = np.arange(
            min(self.boundary) - 50, max(self.boundary) + self.calc_adj + 100
        )
        self.ax[0, 0].plot(
            x_data * self.dx,
            self.boundary[x_data] + self.calc_adj,
            "--",
            linewidth=2,
            color="black",
            label="Boundary",
        )
        self.ax[0, 0].plot(
            x_data * self.dx, self.boundary[x_data], "--", linewidth=2, color="black"
        )
        self.ax[0, 2].plot(
            np.apply_along_axis(self.GenerateGammaObs, 0, y_data), y_data
        )

        for ax in self.ax[0]:
            ax.set_ylim(
                min(self.boundary) - 50, max(self.boundary) + self.calc_adj + 100
            )
        for ax in self.ax[1]:
            ax.set_ylim(0, 1.0)
            ax.set_xlabel("Horizontal Distance (ft)")

        self.ax[0, 0].set_title("Full Plot")
        self.ax[0, 1].set_title("Focused Plot")
        self.ax[0, 2].set_title("Offset Log")

        self.ax[0, 0].set_ylabel("TVD (relative, ft)")
        self.ax[0, 1].set_ylabel("SVD (relative, ft)")
        self.ax[1, 0].set_ylabel("Gamma Ray")

        self.ax[1, 2].axis("off")
        self.ax[0, 2].tick_params(labelleft=False, labelright=False)

        self.lines = []
        for r in range(2):
            for c in range(3):
                self.lines.append(self.ax[r, c])
        return self.lines

    def UpdatePlot(self, frame):
        x_input = frame // 3
        x_data = np.arange(
            x_input * self.DecisionFreq, (x_input + 1) *
            self.DecisionFreq + 1, 1
        )

        for ax in self.ax[:, 1]:
            ax.set_xlim(
                (
                    x_input * self.DecisionFreq * self.dx,
                    (x_input + 1) * self.DecisionFreq * self.dx,
                )
            )
        self.lines[1].plot(
            x_data * self.dx, np.zeros(len(x_data)), "--", linewidth=2, color="black"
        )
        self.lines[1].plot(
            x_data * self.dx,
            np.zeros(len(x_data)) + self.calc_adj,
            "--",
            linewidth=2,
            color="black",
        )

        if frame % 3 == 0:
            self.lines[0].plot(x_data * self.dx, self.bitdepth[x_data],
                               linewidth=2, color="black", label="Trajectory",)
            self.lines[0].scatter(x_data[:: self.DecisionFreq] * self.dx,
                                  self.bitdepth[x_data][:: self.DecisionFreq], color="black",)
            self.lines[1].plot(x_data * self.dx, self.SVD[x_data],
                               linewidth=2, color="black", label="True",)
            self.lines[3].plot(x_data * self.dx, np.array(self.gamma)
                               [x_data], "--", linewidth=2, color="black", label="True",)
            self.lines[4].plot(x_data * self.dx, np.array(self.gamma)
                               [x_data], "--", linewidth=2, color="black", label="True",)

        if self.inputs == "PF":
            svdParticles_array = np.zeros([self.DecisionFreq + 1, self.nParticles])
            prob_array = np.zeros([self.nParticles, 1])

            for jdx in range(self.nParticles):
                if self.resampleType == 1:
                    sample_x = x_input
                else:
                    sample_x = x_input * 2 + 1
                state_array = self.particleStates[sample_x][jdx].state_array
                svdParticles = self.bitdepth[x_data] - state_array[x_data, 0]
                svdParticles_array[:, jdx] = svdParticles
                prob_array[jdx, 0] = self.posteriorMat[x_input *
                                                       self.DecisionFreq + self.DecisionFreq - 1, jdx]
                gamma = self.GenerateGammaObs(svdParticles)

                if frame % 3 == 1:
                    if jdx % 4 == 0:
                        self.lines[0].plot(
                            x_data * self.dx, state_array[x_data, 0], "-", linewidth=0.05, color="grey",)
                        self.lines[1].plot(
                            x_data * self.dx, svdParticles, "-", linewidth=0.05, color="grey",)
                        self.lines[3].plot(
                            x_data * self.dx, gamma, "-", linewidth=0.05, color="grey")
                        self.lines[4].plot(
                            x_data * self.dx, gamma, "-", linewidth=0.05, color="grey")

            indices = np.argsort(
                self.posteriorMat[x_input * self.DecisionFreq + self.DecisionFreq - 1, :])[::-1][: self.bestrl]

            col_idx = 0
            colors = ["blue", "green", "yellow"]

            for jdx in indices:
                state_array = self.particleStates[sample_x][jdx].state_array
                svdParticles = self.bitdepth[x_data] - state_array[x_data, 0]
                gamma = self.GenerateGammaObs(svdParticles)

                if frame % 3 == 2:
                    self.lines[0].plot(
                        x_data * self.dx, state_array[x_data, 0], linewidth=0.7, color=colors[col_idx],)
                    self.lines[1].plot(
                        x_data * self.dx, svdParticles, linewidth=0.7, color=colors[col_idx],)
                    self.lines[3].plot(x_data * self.dx, gamma,
                                       linewidth=0.7, color=colors[col_idx])
                    self.lines[4].plot(x_data * self.dx, gamma,
                                       linewidth=0.7, color=colors[col_idx])

                col_idx += 1
                if col_idx == 3:
                    col_idx = 0

        return self.lines

    def Animate(self):
        anim = FuncAnimation(fig=self.InitPlot()[1].get_figure(
        ), func=self.UpdatePlot, frames=self.nPlots * 3, interval=1000, repeat=False,)
        # anim.save('fig.mp4')
        plt.show()

    def FullPlot(self):
        self.InitPlot()
        for idx in range(self.nPlots * 3):
            self.UpdatePlot(idx)
        plt.show()

    def FullPlot_noPF(self, start_idx, end_idx):
        numPlots = self.nPlots // (end_idx - start_idx)
        for plot_idx in range(numPlots):
            fig, ax = plt.subplots(3, 1, figsize=(10, 9))

            x_data = np.arange((start_idx + end_idx * plot_idx) * self.DecisionFreq,
                               (end_idx + end_idx * plot_idx) * self.DecisionFreq + 1, 1,)
            ax[0].plot(x_data * self.dx, np.array(self.gamma)[x_data],
                       "--", linewidth=2, color="black", label="True",)
            ax[1].plot(x_data * self.dx, self.SVD[x_data], "--",
                       linewidth=2, color="black", label="True",)
            ax[2].plot(x_data * self.dx, self.boundary[x_data] +
                       self.calc_adj, "--", linewidth=2, color="black",)
            ax[2].plot(x_data * self.dx, self.boundary[x_data], "--",
                       linewidth=2, color="black", label="Boundary",)
            ax[2].plot(x_data * self.dx, self.bitdepth[x_data], linewidth=2,
                       color="black", label="Trajectory, contact=%s" % self.contact,)
            ax[2].scatter(x_data[:: self.DecisionFreq],
                          self.bitdepth[x_data][:: self.DecisionFreq], color="black",)

            for axis in ax:
                axis.legend()
                axis.set_xlim(min(x_data), max(x_data))

            ax[1].set_ylabel("SVD (ft)")
            ax[0].set_ylabel("Gamma Ray")
            ax[2].set_ylabel("TVD (relative, ft)")
            fig.supxlabel("Horizontal Distance (ft)")
            plt.show()
