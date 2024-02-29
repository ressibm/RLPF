import numpy as np
import torch as T
from agent import DQNAgent
import random
import time
import copy

from geosteering_v1 import Geosteering
from util.plot import mean_plot, combined_plot

def eval_func(n_games, inputs, best, dec, thickness, seed_eval, bd_agent=0, load=True, seed_load=0, plot=True,):
    
    seed_eval = seed_eval
    env_eval = Geosteering(inputs=inputs, bestrl=best,
                           dec=dec, thickness=thickness, resample=2)

    if inputs == "PF":
        states = ((env_eval.DecisionFreq + 1) * env_eval.bestrl * 2 + env_eval.bestrl + 2,)
        multi = 2
    elif inputs == "Sensor":
        states = ((env_eval.DecisionFreq + 1) * 2 + 2,)
        multi = 4
    elif inputs == "Gamma":
        states = ((env_eval.DecisionFreq + 1) + 2,)
        multi = 4

    bd_actions = 11

    if load:
        bd_agent_eval = DQNAgent(n_actions=bd_actions,
                                 n_states=states, seed=seed_eval,
                                 multi=multi)

        string = "trained_network/v2/%s_best=%s_%s_%s.pth"

        bd_agent_eval.Q_eval.load_state_dict(
            T.load(string % (inputs, best, seed_load, dec)))

    else:
        bd_agent_eval = copy.deepcopy(bd_agent)

    bd_agent_eval.epsilon = 0

    scores = np.full(n_games, np.nan)
    scores_1 = np.full(n_games, np.nan)
    scores_2 = np.full(n_games, np.nan)
    contact = np.full(n_games, np.nan)
    if plot:
        UB = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)
        LB = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)
        bitdepth = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)

    if inputs == "PF":
        beststate = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)

    rmsd_gamma = 0
    rmsd_state = 0
    for i in range(n_games):
        np.random.seed(seed_eval)
        random.seed(seed_eval)

        done = False
        start = time.time()
        observation, reward, done, info = env_eval.reset()

        score = 0
        while not done:
            action = bd_agent_eval.choose_action(observation)
            observation_, reward, done, info = env_eval.step(action)
            if env_eval.nSteps * 3.5 / 10 < env_eval.bd <= env_eval.nSteps + 1:
                score += reward
            if env_eval.error == True:
                score = -224
            observation = observation_
        print('Total = ', (time.time() - start))
        # print('PF = ', env_eval.pf_time)

        if env_eval.faults_num >= 0:# and env_eval.fault_positive == 1:
            scores[i] = score
            # if env_eval.faults_num >= 1:
            #     scores_1[i] = score
            #     if env_eval.faults_num >= 2:
            #         scores_2[i] = score
            contact[i] = env_eval.contact
            env_eval.FullPlot()

            if plot:
                UB[i,] = env_eval.boundary[: env_eval.nDiscretizations + 1] + \
                    env_eval.calc_adj
                LB[i,] = env_eval.boundary[: env_eval.nDiscretizations + 1]
                bitdepth[i,] = env_eval.bitdepth[: env_eval.nDiscretizations + 1]

            if inputs == "PF":
                beststate[i,] = env_eval.best_state

                rmsd_gamma += np.mean(np.abs(np.apply_along_axis(env_eval.GenerateGammaObs,
                                    0, env_eval.bitdepth - env_eval.best_state[:],) - env_eval.gamma))
                rmsd_state += np.mean(
                    np.abs(env_eval.best_state[:] - env_eval.boundary))

        seed_eval += 1

    idx_not_nan = np.where(~np.isnan(scores))[0]
    idx_not_nan_1 = np.where(~np.isnan(scores_1))[0]
    idx_not_nan_2 = np.where(~np.isnan(scores_2))[0]

    if inputs == "PF":
        print(rmsd_gamma / len(idx_not_nan))
        print(rmsd_state / len(idx_not_nan))

    if plot:
        mean_plot(UB=UB[idx_not_nan], LB=LB[idx_not_nan], bitdepth=bitdepth[idx_not_nan], 
                  scores=np.mean(scores[idx_not_nan]), contact=np.mean(contact[idx_not_nan]), 
                  n_reads=env_eval.nDiscretizations, n_games=len(idx_not_nan))
    # return np.mean(scores[idx_not_nan]), np.mean(scores_1[idx_not_nan_1]), np.mean(scores_2[idx_not_nan_2])
    return np.mean(scores[idx_not_nan])#, UB, LB, bitdepth


if __name__ == "__main__":
    UB = []
    LB = []
    bitdepth = []
    for best in [5]:
        for seed_load in [624]:
            for dec in [32]:
                score = eval_func(n_games=1000, inputs="PF", best=best,
                                dec=dec, thickness=20, seed_eval=13063, seed_load=seed_load,
                                plot=False)
                print(score)
                # UB.append(x)
                # LB.append(y)
                # bitdepth.append(z)
    # combined_plot(UB, LB, bitdepth)

#9,  140, 272, 387, 400, 571, 624, 733, 898, 956, 1000