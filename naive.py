import numpy as np
import torch as T
from agent import DQNAgent
from tqdm import tqdm
import random
import math
import copy

from geosteering_v1 import Geosteering
from util.plot import mean_plot

def naive_func(naive, real_bd, thickness = 20, errors=0):
    last_bd = naive[-2]
    last_inc = naive[-1]

    best_reward = -math.inf
    best_action = 5

    errors = np.zeros(len(naive)-3)


    for action in range(11):

        reward_act = 0

        for idx in range(len(naive) - 3):
            if action == 0:
                for jdx in range(32):
                    naive[idx][0].get_next_state_random()   
            boundary = naive[idx][0].state_array[:, 0]

            angle = [86, 94]

            Inc_f = last_inc + (action - 5) / 5 * 32 * 10 * 3 / 100
            Inc_step = (Inc_f - last_inc) / 32
            bd = np.zeros(32)

            reward_par = 0
            Inc_ = last_inc
            bd_ = last_bd
            
            for jdx in range(32):
                Inc = np.clip(Inc_ + Inc_step, angle[0], angle[1])
                beta = Inc_step * math.pi / 180
                if beta == 0:
                    deltavert = 3 / 2 * (2 * math.cos(last_inc * math.pi / 180))
                else:
                    RF = 2 / beta * math.tan(beta / 2)
                    deltavert = (3 / 2 * (math.cos(Inc * math.pi / 180) +
                                math.cos(last_inc * math.pi / 180)) * RF)
                bd[jdx] = bd_ + deltavert
                Inc_ = Inc
                bd_ = bd[jdx]

                norm_DTLB = (bd[jdx] - boundary[real_bd+jdx]) / thickness
                norm_DTUB = (boundary[real_bd+jdx] + thickness - bd[jdx]) / thickness
                alpha = min(norm_DTUB, norm_DTLB)

                reward_par += -4*(alpha - 0.5)**2 + 1

                if action == 0:
                    errors[idx] += abs(naive[-3][real_bd+jdx]-boundary[real_bd+jdx])

            reward_act += reward_par * 1/(len(naive)-3)

        if reward_act > best_reward:
            best_reward = reward_act
            best_action = action

    return best_action, errors

def eval_func(n_games, inputs, best, dec, thickness, seed_eval, seed_load=0, plot=True,):
    
    seed_eval = seed_eval
    env_eval = Geosteering(inputs=inputs, best=best,
                           dec=dec, thickness=thickness, resample=2)

    states = ((env_eval.DecisionFreq + 1) * env_eval.best * 2 + env_eval.best + 2,)

    bd_actions = 11

    bd_agent_eval = DQNAgent(n_actions=bd_actions,
                                n_states=states, seed=seed_eval)

    string = "trained_network/v2/%s_best=%s_%s_%s_small.pth"

    bd_agent_eval.Q_eval.load_state_dict(
        T.load(string % (inputs, best, seed_load, dec)))

    bd_agent_eval.epsilon = 0

    scores = np.full(n_games, np.nan)
    contact = np.full(n_games, np.nan)
    if plot:
        UB = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)
        LB = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)
        bitdepth = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)

    beststate = np.full((n_games, env_eval.nDiscretizations + 1), np.nan)

    rmsd_gamma = 0
    rmsd_state = 0
    errors_list = []

    for i in range(n_games):
        np.random.seed(seed_eval)
        random.seed(seed_eval)

        done = False
        observation, reward, done, naive = env_eval.reset()

        score = 0
        errors = np.zeros(len(naive)-3)
        idx = 0

        while not done:
            if env_eval.bd // env_eval.DecisionFreq >= 3:
                action, errors_ = naive_func(naive=naive, real_bd=env_eval.bd, errors=errors)
                errors += errors_
                idx += env_eval.DecisionFreq
            else:
                action = bd_agent_eval.choose_action(observation)
            observation_, reward, done, naive = env_eval.step(action)
            if env_eval.nSteps * 3.5 / 10 < env_eval.bd <= env_eval.nSteps + 1:
                score += reward
            if env_eval.error == True:
                score = -224
            observation = observation_
        errors_list.append(errors/idx)
        env_eval.LogPlot()

        if env_eval.faults_num >= 0:
            scores[i] = score
            contact[i] = env_eval.contact

            if plot:
                UB[i,] = env_eval.boundary[: env_eval.nDiscretizations + 1] + \
                    env_eval.calc_adj
                LB[i,] = env_eval.boundary[: env_eval.nDiscretizations + 1]
                bitdepth[i,] = env_eval.bitdepth[: env_eval.nDiscretizations + 1]

            if inputs == "PF":
                beststate[i,] = env_eval.best_state[0,
                                                    : env_eval.nDiscretizations + 1]

                rmsd_gamma += np.mean(np.abs(np.apply_along_axis(env_eval.GenerateGammaObs,
                                    0, env_eval.bitdepth[97:] - env_eval.best_state[0, 97:],) - env_eval.gamma[97:]))
                rmsd_state += np.mean(
                    np.abs(env_eval.best_state[0, 97:] - env_eval.boundary[97:]))

        seed_eval += 1
    
    print(np.mean(errors_list, axis=0))

    idx_not_nan = np.where(~np.isnan(scores))[0]

    if inputs == "PF":
        print(rmsd_gamma / len(idx_not_nan))
        print(rmsd_state / len(idx_not_nan))

    if plot:
        mean_plot(UB=UB[idx_not_nan], LB=LB[idx_not_nan], bitdepth=bitdepth[idx_not_nan], 
                  scores=np.mean(scores[idx_not_nan]), contact=np.mean(contact[idx_not_nan]), 
                  n_reads=env_eval.nDiscretizations, n_games=len(idx_not_nan))
    return np.mean(scores[idx_not_nan])


if __name__ == "__main__":
    for best in [1]:
        for seed_load in [624]:
            for dec in [32]:
                score = eval_func(n_games=1000, inputs="PF", best=best,
                                dec=dec, thickness=20, seed_eval=13063, seed_load=seed_load,
                                plot=False)
                print('Score: ', score)
