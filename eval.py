import numpy as np
import torch as T
from agent import DQNAgent
import random
import time
import copy

from geosteering_v1 import Geosteering

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

        string = "trained_network/%s_best=%s_%s_%s.pth"

        bd_agent_eval.Q_eval.load_state_dict(
            T.load(string % (inputs, best, seed_load, dec)))

    else:
        bd_agent_eval = copy.deepcopy(bd_agent)

    bd_agent_eval.epsilon = 0

    scores = np.full(n_games, np.nan)
    contact = np.full(n_games, np.nan)

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
        print('Time = ', (time.time() - start))
        print('Reservoir contact: ', (224 + score)/224)

        if env_eval.faults_num >= 0:
            scores[i] = score
            contact[i] = env_eval.contact
            env_eval.FullPlot()

            if inputs == "PF":
                beststate[i,] = env_eval.best_state

                rmsd_gamma += np.mean(np.abs(np.apply_along_axis(env_eval.GenerateGammaObs,
                                    0, env_eval.bitdepth - env_eval.best_state[:],) - env_eval.gamma))
                rmsd_state += np.mean(
                    np.abs(env_eval.best_state[:] - env_eval.boundary))

        seed_eval += 1

    idx_not_nan = np.where(~np.isnan(scores))[0]

    if inputs == "PF":
        print('Mean gamma error: ', rmsd_gamma / len(idx_not_nan))
        print('Mean state error: ', rmsd_state / len(idx_not_nan))

    return np.mean(scores[idx_not_nan])


if __name__ == "__main__":
    for best in [1]:
        for seed_load in [9]:
            for dec in [32]:
                score = eval_func(n_games=10, inputs="PF", best=best,
                                dec=dec, thickness=20, seed_eval=13063, seed_load=seed_load,
                                plot=False)
                print('Mean score: ', score)