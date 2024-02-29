import os
import numpy as np
from agent import DQNAgent
from tqdm import tqdm
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from geosteering_v1 import Geosteering
from eval import eval_func


def train_func(n_games, inputs, best, dec, thickness, seed, logs):
    if __name__ == "__main__":
        np.random.seed(seed)
        random.seed(seed)

        string_name = "%s_best=%s_%s_%s" % (inputs, best, seed, dec)

        if logs:
            log_dir = "logs/presentation/%s" % (string_name)
            os.makedirs(log_dir, exist_ok=True)
            writer = SummaryWriter(log_dir=log_dir)

        env = Geosteering(inputs=inputs, bestrl=best, dec=dec, thickness=thickness, resample=2)

        eps_min = 0.01
        eps_dec = 0.9997

        scores = np.zeros([n_games])
        avg_score = np.zeros([n_games])
        if inputs == "PF":
            states = ((env.DecisionFreq + 1) * env.bestrl * 2 + env.bestrl + 2,)
            rmsd_gamma = 0
            rmsd_state = 0
            multi = 2
        elif inputs == "Sensor":
            states = ((env.DecisionFreq + 1) * 2 + 2,)
            multi = 4
        elif inputs == "Gamma":
            states = ((env.DecisionFreq + 1) + 2,)
            multi = 4
        bd_actions = 11

        bd_agent = DQNAgent(n_actions=bd_actions, n_states=states, seed=seed, multi=multi, batch_size=64,
                            mem_size=50000, saved_dir="trained_network/", env_name="v2/%s" % (string_name),)

        eval_score = np.zeros(n_games // 500 + 1)
        best_score = -np.inf

        for i in range(n_games):
            done = False
            observation, reward, done, info = env.reset()
            score = 0
            while not done:
                action = bd_agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                if env.nSteps * 3.5 / 10 < env.bd <= env.nSteps + 1:
                    score += reward
                bd_agent.store_transition(
                    observation, action, reward, observation_, done)
                bd_agent.learn()
                observation = observation_

            bd_agent.epsilon = (
                bd_agent.epsilon * eps_dec if bd_agent.epsilon > eps_min else eps_min)

            scores[i] = score
            if i >= 100:
                avg_score[i] = np.mean(scores[i - 99: i + 1])
                if logs:
                    writer.add_scalar("Reward", avg_score[i], i)

            print("===STEP: %s, SCORE: %s, AVERAGE: %s, EPSILON: %s===" %
                  (i, scores[i], avg_score[i], bd_agent.epsilon))

            if i > 0 and i % 500 == 0:
                score = eval_func(n_games=100, inputs=inputs, best=best, dec=dec,
                                  thickness=thickness, seed_eval=0, bd_agent=bd_agent, load=False, plot=False,)
                if logs:
                    writer.add_scalar("Eval Score", score, i // 500)

                print("======EVAL %s======" % (i // 500))
                print("======SCORE: %s======" % score)
                if score > best_score:
                    print("======SAVE NEW AGENT======")
                    bd_agent.save_models()
                    best_score = score
                eval_score[int(i // 500)] = score

            if inputs == "PF":
                rmsd_gamma += np.mean(np.abs(np.apply_along_axis(
                    env.GenerateGammaObs, 0, env.bitdepth - env.best_state) - env.gamma))
                rmsd_state += np.mean(np.abs(env.best_state - env.boundary))

        if inputs == "PF":
            print(rmsd_gamma / n_games)
            print(rmsd_state / n_games)

        print("Best score %.2f" % best_score)


for seed in [9, 140, 272, 387, 400, 571, 624, 733, 898, 956, 1000]:
    for best in [5]:        
        train_func(n_games=20000 + 1, inputs="PF", best=best,
                   dec=32, thickness=20, seed=seed, logs=True,)

#8, 117, 224, 360, 439, 539, 670, 759, 808, 903