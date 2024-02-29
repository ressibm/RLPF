import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sig
import math
import random
import pandas as pd

def cot(angle):
    return math.cos(angle) / math.sin(angle)

class SubsurfaceState:
    def __init__(self, pos, angle, decfreq=16, discrete=500):
        self.pos = pos
        self.angle = angle
        self.step = 0
        self.last_state = [self.pos, self.angle]
        self.state_array = np.zeros([discrete,len(self.last_state)])
        self.state_array[0] = self.last_state

        self.distance = 10
        self.max_angle_change = 0.035
        self.fault_through_max = 20
        self.decfreq = decfreq
        self.discrete = discrete
        self.idx = 0

        self.fault_idx = None
        self.fault_positive = 0
        self.fault_num = 0
        self.fault_max = 2
        self.fault_particle = True
    
    def get_next_state(self):
        self.idx += 1

        self.pos, self.angle = self.last_state
        self.pos += 0.2
        self.last_state = [self.pos, self.angle]

        self.state_array[self.idx] = self.last_state
        return self.last_state

    def get_next_state_random_true(self):
        self.idx += 1
        new_angle = self.angle + (random.random() - 0.5) * 0.01
        new_cot = cot(new_angle)
        delta_angle = 0
        if (0.5 + random.random())*0.03 < abs(new_cot) and random.random() > 0.2:
            sign = -np.sign(-new_cot)
            magnitude = abs(new_cot) * (1.0 + (random.random() - 0.5) * 0.8)
            delta_angle = sign * min(self.max_angle_change, magnitude)
        new_angle += delta_angle

        new_position = self.pos + self.distance * cot(new_angle)
        self.pos = new_position
        self.angle = new_angle

        delta = 0
        if (0.5 + random.random())*7 < abs(new_position) and random.random() > 0.8 \
            and self.idx > (self.discrete - self.decfreq)//3  and self.idx < (self.discrete - self.decfreq)*4//5 \
                and self.fault_num < self.fault_max:
            if self.fault_idx is None:
                sign = np.sign(-new_position)
                magnitude = abs(new_position) * (1.5 + (random.random() - 0.2))
                delta = sign * min(self.fault_through_max, magnitude)

                self.fault_num += 1
                if sign == -1:
                    self.fault_positive += 1
                self.fault_idx = 0
            elif self.fault_idx > self.discrete//3:
                sign = np.sign(-new_position)
                magnitude = abs(new_position) * (1.5 + (random.random() - 0.2))
                delta = sign * min(self.fault_through_max, magnitude)

                self.fault_num += 1
                self.fault_idx = 0

        if self.fault_idx != None:
            self.fault_idx += 1

        new_position += delta
        self.pos = new_position

        self.last_state =  np.array([self.pos, self.angle])
        self.state_array[self.idx] = self.last_state
        return self.last_state
    
    def get_next_state_random(self):
        self.idx += 1
        new_angle = self.angle + (random.random() - 0.5) * 0.01
        new_cot = cot(new_angle)
        delta_angle = 0
        if (0.5 + random.random())*0.03 < abs(new_cot) and random.random() > 0.2:
            sign = -np.sign(-new_cot)
            magnitude = abs(new_cot) * (1.0 + (random.random() - 0.5) * 0.8)
            delta_angle = sign * min(self.max_angle_change, magnitude)
        new_angle += delta_angle

        new_position = self.pos + self.distance * cot(new_angle)
        self.pos = new_position
        self.angle = new_angle

        delta = 0

        if self.idx%32 == 0:
            self.fault_particle = True

        if (0.5 + random.random())*7 < abs(new_position)\
                and random.random() > 0.8 and self.fault_particle == True:
            sign = np.sign(-new_position)
            magnitude = abs(new_position) * (1.5 + (random.random() - 0.2))
            delta = sign * min(self.fault_through_max, magnitude)
            self.fault_particle = False
        

        new_position += delta
        self.pos = new_position

        self.last_state =  np.array([self.pos, self.angle])
        self.state_array[self.idx] = self.last_state
        return self.last_state

    def generate_states(self, random=False):
        self.state_array = np.zeros([self.discrete,len(self.last_state)])
        self.state_array[0] = self.last_state
        for idx in range(1,self.discrete):
            if random:
                next_state = self.get_next_state_random()
            else:
                next_state = self.get_next_state()
            self.state_array[idx] = np.array(next_state)
        return self.state_array

def eval_along_y(state_array, ref_data, noise=False):
    pos_array = state_array

    if noise:
        noize_std=None
        noize_rel_std=0.01
        i0s = np.floor(pos_array).astype(int)
        i1s = i0s + 1
        curves_values0 = ref_data[i0s]
        curves_values1 = ref_data[i1s]
        dists0 = pos_array - i0s
        dists1 = i1s - pos_array
        if noize_std is None:
            min_data = np.min(ref_data)
            max_data = np.max(ref_data)
            data_range = max_data - min_data
            noize_std = data_range * noize_rel_std

        correlation_scale = 8
        dist = np.arange(-correlation_scale, correlation_scale)
        noise = np.random.normal(scale=noize_std, size=curves_values0.size)
        filter_kernel = np.exp(-dist ** 2 / (2 * correlation_scale))
        noise_correlated = sig.fftconvolve(noise, filter_kernel, mode='same')

        gamma = dists1 * curves_values0 + dists0 * curves_values1 + noise_correlated
    else:
        i0s = np.floor(pos_array).astype(int)
        i1s = i0s + 1
        curves_values0 = ref_data[i0s]
        curves_values1 = ref_data[i1s]
        dists0 = pos_array - i0s
        dists1 = i1s - pos_array
        gamma = dists1 * curves_values0 + dists0 * curves_values1
    return gamma
    
def plot_states(states,obs,obs_noise=None):
    x = states[:,2]
    ax1 = plt.subplot2grid((2, 1), (0, 0))
    ax1.plot(x,states[:,0])
    ax2 = plt.subplot2grid((2, 1), (1, 0))
    ax2.plot(x,obs)
    if isinstance(obs_noise,(np.ndarray)):
        ax2.plot(x,obs_noise)
    plt.show()

if __name__ == "__main__":
    num_eval = 1
    discrete = 300

    env = SubsurfaceState(discrete=discrete)
    for idx in range(num_eval):
        state_array = env.generate_states(random=True)
        gamma = env.eval_along_y(noise=False)
        gamma_noise = env.eval_along_y(noise=True)
        env = SubsurfaceState(*env.last_state, discrete=discrete)
        plot_states(states=state_array,obs=gamma,obs_noise=gamma_noise)


