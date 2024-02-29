import os
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQNetwork(nn.Module):
    def __init__(self, lr, out_dims, input_dims, name, saved_dir, fc1_dims, fc2_dims, fc3_dims, seed):
        super().__init__()
        self.checkpoint_file = os.path.join(saved_dir, name)
        T.manual_seed(seed)
        self.fc1 = nn.Linear(*input_dims, fc1_dims)
        self.fc2 = nn.Linear(fc1_dims, fc2_dims)
        self.fc3 = nn.Linear(fc2_dims, fc3_dims)
        self.fc4 = nn.Linear(fc3_dims, out_dims)

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.xavier_normal_(self.fc3.weight)
        nn.init.constant_(self.fc3.bias, 0)
        nn.init.xavier_normal_(self.fc4.weight)
        nn.init.constant_(self.fc4.bias, 0)
        
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.HuberLoss()
        self.device = T.device('cuda')
        self.to('cuda')
        
    def forward(self, state):
        actions = self.fc4(F.relu(self.fc3(F.relu(self.fc2(F.relu(self.fc1(state)))))))
        return actions
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)
        
class ReplayBuffer(object):
    def __init__(self, mem_size, input_shape):
        self.mem_size = mem_size
        self.mem_idx = 0
        
        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.next_state_memory = np.copy(self.state_memory)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, next_state, done):
        idx = self.mem_idx % self.mem_size
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.terminal_memory[idx] = done
        self.mem_idx += 1

    def sample_buffer(self, batch_size):
        mem_size = min(self.mem_idx, self.mem_size)
        batch = np.random.choice(mem_size, batch_size, replace=False)

        states = T.tensor(self.state_memory[batch]).to(T.device('cuda'))
        actions = T.tensor(self.action_memory[batch]).to(T.device('cuda'))
        rewards = T.tensor(self.reward_memory[batch]).to(T.device('cuda'))
        next_states = T.tensor(self.next_state_memory[batch]).to(T.device('cuda'))
        terminal = T.tensor(self.terminal_memory[batch]).to(T.device('cuda'))

        return states, actions, rewards, next_states, terminal