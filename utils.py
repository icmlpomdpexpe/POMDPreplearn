import argparse
import torch
import numpy as np

import random
import os

from envs.Lock_batch import LockBatch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_name', default="test", type=str)
    parser.add_argument('--num_threads', default=10, type=int)
    parser.add_argument('--update_frequency', default=1, type=int)

    parser.add_argument('--temp_path', default="temp", type=str)

    parser.add_argument('--num_envs', default=50, type=int)
    parser.add_argument('--recent_size', default=10000, type=int)
    parser.add_argument('--lsvi_recent_size', default=1000, type=int)
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--dense', default=False, type=bool)

    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--num_warm_start', default=0, type=int)
    parser.add_argument('--num_episodes', default=1000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)


    #environment
    parser.add_argument('--horizon', default=100, type=int)
    parser.add_argument('--switch_prob', default=0.5, type=float)
    parser.add_argument('--anti_reward', default=0.1, type=float)
    parser.add_argument('--anti_reward_prob', default=0.5, type=float)
    parser.add_argument('--num_actions', default=10, type=int)
    parser.add_argument('--observation_noise', default=0.1, type=float)
    parser.add_argument('--variable_latent', default=False, type=bool)
    parser.add_argument('--env_temperature', default=0.2, type=float)

    #rep
    parser.add_argument('--rep_num_update', default=10, type=int)
    parser.add_argument('--rep_num_feature_update', default=8, type=int)
    parser.add_argument('--rep_num_adv_update', default=8, type=int)
    parser.add_argument('--discriminator_lr', default=1e-2, type=float)
    parser.add_argument('--discriminator_beta', default=0.9, type=float)
    parser.add_argument('--feature_lr', default=1e-2, type=float)
    parser.add_argument('--feature_beta', default=0.9, type=float)
    parser.add_argument('--linear_lr', default=1e-2, type=float)
    parser.add_argument('--linear_beta', default=0.9, type=float)
    parser.add_argument('--rep_lamb', default=0.01, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--phi0_temperature', default=0.1, type=float)

    parser.add_argument('--reuse_weights', default=True, type=bool)
    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--softmax', default='vanilla', type=str)

    #lsvi
    parser.add_argument('--alpha', default=0.1, type=float)
    parser.add_argument('--lsvi_lamb', default=1, type=float)

    #eval
    parser.add_argument('--num_eval', default=20, type=int)

    args = parser.parse_args()
    return args

def make_batch_env(args):
    env = LockBatch()
    env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_envs,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    #env.seed(args.seed)
    #env.action_space.seed(args.seed)

    eval_env = LockBatch()
    eval_env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_eval,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    #eval_env.seed(args.seed)
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b

    return env, eval_env

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

class Buffer(object):
    def __init__(self, num_actions):
        
        self.num_actions = num_actions
        self.zs = []
        self.next_zs = []
        self.actions = []
        self.rewards = []
        self.idx = 0
        
    def add(self, z, action, reward, next_z):
        self.zs.append(z)
        
        aoh = np.zeros(self.num_actions)
        aoh[action] = 1
        self.actions.append(aoh)
        self.rewards.append(reward)
        self.next_zs.append(next_z)

        self.idx += 1

    def get_batch(self):
        return self.idx, np.array(self.zs), np.array(self.actions), np.array(self.rewards), np.array(self.next_zs) 

    def get(self, h):
        print(self.next_zs[h].shape)
        return self.zs[h], self.actions[h], self.rewards[h], self.next_zs[h]


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, z_shape, num_actions, capacity, batch_size, device, recent_size=0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions

        self.zs = np.empty((capacity, *z_shape), dtype=np.float32)
        
        self.next_zs = np.empty((capacity, *z_shape), dtype=np.float32)
        self.actions = np.empty((capacity, num_actions), dtype=np.int)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)

        self.recent_size = recent_size

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, z, action, reward, next_z):
        np.copyto(self.zs[self.idx], z)
        aoh = np.zeros(self.num_actions, dtype=np.int)
        aoh[action] = 1
        np.copyto(self.actions[self.idx], aoh)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_zs[self.idx], next_z)
        #print("add_z")
        #print(z.shape)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, z, action, reward, next_z, size):
        #print("add_batch_z")
        #print(z.shape)
        #print("add_batch_zs")
        #print(self.zs.shape)
        np.copyto(self.zs[self.idx:self.idx+size], z)
        aoh = np.zeros((size,self.num_actions), dtype=np.int)
        aoh[np.arange(size), action] = 1
        np.copyto(self.actions[self.idx:self.idx+size], aoh)
        np.copyto(self.rewards[self.idx:self.idx+size], reward)
        np.copyto(self.next_zs[self.idx:self.idx+size], next_z)

        self.idx = (self.idx + size) % self.capacity
        self.full = self.full or self.idx == 0

    def add_from_buffer(self, buf, h):
        z, action, reward, next_z = buf.get(h)
        
        np.copyto(self.zs[self.idx], z)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_zs[self.idx], next_z)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_full(self, recent_size=0, device=None):

        if device is None:
            device = self.device

        if self.idx <= recent_size or recent_size == 0: 
            start_index = 0
        else:
            start_index = self.idx - recent_size

        if self.full:
            self.next_obses = self.next_zs[:, 8:]

            zs = torch.as_tensor(self.zs[start_index:], device=device)
            actions = torch.as_tensor(self.actions[start_index:], device=device)
            rewards = torch.as_tensor(self.rewards[start_index:], device=device)
            next_zs = torch.as_tensor(self.next_zs[start_index:], device=device)
            next_obses = torch.as_tensor(self.next_obses[start_index:], device=device)
            return zs, actions, rewards, next_zs, next_obses
                
        else:
            self.next_obses = self.next_zs[:, 8:]
            zs = torch.as_tensor(self.zs[start_index:self.idx], device=device)
            actions = torch.as_tensor(self.actions[start_index:self.idx], device=device)
            rewards = torch.as_tensor(self.rewards[start_index:self.idx], device=device)
            next_zs = torch.as_tensor(self.next_zs[start_index:self.idx], device=device)
            next_obses = torch.as_tensor(self.next_obses[start_index:self.idx], device=device)
            return zs, actions, rewards, next_zs, next_obses
    def get_full_z(self, z, recent_size=0, device=None):

        if device is None:
            device = self.device

        if self.idx <= recent_size or recent_size == 0: 
            start_index = 0
        else:
            start_index = self.idx - recent_size

        if self.full:
            zs=self.zs
            
            rewards=self.rewards

            
            next_zs=self.next_zs
            next_obses = next_zs[:, 8:]

            rewards = [self.rewards[i] for i in range(len(self.zs)) if self.zs[i] == z]
            actions = [self.actions[i] for i in range(len(self.zs)) if self.zs[i] == z]
            next_obses = [next_obses[i] for i in range(len(self.zs)) if self.zs[i] == z]
            next_zs = [self.next_zs[i] for i in range(len(self.zs)) if self.zs[i] == z]
            zs = [a for a in self.zs if a == z]
            zs = [a for a in zs if a == zs[0]]
            
            
                    
            zs = torch.as_tensor(zs[start_index:], device=device)
            actions = torch.as_tensor(actions[start_index:], device=device)
            rewards = torch.as_tensor(rewards[start_index:], device=device)
            next_zs = torch.as_tensor(next_zs[start_index:], device=device)
            
            return zs, actions, rewards, next_zs, next_obses
                
        else:
            zs=self.zs
            
            rewards=self.rewards

            
            next_zs=self.next_zs
            next_obses = next_zs[:, 8:]

            rewards = [self.rewards[i] for i in range(len(self.zs)) if self.zs[i] == z]
            actions = [self.actions[i] for i in range(len(self.zs)) if self.zs[i] == z]
            next_obses = [next_obses[i] for i in range(len(self.zs)) if self.zs[i] == z]
            next_zs = [self.next_zs[i] for i in range(len(self.zs)) if self.zs[i] == z]
            zs = [a for a in self.zs if a == z]
            zs = [a for a in zs if a == zs[0]]
            
            
                    
            zs = torch.as_tensor(zs[start_index:self.idx], device=device)
            actions = torch.as_tensor(actions[start_index:self.idx], device=device)
            rewards = torch.as_tensor(rewards[start_index:self.idx], device=device)
            next_zs = torch.as_tensor(next_zs[start_index:self.idx], device=device)
            
            return zs, actions, rewards, next_zs, next_obses
    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.recent_size == 0 or self.idx < self.recent_size: 
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size 
            )
        else:
            idxs = np.random.randint(
                self.idx - self.recent_size, self.capacity if self.full else self.idx, size=self.batch_size 
            )


        zs = torch.as_tensor(self.zs[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_zs = torch.as_tensor(self.next_zs[idxs], device=self.device)
        
        return zs, actions, rewards, next_zs

        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.zs[self.last_save:self.idx],
            self.next_zs[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.zs[start:end] = payload[0]
            self.next_zs[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.idx = end
