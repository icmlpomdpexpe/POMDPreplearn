import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class LSVI_UCB(object): 

    def __init__(
        self,
        z_dim, buffers,
        state_dim,
        action_dim,
        horizon,
        alpha,
        device,
        rep_learners,
        lamb = 1,
        recent_size=0,
    ):
        self.z_dim = z_dim
        #self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon
        self.buffers = buffers
        self.feature_dim = state_dim * action_dim

        self.device = device

        self.rep_learners = rep_learners

        self.lamb = lamb
        self.alpha = alpha

        self.recent_size = recent_size

        self.W_z = torch.rand((self.horizon, self.feature_dim)).to(self.device)
        #print("W")
        #print(self.W.shape)
        self.Sigma_z_invs = torch.zeros((self.horizon, self.feature_dim, self.feature_dim)).to(self.device)

        self.Q_max = torch.tensor(self.horizon)

    def Q_values(self, z, h):
        
        self.update(self.buffers, z, h)
        
        if h == self.horizon:
            Qs = torch.zeros((len(z),self.action_dim)).to(self.device)
        else:
            Qs = torch.zeros((len(z),self.action_dim)).to(self.device)
            for a in range(self.action_dim):
                actions = torch.zeros((len(z),self.action_dim)).to(self.device)
                actions[:,a] = 1
                with torch.no_grad():
                    feature = self.rep_learners[h].phi(z,actions)
            
            #print("feature")
            #print(feature.shape)
            
                Q_est = torch.matmul(feature, self.W_z[h].to(self.device)) 
                ucb = torch.sqrt(torch.sum(torch.matmul(feature, self.Sigma_z_invs[h].to(self.device))*feature, 1))
            
                Qs[:,a] = torch.minimum(Q_est + self.alpha * ucb, self.Q_max)
        #print(h)
        return Qs

    def act_batch(self, z, h):
        with torch.no_grad():
            z = torch.FloatTensor(z).to(self.device)
            Qs = self.Q_values(z, h)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()


    def act(self, z, h):
        with torch.no_grad():
            z = torch.FloatTensor(z).to(self.device)
            z = z.unsqueeze(0)
            Qs = self.Q_values(z, h)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()

    def update(self, buffers, z, h):
        assert len(buffers) == self.horizon

        
        if self.recent_size > 0:
            zs,  actions, rewards, next_zs, next_obses = buffers[h].get_full(device=self.device, recent_size=self.recent_size)
        else:
            zs, actions, rewards, next_zs, next_obses = buffers[h].get_full(device=self.device)
            
        with torch.no_grad():
            
            feature_z_phi = self.rep_learners[h].phi(zs,actions)
            
                
        Sigma_z = torch.matmul(feature_z_phi.T, feature_z_phi) + self.lamb * torch.eye(self.feature_dim).to(self.device)
        self.Sigma_z_invs[h] = torch.inverse(Sigma_z)
        next_zs_z=next_obses.numpy()
        zs_z=zs.numpy()
        prev_obses = zs_z[:, 8:]
        next_zs_z = np.concatenate((prev_obses, next_zs_z), axis=1)
        next_zs_z = torch.as_tensor(next_zs_z)
        if h == self.horizon - 1:
            target_Q = rewards
                
        else:
            Q_prime = torch.max(self.Q_values(next_zs_z, h+1),dim=1)[0].unsqueeze(-1)
            target_Q = rewards + Q_prime
                
            #target_Q = target_Q.expand(feature_z_mu.shape[0], 1)
            
            #print("feature_z_mu size:", feature_z_mu.shape)
            #print("target_Q size:", target_Q.shape)
        self.W_z[h] = torch.matmul(self.Sigma_z_invs[h].to(self.device), torch.sum(feature_z_phi * target_Q, 0))            
       
            #print("W")
            #print(self.W[h].shape)
    def save_weight(self, path):
        for h in range(self.horizon):
           
            torch.save(self.Sigma_invs[h], "{}/Sigma_{}.pth".format(path,str(h)))

    def load_weight(self, path):
        for h in range(self.horizon):
           
            self.Sigma_invs[h] = torch.load("{}/Sigma_{}.pth".format(path,str(h)))








