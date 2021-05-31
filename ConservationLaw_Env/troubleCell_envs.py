import os,sys
sys.path.append(os.getcwd())

import gym
from gym import spaces
import numpy as np
from DG import conservation_law
from DG.utils import *
import warnings
warnings.filterwarnings("error")

class Advection_Envs(gym.Env):
    def __init__(self,basis_order = 4):
        self.solver = conservation_law.CL_Solver(legendre_basis,basis_order)

        self.action_space = spaces.Discrete(2)

    def reset(self,init_func,interval,final_T,ele_num,velocity):
        self.init_func = init_func
        self.interval = interval
        self.c = velocity
        self.ele_num = ele_num
        self.final_T = final_T

        self.real_wave = transfer_wave(self.init_func,self.interval,self.c)
        self.first_derivative_weights = 1

        self.flux = lambda x:self.c*x
        self.cfl = 0.1/np.abs(self.c)

        self.solver.reset(self.init_func,self.flux,self.interval,self.ele_num)
        self.delta_t =  self.cfl*np.max(self.solver.delta_x)
        self.solver.delta_t = self.delta_t

        self.solver.BasisWeights = self.solver.step(self.delta_t,evolution_method = "RK3")
        self.done = False
        self.current_time = self.delta_t
        self.cell_quantites = self.solver.limiter_quantites(self.solver.BasisWeights)
        return self.cell_quantites

    def step(self,action):
        BasisWeights =self.solver.poly_reconstruction(self.solver.BasisWeights,action,self.cell_quantites,lambda a,b,c,h:TVB_limiter(a,b,c,h,10))
        reward = self.calcu_reward(BasisWeights,self.current_time)

        self.solver.BasisWeights = BasisWeights.reshape((-1,1))
        self.solver.WeightContainer = np.concatenate((self.solver.WeightContainer,self.solver.BasisWeights),axis = 1)

        # calculate next obs
        try:
            self.solver.BasisWeights = self.solver.step(self.delta_t,evolution_method = "RK3")
            self.cell_quantites = self.solver.limiter_quantites(self.solver.BasisWeights)

            self.current_time += self.delta_t
            if self.current_time> self.final_T:
                self.done = True
        except Exception as e:
            self.done = True

        return self.cell_quantites,reward,self.done,[]

    def calcu_reward(self,weights,time):
        gauss_point = np.array([-0.90618,-0.538469,0,0.90618,0.538469])
        gauss_weight = np.array([0.236927,0.478629,128/255,0.478629,0.236927])
        current_wave = lambda x: self.real_wave(x,time)
        reward = np.zeros((self.solver.K,1))
        for i in range(self.solver.K):
            ele_interval = self.solver.x_h[i,:]
            ele_weights = weights[i,:]
            ele_func = lambda x:sum([weights[i][j]*self.solver.basis[j](x) for j in range(self.solver.N)])
            ele_dfunc = lambda x:sum([weights[i][j]*self.solver.Dbasis[j](x) for j in range(self.solver.N)])
            # h2 error
            h2_error = sum([gauss_weight[i]*np.abs(ele_func(x) - current_wave(x*(ele_interval[1] - ele_interval[0])/2 + (ele_interval[0] + ele_interval[1])/2)) for i,x in enumerate(gauss_point)])*(ele_interval[1]-ele_interval[0])/2
            # l1 error
            l1_error = sum([gauss_weight[i]*np.abs(ele_dfunc(x) -\
                            (current_wave((x + 0.001)*(ele_interval[1] - ele_interval[0])/2 + (ele_interval[0] + ele_interval[1])/2) -\
                             current_wave((x - 0.001)*(ele_interval[1] - ele_interval[0])/2 + (ele_interval[0] + ele_interval[1])/2))/0.002)\
                           for i,x in enumerate(gauss_point)])*(ele_interval[1]-ele_interval[0])/2
            reward[i,0] = -np.log10(h2_error + 1e-18) - 0.01*self.first_derivative_weights*np.log10(l1_error + 1e-18)
        return reward


    def render(self,):
        self.solver.render()
        self.solver.draw_troubleCell()


if __name__ == "__main__":
    slope_limiter = [minmod_limiter, #minmod
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,10), # TVB-1
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,100), # TVB-2
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,1000)][1] # TVB-3

    e = Advection_Envs()

    obs = e.reset(b_l_initial,[0,1],.3,100,1)
    import torch
    import torch.nn as nn
    import torch.nn.functional as f

    class NNPolicy(nn.Module):
        def __init__(self,feature_size,hidden_units,num_action):
            super().__init__()
            self.linear1 = nn.Linear(feature_size,hidden_units)
            self.linear2 = nn.Linear(hidden_units,hidden_units)
            self.critic_linear, self.actor_linear = nn.Linear(hidden_units,1),nn.Linear(hidden_units,num_action)
    
        def forward(self,inputs):
            x = f.elu(self.linear1(inputs))
            x = f.elu(self.linear2(x))
            return self.critic_linear(x),self.actor_linear(x)
    model = torch.load('model.pkl')
    
    while True:
        # action = e.solver.trouble_cell_indicator(obs,slope_limiter).astype(np.int8)
        value,logit =model.forward(torch.FloatTensor(obs))
        logp = f.log_softmax(logit,dim=1)
        action = torch.exp(logp)#.multinomial(num_samples=1).numpy()

        action = torch.where(action[:,1]>0.7,1,0).view((-1,1))

        e.solver.Trouble_Cell = np.concatenate((e.solver.Trouble_Cell,action),axis =1)

        obs,reward,done,info = e.step(action)
        if done:
            break
    e.render()


