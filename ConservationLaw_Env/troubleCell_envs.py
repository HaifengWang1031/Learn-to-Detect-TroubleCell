import os,sys
sys.path.append(os.getcwd())

import gym 
from gym import spaces
import numpy as np
from DG import conservation_law
from DG.utils import *

class Advection_Envs(gym.Env):
    def __init__(self,basis_order = 4):
        self.solver = conservation_law.CL_Solver(legendre_basis,basis_order)

        self.obseravtion_space = spaces.Box(low = np.finfo(np.float32).min,high = np.finfo(np.float32).max,shape=(7,))
        self.action_space = spaces.Discrete(2)

    def reset(self,init_func,interval,final_T,ele_num,velocity):
        self.init_func = init_func
        self.interval = interval
        self.c = velocity
        self.ele_num = ele_num
        self.final_T = final_T

        self.real_wave = transfer_wave(self.init_func,self.interval,self.c)
        
        self.flux = lambda x:self.c*x
        self.cfl = 0.1/np.abs(self.c)
        
        self.solver.reset(self.init_func,self.flux,self.interval,self.ele_num)
        self.delta_t =  self.cfl*np.max(self.solver.delta_x)
        self.solver.delta_t = self.delta_t

        self.solver.BasisWeights = self.solver.step(self.delta_t,evolution_method = "Euler")
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
        self.solver.BasisWeights = self.solver.step(self.delta_t,evolution_method = "RK3")
        self.cell_quantites = self.solver.limiter_quantites(self.solver.BasisWeights)

        self.current_time += self.delta_t
        if self.current_time> self.final_T:
            self.done = True

        return self.cell_quantites,reward,self.done,[]

    def calcu_reward(self,weights,time):
        #TODO
        return np.zeros((len(weights),1))
        

    def render(self,):
        self.solver.render()
        self.solver.draw_troubleCell()

        
if __name__ == "__main__":
    slope_limiter = [minmod_limiter, #minmod
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,10), # TVB-1
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,100), # TVB-2
                    lambda a,b,c,h:TVB_limiter(a,b,c,h,1000)][2] # TVB-3

    e = Advection_Envs()
    
    obs = e.reset(compound_wave,[-4,4],1,100,1)
    print(obs.shape)
    for i in range(1):
        action = e.solver.trouble_cell_indicator(obs,slope_limiter)
        print(action.shape)
        obs,reward,done,info = e.step(action)
        if done:
            break
    e.render()
    