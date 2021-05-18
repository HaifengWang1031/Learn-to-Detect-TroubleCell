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

    def reset(self,init_func,interval,velocity=1):
        self.init_func = init_func
        self.interval = interval
        self.c = velocity

    def step(self,action):
        pass

    def render(self,):
        pass

        
if __name__ == "__main__":
    pass
    