import os,sys
sys.path.append(os.getcwd())

import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import numpy as np
from ConservationLaw_Env.troubleCell_envs import Advection_Envs
from DG.utils import *
import argparse
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

init_state = [
    [sine_wave,[0,1],.5,100,1],
    [compound_wave,[-4,4],.3,800,1],
    [multi_wave,[0,1.4],0.5,140,1],
    [shock_collision,[0,1],0.5,100,1],
    [b_l_initial,[0,1],0.5,100,1]
]

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument("--Processes",default=20,type=int)
    parser.add_argument("--epoch",default=200,type=int)
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument("--gamma",default=0.99,type=float)
    parser.add_argument("--hidden",default=64,type=int)

    return parser.parse_args()

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
    
def cost_func():
    pass

def train(shared_model,optim,rank,args):
    env = Advection_Envs()
    model = NNPolicy(feature_size = 7,hidden_units = args.hidden,num_action = env.action_space.n).to(device)
    
    for _ in (range(args.epoch) if rank>0 else tqdm.trange(args.epoch)):
        model.load_state_dict(shared_model.state_dict()) # sync with shared vector
        obs = torch.FloatTensor(env.reset(*random.choice(init_state))).to(device)

        while True:
            value, logit = model(obs)
            logp = f.log_softmax(logit,dim=1)

            action = torch.exp(logp).multinomial(num_samples=1)
            new_obs,reward,done,_ = env.step(action.numpy())
            new_obs = torch.FloatTensor(new_obs).to(device)
            reward = torch.FloatTensor(reward).to(device)

            with torch.no_grad():
                next_value,_ = model(new_obs)
                v_target = reward + args.gamma*next_value
                advantage_value = v_target - value

            # critic loss
            value_loss = .5*f.mse_loss(value,v_target)            

            # actor loss
            policy_loss = -0.5*(logp*advantage_value).sum() - 0.01*(logp*torch.exp(logp)).sum()

            loss = value_loss + policy_loss
            optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),40)

            for param, shared_param in zip(model.parameters(),shared_model.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad
            optim.step()

            if done:
                break
        



if __name__ == "__main__":
    args = get_args()

    shared_model = NNPolicy(7,args.hidden,2)
    optim = torch.optim.Adam(shared_model.parameters(),lr = args.lr)

    processes = []
    for rank in range(args.Processes):
        p = mp.Process(target = train,args = (shared_model,optim,rank,args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    torch.save(shared_model,'model.pkl')