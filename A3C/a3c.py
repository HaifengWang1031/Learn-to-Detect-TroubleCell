import os,sys
sys.path.append(os.getcwd())

import random
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
import numpy as np
from scipy.signal import lfilter
from ConservationLaw_Env.troubleCell_envs import Advection_Envs
from DG.utils import *
import argparse
import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

discount = lambda x, gamma: lfilter([1],[1,-gamma],x[:,::-1])[:,::-1] # discounted rewards one liner

init_state = [
    [sine_wave,[0,1],.5,100,1],
    [compound_wave,[-4,4],.3,800,1],
    [multi_wave,[0,1.4],0.5,140,1],
    [shock_collision,[0,1],0.5,100,1],
    [b_l_initial,[0,1],0.5,100,1]
]

def get_args():
    parser = argparse.ArgumentParser(description = None)
    parser.add_argument("--Processes",default=8,type=int)
    parser.add_argument("--epoch",default=30,type=int)
    parser.add_argument("--lr",default=1e-4,type=float)
    parser.add_argument('--tau', default=1.0, type=float)
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

class SharedAdam(torch.optim.Adam): # extend a pytorch optimizer so it shares grads across processes
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['shared_steps'], state['step'] = torch.zeros(1).share_memory_(), 0
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_().share_memory_()
                
        def step(self, closure=None):
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None: continue
                    self.state[p]['shared_steps'] += 1
                    self.state[p]['step'] = self.state[p]['shared_steps'][0] - 1 # a "step += 1"  comes later
            super.step(closure)

def cost_func(args, values, logps, actions, rewards):
    np_values = values.data.numpy()
    # generalized advantage estimation using \delta_t residuals (a policy gradient method)
    delta_t = np.asarray(rewards) + args.gamma * np_values[:,1:] - np_values[:,:-1]
    logpys = logps.view(-1,2).gather(1,actions.clone().detach().view(-1,1)).view(actions.shape)

    gen_adv_est = discount(delta_t, args.gamma * args.tau)
    policy_loss = -(logpys * torch.FloatTensor(gen_adv_est.copy())).sum()
    
    # l2 loss over value estimator
    rewards[:,-1] += args.gamma * np_values[:,-1]
    discounted_r = discount(np.asarray(rewards), args.gamma)
    discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
    value_loss = .5 * (discounted_r - values[:,:-1]).pow(2).sum()

    entropy_loss = (-logps * torch.exp(logps)).sum() # entropy definition, for entropy regularization
    return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

def train(shared_model,optim,rank,args):
    env = Advection_Envs()
    model = NNPolicy(feature_size = 7,hidden_units = args.hidden,num_action = env.action_space.n).to(device)
    
    model.load_state_dict(shared_model.state_dict()) # sync with shared vector
    values,logps,actions,rewards = [],[],[],[]

    for _ in (range(args.epoch) if rank>0 else tqdm.trange(args.epoch)):
        obs = torch.FloatTensor(env.reset(*random.choice(init_state))).to(device)

        counter = 0
        while True:
            value, logit = model(obs)
            logp = f.log_softmax(logit,dim=1)

            action = torch.exp(logp).multinomial(num_samples=1)
            obs,reward,done,_ = env.step(action.numpy())
            obs = torch.FloatTensor(obs).to(device)

            values.append(value); logps.append(logp); actions.append(action); rewards.append(reward)

            if counter == 9 or done:
                next_value,_ = model(obs)
                values.append(next_value.detach())
                loss = cost_func(args, torch.cat(values,dim=1),torch.cat(logps,dim = 1),torch.cat(actions,dim = 1),np.hstack(rewards))

                optim.zero_grad()
                loss.backward() 
                torch.nn.utils.clip_grad_norm_(model.parameters(),40)
                for param, shared_param in zip(model.parameters(),shared_model.parameters()):
                    if shared_param.grad is None:    
                        shared_param._grad = param.grad
                optim.step()

                model.load_state_dict(shared_model.state_dict())
                values,logps,actions,rewards = [],[],[],[]
                counter = 0
            else:
                counter += 1
            
            if done:
                break

    torch.save(shared_model,f"model-{rank}.pkl")
        



if __name__ == "__main__":
    args = get_args()

    shared_model = NNPolicy(7,args.hidden,2)
    optim = SharedAdam(shared_model.parameters(),lr = args.lr)

    processes = []
    for rank in range(args.Processes):
        p = mp.Process(target = train,args = (shared_model,optim,rank,args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


    torch.save(shared_model,'model.pkl')