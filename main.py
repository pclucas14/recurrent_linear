import torch
import torch.nn as nn
import torch.nn.functional as F
from comet_ml import Experiment
import numpy as np
import argparse
import pdb
import gym
import os


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='cartpole')
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--loss', type=str, default='cramer')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--num_episodes', type=int, default=1000)
parser.add_argument('--n_backprop', type=int, default=1)
parser.add_argument('--epsilon', type=float, default=0.2)
args = parser.parse_args()

# reproducibility
if args.seed:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# GPU settings 
USE_CUDA = False #torch.cuda.is_available()
default_tensor = 'torch.cuda.FloatTensor' if USE_CUDA else 'torch.FloatTensor'
torch.set_default_tensor_type(default_tensor)

experiment = Experiment(api_key="HFFoR5WtTjoHuBGq6lYaZhG0c",
                        project_name="recurrent-linear", 
                        workspace="pierthodo",
                        log_code=False,
                        auto_output_logging=None,
                        log_graph=False, 
                        auto_param_logging=False, 
                        auto_metric_logging=False, 
                        parse_args=False, 
                        log_env_details=False, 
                        log_git_metadata=False,
                        log_git_patch=False)

experiment.log_multiple_params(vars(args))

# environment settings
if 'cartpole' in args.env.lower():
    env = gym.make('CartPole-v0')
    num_frames = 20000
elif 'acrobot' in args.env.lower():
    env = gym.make('Acrobot-v1')
    num_frames = 40000
else: 
    raise ValueError('invalid environment')

state_space  = env.observation_space.shape[0]
action_space = env.action_space.n

# useful wrappers
to_np = lambda x : x.cpu().data.numpy() if 'Tensor' in str(type(x)) else x

""" linear fct approximator """
class linear_model(nn.Module):
    def __init__(self, state_size, action_size):
        super(linear_model, self).__init__()
        init_ = torch.nn.init.orthogonal_
        self.main = nn.Linear(state_space, action_space, bias=False)
        init_(self.main.weight)
        
        if args.n_backprop > 1: 
            self.beta_n = nn.Sequential(nn.Linear(state_space, 1, bias=False), nn.Sigmoid())
            init_(self.beta_n[0].weight)

    def choose_action(self, state, det=False):
        state = torch.Tensor(state)
        if np.random.rand() < eps and not det:
            action = np.random.randint(action_space)
        else:
            state  = torch.tensor(state)
            probs  = self.main(state)
            action = probs.max(dim=-1)[1]

        return action

    def q_value(self, state, action):
        state = torch.Tensor(state)
        return self.main(state)[action]

    def forward(self, state):
        return self.main(torch.Tensor(state))

    def beta(self, state):
        return self.beta_n(torch.Tensor(state))

model = linear_model(state_space, action_space)
model = model.cuda() if USE_CUDA else model
optim = torch.optim.SGD(model.parameters(), lr=0.1)
eps = args.epsilon

all_rewards, all_betas = [], []

for ep in range(args.num_episodes):
    state = env.reset()
    ep_reward = []
    action_counter = np.zeros((action_space))
    done = False

    # TODO: check with Pierre what 1st value should be
    previous_value = model(state).mean().detach()

    while not done: 
        # pick action
        action = to_np(model.choose_action(state))
        q_hat  = model.q_value(state, action)
        action_counter[action] += 1
        betas = []

        # execute chosen action
        next_state, reward, done, _ = env.step(action)
        ep_reward += [reward]

        if args.n_backprop > 1:
            beta = model.beta(state)
            betas += [beta.data]
            q_est = (1 - beta) * (previous_value - reward) + beta * q_hat
            previous_value = q_est
        else:
            q_est = q_hat

        if done: 
            loss = (q_est - reward) ** 2 * 0.5
        else:
            next_action = model.choose_action(state) 
            q_hat_next  = model.q_value(next_state, next_action)
            loss = (q_est - (reward + args.gamma * q_hat_next).detach()) ** 2 * 0.5

        optim.zero_grad()
        loss.backward(retain_graph=args.n_backprop > 1)
        optim.step()

        state = next_state

    all_rewards += [sum(ep_reward)]
    if args.n_backprop > 1: all_betas += [np.mean(betas)]
    
    if (ep + 1) % 100 == 0: 
        eps = eps / 2.
        print('new eps : {:.4f}'.format(eps))


    if (ep + 1) % 10 == 0: 
        value_dict = {"mean_reward" : np.mean(all_rewards)}
        if args.n_backprop: value_dict["mean_beta"] = np.mean(all_betas)
        experiment.log_multiple_metrics(value_dict, step=ep)

