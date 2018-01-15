import argparse
import numpy as np
import gym
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn import Linear
from torch.autograd import Variable
from torch.distributions import Categorical

"""
SIMPLE REINFORCE IMPLEMENTATION BASE ON:
    -> CS294-Berkeley Deep RL Course.
    -> https://github.com/pytorch/examples/blob/master/reinforcement_learning/reinforce.py
    -> https://github.com/JamesChuanggg/pytorch-REINFORCE
"""


class Policy(nn.Module):
    def __init__(self, in_sz, hidden_sz, out_sz):
        super(Policy, self).__init__()
        #print(in_sz, hidden_sz)
        self.fc1 = Linear(in_sz, hidden_sz)
        self.fc2 = Linear(hidden_sz, out_sz)

        self.log_probs = list()
        self.rewards = list()

    def forward(self, inputs):
        x = self.fc1(inputs)
        x = F.relu(x)
        logits = self.fc2(x)
        return F.softmax(logits)

    def act(self, inputs):
        if torch.cuda.is_available():
            inputs = Variable(Tensor(inputs).cuda())
        else:
            inputs = Variable(Tensor(inputs))

        probs = self(inputs)
        m = Categorical(probs)
        action = m.sample()
        self.log_probs.append(m.log_prob(action))
        return action.data[0]
    def learn(self, optimizer):
        losses = []
        for log_prob, reward in zip(self.log_probs, self.rewards):
            losses.append(log_prob*reward)
        optimizer.zero_grad()
        losses = torch.cat(losses).sum()
        losses.backward()
        optimizer.step()

        del self.rewards[:]
        del self.log_probs[:]

    def weight_reward(self):
        R = 0
        rewards = []

        for r in self.rewards[::-1]:
            R = r + 0.99 * R
            rewards.insert(R)
        self.rewards = Tensor(rewards)

def REINFORCE(args):
    """ ENV STUFF"""
    env = gym.make(args.env_name)
    S = env.reset()
    env.seed(args.seed)
    # ---
    in_sz = S.shape[0]
    hidden_sz = 100
    out_sz = env.action_space.n

    """ MODEL STUFF """
    model = Policy(in_sz, hidden_sz, out_sz)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()
    if torch.cuda.is_available():
        model.cuda()

    for i in range(int(1e6)):
        action = model.act(S)
        S, R, done, _ = env.step(action)

        model.rewards.append(R)

        if args.render:
            env.render()
        if done == True:
            if(i%20):
                print("EPISODE REWARD:", sum(model.rewards))
            model.learn(optimizer)
            S = env.reset()

def main():
    """ ARGUMENTS """
    parser = argparse.ArgumentParser(description="REINFORCE arguments!")
    parser.add_argument('--env_name', type=str, default="CartPole-v0")
    parser.add_argument('--render', type=bool, default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    """ CALL TO REINFORCE ALGORITHM """
    torch.manual_seed(args.seed)
    REINFORCE(args)

if __name__ == "__main__":
    main()
