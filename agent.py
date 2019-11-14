import torch
import torch.nn.functional as F
import numpy as np

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def fanin_init(size, fanin=None):
    """Utility function for initializing actor and critic"""
    fanin = fanin or size[0]
    w = 1./ np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-w, w)

class Actor(torch.nn.Module):
    def __init__(self, inputs, actions, lr, tau=0.001):
        super(Actor, self).__init__()

        self.tau = tau
        self.action_len = actions

        self._fc1 = torch.nn.Linear(inputs, 64)
        self._relu1 = torch.nn.ReLU(inplace=True)

        self._fc2 = torch.nn.Linear(64, 64)
        self._relu2 = torch.nn.ReLU(inplace=True)

        self._fc3 = torch.nn.Linear(64, actions)

        self._fc1.weight.data = fanin_init(self._fc1.weight.data.size())      
        self._fc2.weight.data = fanin_init(self._fc2.weight.data.size())
        self._fc3.weight.data.uniform_(-0.003, 0.003)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, inputs):
        fc1 = self._relu1(self._fc1(inputs))
        fc2 = self._relu2(self._fc2(fc1))
        
        return self._fc3(fc2)

    def train_step(self, q_target, q_expected):
        loss = torch.nn.functional.mse_loss(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def hard_copy(self, actor):
        for param, target_param in zip(actor.parameters(), self.parameters()):
            target_param.data.copy_(param.data)
    
    def save_model(self, filename):
        torch.save(self.state_dict(), filename)

    def load_model(self, filename):
        self.load_state_dict(torch.load(filename))
        self.eval()