import torch
from torch import nn
from models.network_fbcnn import QFAttention, ResBlock, sequential


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class QFNet(nn.Module):
    def __init__(self, in_c):
        super(QFNet, self).__init__()
        self.fuses = nn.ModuleList([])
        nb = 4
        nc = [1280, 1280, 640, 320]
        # for i in range(4):
        #     self.fuses.append(nn.ModuleList([*[QFAttention(nc[i], nc[i], bias=True, mode='C' + 'R' + 'C') for _ in range(nb)]]))

        self.qf_pred = sequential(*[ResBlock(in_c, in_c, bias=True, mode='C' + 'R' + 'C') for _ in range(nb)],
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(in_c, 256),
                                  nn.ReLU(),
                                  torch.nn.Linear(256, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 1),
                                  nn.Sigmoid()
                                  )

        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                   nn.ReLU(),
                                   # torch.nn.Linear(512, 512),
                                   # nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )
        self.to_gamma_4 = sequential(zero_module(torch.nn.Linear(512, nc[3])), nn.Sigmoid())
        self.to_beta_4 = sequential(zero_module(torch.nn.Linear(512, nc[3])), nn.Tanh())
        self.to_gamma_3 = sequential(zero_module(torch.nn.Linear(512, nc[2])), nn.Sigmoid())
        self.to_beta_3 = sequential(zero_module(torch.nn.Linear(512, nc[2])), nn.Tanh())
        self.to_gamma_2 = sequential(zero_module(torch.nn.Linear(512, nc[1])), nn.Sigmoid())
        self.to_beta_2 = sequential(zero_module(torch.nn.Linear(512, nc[1])), nn.Tanh())
        self.to_gamma_1 = sequential(zero_module(torch.nn.Linear(512, nc[0])), nn.Sigmoid())
        self.to_beta_1 = sequential(zero_module(torch.nn.Linear(512, nc[0])), nn.Tanh())