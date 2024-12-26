import torch
from torch import nn
from models.network_fbcnn import QFAttention, ResBlock, sequential


class QFNet(nn.Module):
    def __init__(self):
        super(QFNet, self).__init__()
        self.fuses = nn.ModuleList([])
        nb = 4
        nc = [1280, 1280, 640, 320]
        for i in range(4):
            self.fuses.append(nn.ModuleList([*[QFAttention(nc[i], nc[i], bias=True, mode='C' + 'R' + 'C') for _ in range(nb)]]))

        self.qf_pred = sequential(*[ResBlock(320, 320, bias=True, mode='C' + 'R' + 'C') for _ in range(nb)],
                                  torch.nn.AdaptiveAvgPool2d((1, 1)),
                                  torch.nn.Flatten(),
                                  torch.nn.Linear(320, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 512),
                                  nn.ReLU(),
                                  torch.nn.Linear(512, 1),
                                  nn.Sigmoid()
                                  )

        self.qf_embed = sequential(torch.nn.Linear(1, 512),
                                   nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU(),
                                   torch.nn.Linear(512, 512),
                                   nn.ReLU()
                                   )

        self.to_gamma_4 = sequential(torch.nn.Linear(512, nc[3]), nn.Sigmoid())
        self.to_beta_4 = sequential(torch.nn.Linear(512, nc[3]), nn.Tanh())
        self.to_gamma_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Sigmoid())
        self.to_beta_3 = sequential(torch.nn.Linear(512, nc[2]), nn.Tanh())
        self.to_gamma_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Sigmoid())
        self.to_beta_2 = sequential(torch.nn.Linear(512, nc[1]), nn.Tanh())
        self.to_gamma_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Sigmoid())
        self.to_beta_1 = sequential(torch.nn.Linear(512, nc[0]), nn.Tanh())