# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule, Lines
from . import s17fnncfg as netconfig

cfg = netconfig.cfg


class Fnn(BasicModule):
    
    def __init__(self, cfg):
        super(Fnn, self).__init__(cfg.save_dir)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        start = cfg.emb_dim * cfg.sent_length * 2
        self.lines = Lines([start] + cfg.fnn_nodes + [2])
        self.load()
    
    def forward(self, x1, x2):  # x1和x2的size = (batch, sent_length)
        x = torch.cat((x1, x2), dim=1)  # (batch, sent_length * 2)
        x = self.emb(x)  # (batch, sent_length * 2, emb_dim)
        x = x.view(x.size(0), -1)  # (batch, sent_length * 2 * emb_dim)
        x = self.lines(x)  # (batch, tcnn_label_num)
        x = F.softmax(x, dim=1)  # (batch, 2)
        return x
    
    def predict(self, x1, x2):
        y = self(x1, x2)  # (1, 2)
        y = y.topk(1)[1].squeeze(1).tolist()  # (1)
        return y


model = Fnn(cfg)


def test():
    print('running the Simnet...')
    simnet = CnnSim(cfg)
    x1 = torch.autograd.Variable(torch.arange(0, 14).view(2, cfg.sent_length)).long()
    x2 = torch.autograd.Variable(torch.arange(0, 14).view(2, cfg.sent_length)).long()
    o = simnet(x1, x2)
    print(o.size())
    print(o.data)
