# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule, Lines
from . import s16netconfig as netconfig

cfg = netconfig.cfg


class CnnBlock(BasicModule):
    
    def __init__(self, cfg):
        super(CnnBlock, self).__init__()
        self.cfg = cfg
        self.conv1 = nn.Conv1d(cfg.in_channel, cfg.out_channel, (1, cfg.emb_dim), stride=1, padding=(0, 0))
        self.conv2 = nn.Conv1d(cfg.in_channel, cfg.out_channel, (2, cfg.emb_dim), stride=1, padding=(1, 0))
        self.conv3 = nn.Conv1d(cfg.in_channel, cfg.out_channel, (3, cfg.emb_dim), stride=1, padding=(1, 0))
        # self.conv4 = nn.Conv1d(cfg.in_channel, cfg.out_channel, (4, cfg.emb_dim), stride=1, padding=(2, 0))
    
    def forward(self, x):  # (batch, in_channel, sent_length, emb_dim)
        x1 = F.relu(self.conv1(x))  # (batch, out_channel, sent_length, 1)
        x2 = F.relu(self.conv2(x))  # (batch, out_channel, sent_length + 1, 1)
        x3 = F.relu(self.conv3(x))  # (batch, out_channel, sent_length, 1)
        # x4 = F.relu(self.conv4(x))  # (batch, out_channel, sent_length + 1, 1)
        x2 = x2[:, :, :-1, :]  # (batch, out_channel, sent_length, 1)
        # x4 = x4[:, :, :-1, :]  # (batch, out_channel, sent_length, 1)
        x = torch.cat((x1, x2, x3), dim=1)  # (batch, out_channel * 3, sent_length, 1)
        return x


class TextCnn(BasicModule):
    
    def __init__(self, cfg):
        super(TextCnn, self).__init__(cfg.save_dir)
        self.cfg = cfg
        cfg1 = netconfig.DefaultConfig(in_channel=1, out_channel=cfg.tcnn_channel)
        self.cnnblock1 = CnnBlock(cfg1)
        cfgs = netconfig.DefaultConfig(
            in_channel=cfg.tcnn_channel * 3, out_channel=cfg.tcnn_channel, emb_dim=1)
        self.cnnblocks = torch.nn.ModuleList()
        for _ in range(cfg.tcnn_block_num):
            self.cnnblocks.append(CnnBlock(cfgs))
        start = cfg.tcnn_channel * 3 * cfg.sent_length
        print('[25] linear input dim = ', start)
        self.lines = Lines([start] + cfg.tcnn_lines_nodes + [cfg.tcnn_label_num])
    
    def forward(self, x):  # (batch, sent_length, emb_dim)
        x = x.unsqueeze(1)  # (batch, 1, sent_length, emb_dim)
        x = self.cnnblock1(x)  # (batch, tcnn_channel * 3, sent_length, 1)
        for b in self.cnnblocks:
            x = b(x)  # (batch, tcnn_channel * 3, sent_length, 1)
        x = x.view(x.size(0), -1)  # (batch, tcnn_channel * 3 * sent_length)
        x = self.lines(x)  # (batch, tcnn_label_num)
        x = F.softmax(x, dim=1)  # (batch, tcnn_label_num)
        return x




class CnnSim(BasicModule):
    
    def __init__(self, cfg):
        super(CnnSim, self).__init__(cfg.save_dir)
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.tcnn = TextCnn(cfg)
        start = cfg.tcnn_label_num * 2
        self.lines = Lines([start] + cfg.cnnsim_nodes + [2])
        self.load()
    
    def forward(self, x1, x2):  # x1和x2的size = (batch, sent_length)
        show = False
        x1 = self.emb(x1)  # (batch, sent_length, emb_dim)
        x1 = self.tcnn(x1)  # (batch, tcnn_label_num)
        
        x2 = self.emb(x2)  # (batch, sent_length, emb_dim)
        x2 = self.tcnn(x2)  # (batch, tcnn_label_num)
        
        x = torch.cat((x1, x2), dim=1)  # (batch, tcnn_label_num * 2)
        x = self.lines(x)  # (batch, 2)
        x = F.softmax(x, dim=1)  # (batch, 2)
        return x
    
    def predict(self, x1, x2):
        y = self(x1, x2)  # (1, 2)
        y = y.topk(1)[1].squeeze(1).tolist()  # (1)
        return y


model = CnnSim(cfg)


def test():
 
    print('running the Simnet...')
    simnet = CnnSim(cfg)
    x1 = torch.autograd.Variable(torch.arange(0, 14).view(2, cfg.sent_length)).long()
    x2 = torch.autograd.Variable(torch.arange(0, 14).view(2, cfg.sent_length)).long()
    o = simnet(x1, x2)
    print(o.size())
    print(o.data)
