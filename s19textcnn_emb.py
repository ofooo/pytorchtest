# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from .BasicModule import BasicModule, Lines
from . import s19cfg as config
from .Emb import emb, vocab

kernel_sizes = [1, 2, 3, 4]
kernel_sizes2 = [1, 2, 3, 4]


class TextCnn(BasicModule):
    
    def __init__(self, cfg):
        super(TextCnn, self).__init__(cfg.save_dir)
        self.cfg = cfg
        self.inputcount = 0
        self.encoder = emb.emb
        
        x1_convs = [nn.Sequential(
            nn.Conv1d(in_channels=cfg.emb_dim,
                      out_channels=cfg.x1_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(cfg.x1_dim),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(in_channels=cfg.x1_dim,
                      out_channels=cfg.x1_dim,
                      kernel_size=kernel_size),
            nn.BatchNorm1d(cfg.x1_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=(cfg.sent_length - kernel_size * 2 + 2))
        )
            for kernel_size in kernel_sizes]
        
        self.x1_convs = nn.ModuleList(x1_convs)
        
        self.fc = nn.Sequential(
            nn.Linear(len(kernel_sizes) * (cfg.x1_dim + cfg.x1_dim), cfg.linear_hidden_size),
            nn.BatchNorm1d(cfg.linear_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.linear_hidden_size, cfg.num_classes)
        )
        self.load()
        print(self.parameters)
    
    def forward(self, x1, x2):  # x1和x2的size = (batch, sent_length)
        self.inputcount += 1
        if self.inputcount < 3:
            print('INFO-49 size={} x1={}'.format(x1.size(), x1))
            print('INFO-50 size={} x2={}'.format(x2.size(), x2))
        x1 = self.encoder(x1)
        x2 = self.encoder(x2)
        # if self.cfg.static:
        x1.detach()
        x2.detach()
        
        x1_out = [x1_conv(x1.permute(0, 2, 1)) for x1_conv in self.x1_convs]
        x2_out = [x1_conv(x2.permute(0, 2, 1)) for x1_conv in self.x1_convs]
        conv_out = torch.cat((x1_out + x2_out), dim=1)
        reshaped = conv_out.view(conv_out.size(0), -1)
        logits = self.fc((reshaped))
        return logits
    
    def predict(self, x1, x2):
        y = self(x1, x2)  # (batch, 2)
        y = y.topk(1)[1].squeeze(1).tolist()  # (batch)
        return y


model = TextCnn(config.cfg)
