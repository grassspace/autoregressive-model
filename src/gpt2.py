import pytorch_lightning as pl
import torch
from torchvision import datasets
import torchvision.transforms as T
import os
import glob
from PIL import Image
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split, Dataset
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import math
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x):

        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device= x.device, dtype=x.dtype
        )
        print(x.device) # cuda:0
        print(x.dtype)  # float32
        print(attn_mask)
        print(attn_mask.shape) # torch.Size([8192, 8192])
        attn_mask = torch.triu(attn_mask, diagonal=1)
        print(attn_mask)
        print(attn_mask.shape) # torch.Size([8192, 8192])

        x = self.ln_1(x) # torch.Size([8192, 1, 512])
        print(x.shape)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False) # torch.Size([8192, 1, 512])
        print(a.shape) 
        x = x + a # torch.Size([8192, 1, 512])
        print(x.shape)
        m = self.mlp(self.ln_2(x)) # torch.Size([8192, 1, 512])
        print(m.shape)
        x = x + m # torch.Size([8192, 1, 512])
        print(x.shape)
        return x

class GPT2(nn.Module):
    def __init__(
        self, embed_dim, num_heads, num_layers, num_positions, num_vocab
    ):
        super(GPT2, self).__init__()

        self.embed_dim = embed_dim

        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim)) #torch.Size([512])
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)

    def forward(self, x):
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape # torch.Size([8192, 1])
        print(x.shape)

        h = self.token_embeddings(x) # torch.Size([8192, 1, 512])
        print(h.shape)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos # torch.Size([1, 1, 512]) SOSはパラメーター
        print(torch.ones(1, batch, self.embed_dim, device=x.device).shape) # torch.Size([1, 1, 512])
        print(self.sos.shape) # torch.Size([512])
        print(sos.shape)
        h = torch.cat([sos, h[:-1, :, :]], axis=0) # torch.Size([8192, 1, 512])
        print(h[:-1, :, :].shape) # torch.Size([8191, 1, 512])
        print(h.shape)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1) # torch.Size([8192, 1])
        print(positions.shape)
        h = h + self.position_embeddings(positions).expand_as(h) # torch.Size([8192, 1, 512]) expandiranakune
        print(self.position_embeddings(positions).shape) # torch.Size([8192, 1, 512])
        print(h.shape)
     
        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h) #torch.Size([8192, 1, 512])
        print(h.shape)

        logits = self.head(h) #torch.Size([8192, 1, 2])
        print(logits.shape)


            
        return logits

