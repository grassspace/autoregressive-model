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
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import loggers as pl_loggers
from tqdm import tqdm
import matplotlib.pyplot as plt
from gpt2 import GPT2

def learning_rate_schedule(warmup_steps, total_steps):
    """Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps"""

    def learning_rate_fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        else:
            progress = float(step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return 0.5 * (1.0 + math.cos(math.pi * progress))

    return learning_rate_fn

def _to_sequence(x):
    """shape batch of images for input into GPT2 model"""
    x = x.view(x.shape[0], -1)  # flatten images into sequences
    x = x.transpose(0, 1).contiguous()  # to shape [seq len, batch]
    return x

class ImageGPT(pl.LightningModule):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        num_layers=32,
        num_vocab=2,
        learning_rate=0.001,
        steps=95000,
        warmup_steps=9500,
        **kwargs,
    ):
        super(ImageGPT, self).__init__()
        self.save_hyperparameters()
        self.gpt = GPT2(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            num_positions=128 * 64,
            num_vocab=num_vocab
        )



        self.criterion = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.steps = steps
        self.warmup_steps = warmup_steps



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.gpt.parameters(), lr=self.learning_rate)

        scheduler = LambdaLR(optimizer, learning_rate_schedule(self.warmup_steps, self.steps))
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.gpt(x)

    def training_step(self, batch, batch_idx):

        x = batch
        x = _to_sequence(x)
        

        logits = self.gpt(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))
        print(logits.view(-1, logits.size(-1)).shape) # torch.Size([8192, 1, 2]) device=cuda:0 dtype=torch.float16
        print(x.view(-1).shape) # torch.Size([8192]) device=cuda:0
        print(logits.view(-1, logits.size(-1)))
        print(x.view(-1))
        self.log(
            'train_loss',
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True

        )



        return {"loss": loss}
    
    
"""
    def validation_step(self, batch, batch_idx):
        x = batch
        x = _to_sequence(x)

        logits = self.gpt(x)
        loss = self.criterion(logits.view(-1, logits.size(-1)), x.view(-1))


        return {"val_loss": loss}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([x["val_loss"] for x in outs]).mean()
        self.log(
            'val_avg_loss',
            avg_loss,
            prog_bar=True,
            logger=True,
            on_epoch=True
        )



        return {"val_loss": avg_loss}

"""
