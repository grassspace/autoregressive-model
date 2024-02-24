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
from image_gpt import ImageGPT
import data
from data import train_dl


model = ImageGPT()

logger = pl_loggers.TensorBoardLogger("logs_5", name = "MyModel")

# pretraining

checkpoint = pl.callbacks.ModelCheckpoint(monitor = 'train_loss',
                                          verbose = 1,
                                          mode = "min")
#early_stopping = pl.callbacks.early_stopping.EarlyStopping(monitor = "val_avg_loss", mode = "min")
trainer = pl.Trainer(
            max_epochs= 500,
            accelerator='gpu', 
            devices=1,
            precision= 16,
            #accumulate_grad_batches= 16,
            callbacks= checkpoint, #early_stopping],
            logger=logger
        )

trainer.fit(model, train_dl)


