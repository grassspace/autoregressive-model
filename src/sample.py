from image_gpt import ImageGPT
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
import data
from data import test_dl


def sample(model, context, length, num_samples=1, temperature=0.01):

    output = context.unsqueeze(-1).repeat_interleave(
        num_samples, dim=-1
    ).cuda()  # add batch so shape [seq len, batch] # torch.Size([4096, 1])
    print(output.shape)
    pad = torch.zeros(1, 1, dtype=torch.long).cuda()  # to pad prev output # torch.Size([1, 1])
    print(pad.shape)
    with torch.no_grad():
        for _ in tqdm(range(length), leave=False):
            logits = model(torch.cat((output, pad), dim=0)) # torch.Size([4097, 1, 2])
            print(logits.shape)
            logits = logits[-1, :, :] / temperature # torch.Size([1, 2])
            print(logits.shape)
            probs = F.softmax(logits, dim=-1) # torch.Size([1, 2])
            print(probs.shape)
            pred = torch.multinomial(probs, num_samples=1).transpose(1, 0)#ピクセルの予測 # torch.Size([1, 1])
            print(pred.shape)
            output = torch.cat((output, pred), dim=0)#つなげる # torch.Size([4097, 1])
            print(output.shape)
    return output 


def make_figure(rows):
    figure = np.stack(rows, axis=0)
    print(figure.shape) # (1, 3, 128, 64)
    rows, cols, h, w = figure.shape
    figure = figure.swapaxes(1, 2).reshape(h * rows, w * cols) # (128, 192)
    print(figure.shape)
    figure = (figure * 255).astype(np.uint8)
    print(figure.max()) # 255
    return Image.fromarray(np.squeeze(figure))

model = ImageGPT.load_from_checkpoint("/home/ryosuke/ImageGPT/logs_5/MyModel/105M/checkpoints/epoch=479-step=91200.ckpt").gpt.eval().cuda()

dl = iter(DataLoader(test_dl.dataset, shuffle=True))

# rows for figure
rows = []

for example in tqdm(range(1)):
    img = next(dl) #torch.Size([1, 1, 128, 64])
    print(img.shape)
    h, w = img.shape[-2:] #torch.Size([128, 64])
    print(img.shape[-2:])
    seq = img.reshape(-1) #torch.Size([8192])
    print(seq.shape) 

    # first half of image is context

    context = seq[: int((len(seq)) * 0.5) ] # torch.Size([4096])
    print(context.shape)
    context_img = np.pad(context, (0, int((len(seq)) * 0.5))).reshape(h, w) # (128, 64)
    print(context_img.shape)
    #context = torch.from_numpy(context).cuda()

    # predict second half of image
    preds = sample(model, context, int((len(seq)) * 0.5), num_samples=1).cpu().numpy().transpose() # (1, 8192)
    print(preds.shape)
    print(preds.max()) # 1
    preds = preds.reshape(-1, h, w) # (1, 128, 64)
    print(preds.shape)

    # combine context, preds, and truth for figure
    img = img.numpy()
    img = np.squeeze(img, 0) # (1, 128, 64)
    print(img.shape)
    context_img = np.expand_dims(context_img, 0) #(1, 128, 64)
    print(context_img.shape)
    print((np.concatenate([context_img, preds, img], axis=0)).shape) 
    rows.append(
         np.concatenate([context_img, preds, img], axis=0) # (3, 128, 64)
       )

figure = make_figure(rows)
print(figure.mode) # L
figure.save("/home/ryosuke/ImageGPT/105M/tooth26_105M.jpg")
