import torch
import numpy as np
import visdom
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset import PCGDataset
from model import Decoder


class Trainer(object):
    def __init__(self,
                 config,
                 net,
                 opt,
                 schedule,
                 train_data_loader,
                 val_data_loader):
        super(Trainer, self).__init__()
        

    