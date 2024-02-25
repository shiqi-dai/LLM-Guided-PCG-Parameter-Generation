import torch
import numpy as np
import visdom
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from dataset import PCGDataset
from model import Decoder
from utils import to_device


class Trainer(object):
    def __init__(self,
                 cfg, # Global configuration
                 model,
                 opt, # optimizer for the model
                 schedule, # scheduler for the model optimizer
                 train_dataloader,
                 val_dataloader):
        super(Trainer, self).__init__()
        
        # Initialize models and data loaders
        self.cfg = cfg
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.is_inference = train_dataloader is None
        self.opt = opt
        self.schedule = schedule
        
        # Initialize loss functions
        self.criteria = torch.nn.MSELoss()

        # Initialize logging attributes
        self.current_iteration = 0
        self.current_epoch = 0

        # 待写：初始化验证参数 
    
    def _start_of_epoch(self, current_epoch):
        torch.cuda.empty_cache() # Prevent the first iteration from running OOM.
    
    def _start_of_iteration(self, data, current_iteration):
        data = to_device(data, 'cuda')

        # 待写：是否需要采样辅助信息eg 相机角度、真实值

        return {**data}
    

    def train_one_loop(self, data):
        pass

    def load_checkpoint(self, cfg, checkpoint_path, resume=None, load_sch=True):
        pass

    def _get_visualizations(self, data):
        pass 