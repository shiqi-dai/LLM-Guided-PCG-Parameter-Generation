import torch
import random
import yaml
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional
import os
from glob import glob
import numpy as np
import json


class PCGDataset(Dataset):
    def __init__(self, 
                 imgemb_dir: Optional[str] = None, 
                 txtemb_dir: Optional[str] = None,
                 pcg_dir: Optional[str] = None):
        super(PCGDataset, self).__init__()

        # 得到由clip处理后图像，文本向量
        self.imgemb_paths = self.get_all_paths(imgemb_dir)
        self.txtemb_paths = self.get_all_paths(txtemb_dir)
        self.pcg_paths = self.get_all_paths(pcg_dir)

        self.imgemb = []
        self.txtemb = []
        self.pcg = [] 
        self.pcg_id2pn = dict() # pcg索引-参数名查找表 
        for p in self.imgemb_paths:
            self.imgemb.append(torch.tensor(np.load(p)))
        for p in self.txtemb_paths:
            self.txtemb.append(torch.tensor(np.load(p)))
        for p in self.pcg_paths:
            self.pcg.append(torch.tensor(np.load(p)))
        with open(os.path.join(pcg_dir, 'id2pn.json'), "r") as f:
            self.pcg_id2pn = json.load(f)

    def __len__(self):
        return len(self.imgemb_paths)
    
    def __getitem__(self, idx):
        # Q：是否需要加噪声？看数据集大小。未加噪版如下：
        idx = random.randint(0, self.__len__ - 1)
        data_pair = [self.imgemb[idx].cuda(), self.txtemb[idx].cuda()]
        return data_pair, self.pcg[idx]


    def get_all_paths(self, dir):
        all_paths = glob(os.path.join(dir, '*.npy'))
        return all_paths
    
