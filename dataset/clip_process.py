from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch
from typing import Any, Dict, List, Optional
from glob import glob
import os
import numpy as np

def get_paths(dir, option: Optional[str] = None):
    if option == "image":
        all_paths = glob(os.path.join(dir, '*.jpg'))
    elif option == "text":
        all_paths = glob(os.path.join(dir, '*.txt'))
    else:
        print("Error in get_paths: Unsupported Type")
        
    return all_paths

def get_emb(img_dir, txt_dir):
    # 加载CLIP processor和model
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  

    # 图像处理
    imgemb = []
    img_paths = get_paths(img_dir, "image")
    for p in img_paths:
        img = Image.open(p)
        inputs =  clip_processor(images=img, return_tensors="pt")  
        with torch.no_grad():
            img_features = clip_model(**inputs).last_hidden_state[:, 0, :]
        imgemb.append(img_features)
        # save
        fname, fext = os.path.splitext(p) #分离文件名和后缀
        np.save(fname + '.npy', np.array(img_features))

    # 文本处理  
    txtemb = []
    txt_paths = get_paths(txt_dir, "text")
    for p in txt_paths:
        txt = open(p)
        inputs = clip_processor(text=txt, return_tensors="pt")
        with torch.no_grad():
            txt_features = clip_model(**inputs).last_hidden_state[:, 0, :]
        txtemb.append(txt_features)
        # save
        fname, fext = os.path.splitext(p) #分离文件名和后缀
        np.save(fname + '.npy', np.array(txt_features))