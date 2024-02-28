# from transformers import CLIPProcessor, CLIPModel # huggingface的库需要支持pytorch 1.11+
import clip #
from PIL import Image
import torch
from typing import Any, Dict, List, Optional
from glob import glob
import os
import numpy as np
import yaml

def get_paths(dir, option: Optional[str] = None):
    if option == "image":
        all_paths = glob(os.path.join(dir, '*.jpg'))
    elif option == "text":
        all_paths = glob(os.path.join(dir, '*.txt'))
    else:
        print("Error in get_paths: Unsupported Type")
        
    return all_paths

def get_emb_new(img_dir, txt_dir): # 改为用https://github.com/openai/CLIP里的api
    # 加载CLIP processor和model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_processor = clip.load('ViT-B/32', device)
    
    # 图像处理
    imgemb = []
    if os.path.isdir(img_dir): #获取文件夹下所有图片路径
        img_paths = get_paths(img_dir, "image") 
    else: # 单张图片
        img_paths = [img_dir]
    for p in img_paths:
        img = Image.open(p)
        inputs =  clip_processor(img).unsqueeze(0).to(device)
        with torch.no_grad():
            img_features = clip_model.encode_image(inputs) #[1,512]
        imgemb.append(img_features)
        
    # 文本处理  
    txtemb = []
    if os.path.isdir(txt_dir): #获取文件夹下所有文本路径
        txt_paths = get_paths(txt_dir, "text") 
    else: # 单条文本
        txt_paths = [txt_dir]
    
    for p in txt_paths:
        txt = open(p).read()
        inputs = clip.tokenize(txt.split(",")).to(device) # 注意max context length = 77
        with torch.no_grad():
            txt_features = clip_model.encode_text(inputs) #[token数, 512]
        txtemb.append(txt_features) 
    
    return imgemb, txtemb
    
    
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

def main():
    """
    处理raw数据集为训练集, 如下格式:
    id: 0
    pcg_params:
        Floor_Amount:7
        Wide:1
        Long:4
        Use_Corners:false
        Front_Windows:false
        Back_Windows:true
        Left_Windows:false
        RightWindows:true
    BackCam:
        SceneColorHDR_img:
        [ ... ]
        SceneColorHDR_txt:
        [ ... ]
    ......
  
    """ 
    
    input_dir = "rawdata/0-49"
    id_dir = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))] # ['0', '1', '2', '3'...]
    
    for id in id_dir:
        # 写进yaml格式
        data = dict()
        data['id'] = int(id)
        
        # pcg_params:
        pcg_param_path = os.path.join(input_dir, id, 'pcg_parameters.txt')
        pcg_params = open(pcg_param_path).read().split(',')
        pdata = dict()
        for param in pcg_params:
            key, value = param.split(':')
            if value.lower() == "true":
                value = True
            elif value.lower() == "false":
                value = False
            else:
                value = int(value)
            pdata[key] = value
        data['pcg_params'] = pdata
       
        # Camera params
        cam_dir = [p for p in os.listdir(os.path.join(input_dir, id)) if os.path.isdir(os.path.join(input_dir, id, p))]
        for cam in cam_dir:
            img_path = os.path.join(input_dir, id, cam, 'SceneColorHDR.jpg')
            txt_path = os.path.join(input_dir, id, cam, 'SceneColorHDR.txt')
            imgemb, txtemb = get_emb_new(img_path, txt_path)          
            data[cam] = {'SceneColorHDR_img':str(imgemb[0].cpu().numpy().tolist()),
                         'SceneColorHDR_txt':str(txtemb[0].cpu().numpy().tolist())}
            
        save_path = os.path.join('data', id + '.yaml')
        with open(save_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        
        print("done and saved in ", save_path)
if __name__ == '__main__':
    main()
    