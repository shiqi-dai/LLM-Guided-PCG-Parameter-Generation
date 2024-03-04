import yaml
import argparse
import os
from configs import Config
from utils import init_logging, make_logging_dir


# 读取配置文件信息
def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def parse_args(): #解析命令行参数
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--config', help='Path to the training config file.', required=True)
    parser.add_argument('--logdir', help='Dir for saving logs and models.')
    parser.add_argument('--checkpoint', default='', help='Checkpoint path')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_name', default='default', type=str)
    parser.add_argument('--wandb_id', type=str)
    
    args = parser.parse_args()
    return args
    
def main():
    args = parse_args()
    cfg = Config(args.config) # 待实现
    # Create log directory for storing training results.
    cfg.date_uid, cfg.logdir = init_logging(args.config, args.logdir)
    make_logging_dir(cfg.logdir)
    
    