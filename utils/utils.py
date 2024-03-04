import torch 
import torch.nn.functional as F
import collections
import datetime
import os
string_classes = (str, bytes)

""" Device """
def to_device(data, device):
    """ Move all tensors inside data to device
    Args:
        data(dict, list, or tensor): Input data.
        device(str):'cpu' or 'cuda'.
    """
    assert device in ['cpu', 'cuda']
    if isinstance(data, torch.Tensor):
        data = data.to(torch.device(device))
        return data
    elif isinstance(data, collections.abc.Mapping):
        return {key: to_device(data[key], device) for key in data}
    elif isinstance(data, collections.abc.Sequence) and not isinstance(data, string_classes):
        return [to_device(d, device) for d in data]
    else:
        return data
def to_cuda(data):
    return to_device(data, 'cuda')
def to_cpu(data):
    return to_device(data, 'cpu')

""" Logging """
def get_date_uid():
    r"""Generate a unique id based on date.
    Returns:
        str: Return uid string, e.g. '2017_1122_1713_07'.
    """
    return str(datetime.datetime.now().strftime("%Y_%m%d_%H%M_%S"))

def init_logging(config_path, logdir):
    r"""Create log directory for storing checkpoints and output images.
    Args:
        config_path (str): Path to the configuration file.
        logdir (str): Log directory name
    Returns:
        str: Return log dir
    """
    config_file = os.path.basename(config_path) # 文件名.格式
    root_dir = 'logs'
    date_uid = get_date_uid()
    # example: 2019_0125_1047_58_spade_cocostuff
    log_file_dir = '_'.join([date_uid, os.path.splitext(config_file)[0]])
    if logdir is None:
        logdir = os.path.join(root_dir, log_file_dir)
    return date_uid, logdir

def make_logging_dir(logdir):
    r"""Create the logging directory

    Args:
        logdir (str): Log directory name
    """
    print('Make folder {}'.format(logdir))
    os.makedirs(logdir, exist_ok=True)
    
    global LOG_DIR
    LOG_DIR = logdir