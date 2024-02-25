import torch 
import torch.nn.functional as F
import collections
string_classes = (str, bytes)

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

