import torch
from torch import nn

class Decoder(nn.Module):
	"""
	输入: image或者text的embedding
	输出: 预测的pcg参数数组
	input_dim: clip特征向量长度512
	ouput_dim: pcg参数数组长度(待确认)
	"""
	def __init__(self, input_dim, output_dim): 
		super(Decoder, self).__init__()
		d = 512
		self.net = nn.Sequential(
			nn.Linear(input_dim, d), # 512, 512
			nn.ReLU(True),
			nn.Dropout(p=0.20),
			nn.Linear(d, d/2), # 512, 256
			nn.ReLU(True),
			nn.Dropout(p=0.20),
			nn.Linear(d/2, d/4), # 256, 128
			nn.ReLU(True),
			nn.Dropout(p=0.20),
			nn.Linear(d/4, d/8), # 128, 64
			nn.ReLU(True),
			nn.Linear(d/8, output_dim),
			nn.Sigmoid() #将结果压缩至0-1
		)
	def foward(self, x):
		return self.net(x).reshape(-1)