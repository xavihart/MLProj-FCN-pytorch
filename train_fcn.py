import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data.dataloader
import numpy as np
import torchvision
import models
import voc_base
from torch.optim import Adam, SGD
from argparse import ArgumentParser
import tool_lib


parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=2, help="batch size of the data")
parser.add_argument('-e', '--epochs', type=int, default=300, help='epoch of the train')
parser.add_argument('-c', '--n_class', type=int, default=21, help='the classes of the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
args = parser.parse_args()


batch_size = args.batch_size
learning_rate = args.learning_rate
epoch_num = args.epochs
n_class = args.n_class

best_test_loss = np.inf

data_pth = os.path.join(os.getcwd(), '..', 'zhanghao/interpretable_method_eval')
print(data_pth)













