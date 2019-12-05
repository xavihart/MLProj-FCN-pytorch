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
import torch.nn as nn
import torch.nn.functional as F
import  matplotlib.pylab as plt

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)
    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)


parser = ArgumentParser()
parser.add_argument('-bs', '--batch_size', type=int, default=2, help="batch size of the data")
parser.add_argument('-e', '--epochs', type=int, default=300, help='epoch of the train')
parser.add_argument('-c', '--n_class', type=int, default=21, help='the classes of the dataset')
parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('-d', '--device', type=str, default=2, help='setting cuda devices')
args = parser.parse_args()


batch_size = args.batch_size
learning_rate = args.learning_rate
epoch_num = args.epochs
n_class = args.n_class
device = args.device

best_test_loss = np.inf

data_pth = os.path.join(os.getcwd(), '..', '..', 'zhanghao/interpretable_method_eval/data')
print(data_pth)

train_data = voc_base.VOC2012ClassSeg(root=data_pth, split='train', transform=True)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=5)
val_data = voc_base.VOC2012ClassSeg(root=data_pth, split='val', transform=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=5)


vgg_model = models.VGGNET(requires_grad=True)
fcn_net = models.FCN8s(pretrained_net=vgg_model, n_class=n_class)

criterion = CrossEntropyLoss2d()
optimizer = Adam(fcn_net.parameters())
epoch_loss_list = []

def train(epoch):
    fcn_net.to(device)
    fcn_net.train()
    tot_loss = 0.0
    for batch_index, (img, lbl) in enumerate(train_loader):
        N = img.size(0)  # numbers of img in a batch
        assert N == batch_size
        img = Variable(img)
        lbl = Variable(lbl)
        img = img.to(device)
        lbl = lbl.to(device)
        out = fcn_net(img)
        loss = criterion(out, lbl)
        loss /= N
        #print("test1", tot_loss / len(train_loader))                        #flag1
        tot_loss += loss.data
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_index % 3 == 0:
            print("epoch:%d " % (epoch), "batch:%d " % (batch_index), "avg_loss:{}".format(tot_loss / (batch_index + 1)))
    assert tot_loss is not np.inf
    assert tot_loss is not np.nan
    epoch_loss_list.append(tot_loss.data / len(train_loader))
    if epoch % 10 == 0:                                                           #flag2
        torch.save(fcn_net.state_dict(), './result/pretrained_model_epoch%d.pth' % (epoch))
        print("saved successfully" + "--" * 5)


if __name__ == "__main__":
    for epoch in range(epoch_num):
        train(epoch)
        if epoch == 20:
            learning_rate *= 0.01
            optimizer.param_groups[0]['lr'] = learning_rate

    plt.plot(epoch_loss_list)
    plt.xlabel("epoch")
    plt.ylabel("epoch avg loss")
    plt.title("training loss curve")
    plt.savefig('./result/curve.jpg')

















