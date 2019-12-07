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
import time

data_pth = os.path.join(os.getcwd(), '..', '..', 'zhanghao/interpretable_method_eval/data')
val_data = voc_base.VOC2012ClassSeg(root=data_pth, split='train', transform=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=1, shuffle=False, num_workers=5)
vgg_model = models.VGGNET(requires_grad=True)
net = models.FCN8s(pretrained_net=vgg_model, n_class=21)
net.load_state_dict(torch.load('./result_new/pretrained_model_epoch290.pth'))
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = 0
print("model loaded successfully" + "--" * 5)


# pixel accuracy -->  right markdown / pixel_numbers
def PA(gt, pred):
    """

    :param gt: the ground truth : 320 * 320
    :param pred:  320 *320
    :return: double acc
    """
    assert len(gt.shape) == len(pred.shape)
    assert gt.shape[0] == pred.shape[0]
    acc = (gt == pred).sum().double() / (pred.shape[0] * pred.shape[1])
    print(acc)
    return acc

# cal the acc of each class and then cal the mean of them

def mPA(gt, pred):
    acc_list = []
    for i in range(21):
        if (gt == i).sum() == 0:
            continue
        acc = (pred[gt == i] == i).sum().double() / (gt == i).sum()
        acc_list.append(acc)
    acc = torch.tensor(acc_list)
    acc = acc.float()
    print(acc)
    return acc.mean()


def MIoU(gt, pred):
    acc_list = []
    for i in range(21):
        if (gt == i).sum() == 0:
            continue
        SumGT = (gt == i).sum()
        SumPred = (pred == i).sum()
        intersection = (pred[gt == i] == i).sum().double()
        if (SumGT + SumPred - intersection) == 0:
            continue
        IoU = intersection / (SumGT + SumPred - intersection)
        acc_list.append(IoU)
    acc = torch.tensor(acc_list)
    acc = acc.float()
    print("iou", acc)
    return acc.mean()


image_saving_pth = "./result_new/image_exp_new"


if __name__ == "__main__":
    net.eval()
    net = net.to(device)
    image_tot = 0
    global_PA_list = []
    global_mPA_list = []
    global_mIoU_list = []
    for index, (img, lbl) in enumerate(val_loader):
        save_path = os.path.join(image_saving_pth, "Testing_image(train)%d" %(image_tot))
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        image_tot += 1
        img = img.to(device)
        lbl = lbl.to(device)
        assert len(img.shape) == 4 and img.shape[0] == 1

        out = net(img)
        pred = out.data.max(1)[1].squeeze_(0)  # 320 * 320
        lbl = lbl[0]

        PA_acc = PA(lbl, pred)
        mPA_acc = mPA(lbl, pred)
        mIoU_acc = MIoU(lbl, pred)
        global_PA_list.append(PA_acc)
        global_mPA_list.append(mPA_acc)
        global_mIoU_list.append(mIoU_acc)

        tool_lib.labelTopng(pred.cpu(), os.path.join(save_path, 'seg_pred.png'))
        torchvision.utils.save_image(img[0], os.path.join(save_path , 'img_raw.jpg'), normalize=True)
        tool_lib.labelTopng(lbl.cpu(), os.path.join(save_path, 'seg_grtruth.png'))

        f = open(os.path.join(save_path, 'acc.txt'), 'w')
        f.write("PA:{}\nmPA:{}\nmIoU:{}\n".format(PA_acc.data, mPA_acc.data, mIoU_acc.data))
        f.close()
        print("Image [{}/{}] has been calculated------".format(index+1, len(val_loader)))

    mean1 = torch.tensor(global_PA_list).mean()
    mean2 = torch.tensor(global_mPA_list).mean()
    mean3 = torch.tensor(global_mIoU_list).mean()

    print(mean1, mean2, mean3)
    print("finished---------------")















