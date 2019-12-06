import models
import torch
import torchvision as tv
import voc_base
import os

def main():
    data_pth = os.path.join(os.getcwd(), '..', '..', 'zhanghao/interpretable_method_eval/data')
    train_data = voc_base.VOC2012ClassSeg(root=data_pth, split='train', transform=True)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=1,  shuffle=True, num_workers=5)
    vgg_model = models.VGGNET(requires_grad=False)
    fcn_net = models.FCN8s(pretrained_net=vgg_model, num_class=21)


