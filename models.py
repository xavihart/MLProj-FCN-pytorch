from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from torchvision.models.vgg import VGG
import torch.nn.functional as F
from torch.nn import init
import numpy as np

def get_upsample_weight(in_channels, out_channels, kernel_size):

    '''

    make a 2D bilinear kernel suitable for upsampling

    '''

    factor = (kernel_size + 1) // 2

    if kernel_size % 2 == 1:

        center = factor - 1

    else:

        center = factor - 0.5

    og = np.ogrid[:kernel_size, :kernel_size]  # list (64 x 1), (1 x 64)

    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)  # 64 x 64

    weight = np.zeros((in_channels, out_channels, kernel_size,

                       kernel_size), dtype=np.float64)

    weight[range(in_channels), range(out_channels), :, :] = filt



    return torch.from_numpy(weight).float()
model_dict = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


ranges = {
    'vgg11': ((0, 3), (3, 6),  (6, 11),  (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

def make_layers(model_dict, batch_norm=False):
    """

    :param model_dict: model_dict['vgg16']
    :param batch_norm:
    :return: numer means conv , M means pooling
    """
    layers = []
    in_channels = 3
    for v in model_dict:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGNET(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_param=False):
        super().__init__(make_layers(model_dict[model]))
        self.ranges = ranges[model]
        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)
            print("model prepared successfully--")
        if not requires_grad:
            for p in super().parameters():
                p.requires_grad = False
        if remove_fc:
            del self.classifier
        if show_param:
            for name, p in self.named_parameters():
                print(name, p.size())
    def forward(self, x):
        output = {}
        for idx in range(len(self.ranges)):
            for layer in range(self.ranges[idx][0], self.ranges[idx][1]):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x
        return output

class FCN8s(nn.Module):
    def __init__(self, pretrained_net, n_class):
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)
    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']
        #print(x5.shape)
        score = self.relu(self.deconv1(x5))   #[n, 512, x.h/16, x.w/16]
        #print(score.shape)
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))   # [, , /8, /8]
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))  #[, , /4, /4]
        score = self.bn4(self.relu(self.deconv4(score)))   #[, , /2, /2]
        score = self.bn5(self.relu(self.deconv5(score)))   #[n, 32, , w, h]
        score = self.classifier(score)          #[, classed, w, d]
        return score

    def _init_weights(self):

        '''

        hide method, used just in class

        '''

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                m.weight.data.zero_()

                # if m.bias is not None:

                m.bias.data.zero_()

            if isinstance(m, nn.ConvTranspose2d):
                assert m.kernel_size[0] == m.kernel_size[1]

                initial_weight = get_upsample_weight(m.in_channels,

                                                     m.out_channels, m.kernel_size[0])

                m.weight.data.copy_(initial_weight)

if __name__ == "__main__":
    # testing the model
    net = VGGNET(model='vgg16', requires_grad=True, show_param=True)
    FCN = FCN8s(pretrained_net=net, n_class=20)
    x = FCN(torch.rand(1, 3, 320, 320))
    print(x.shape)


