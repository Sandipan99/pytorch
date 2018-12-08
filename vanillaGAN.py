# implements vanilla generative adversarial network (GAN) as proposed by Goodfellow et. al.
__author__ = "Sandipan Sikdar"

from io import open
import random
import torch
import torch.nn as nn
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch import optim
import torch.nn.functional as F


def mnist_data():
    compose = transforms.Compose([transforms.ToTensor(),transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)


#class Generator(nn):
#    def __init__(self,input_size):
#        super(Generator,self).__init__()
#        self.input_size = input_size
data = mnist_data()

print(data[0][0].size())
