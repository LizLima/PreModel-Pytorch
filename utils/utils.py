import torch.nn as nn
import torch

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def KLdivergence(mean, stddev):
    kl = 0.5 * torch.sum(torch.exp(stddev) + mean**2 - 1.0 - stddev)

    return kl

def heapmap(image):
    return heapmap

