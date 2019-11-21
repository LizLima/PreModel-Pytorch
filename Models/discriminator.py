import torch
import torch.nn as nn 

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.bias = False
        self.conv0 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn0 = nn.BatchNorm2d(16)

        self.conv1 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn4 = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, bias = self.bias)
        self.bn5 = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=2, padding=1, bias = self.bias)
 
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, input):

        conv0 =  self.lrelu(self.bn0(self.conv0(input)))
        conv1 =  self.lrelu(self.bn1(self.conv1(conv0)))
        conv2 =  self.lrelu(self.bn2(self.conv2(conv1)))
        conv3 =  self.lrelu(self.bn3(self.conv3(conv2)))
        conv4 =  self.lrelu(self.bn4(self.conv4(conv3)))
        conv5 =  self.lrelu(self.bn5(self.conv5(conv4)))
        conv6 = self.sigmoid(self.conv6(conv5))

        return conv6