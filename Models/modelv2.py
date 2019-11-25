import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.dim = 256

        # Endoder
        self.encoder =  models.vgg19_bn(pretrained=True)
         # change last layer input(4096) -> output (256)
        # VGG
        self.n_infeatures = self.encoder.classifier._modules['6'].in_features
        self.encoder.classifier._modules['6'] = nn.Linear(self.n_infeatures, self.dim)
        # self.fc = nn.Linear(self.dim, num_class)

        # RESNET50
        # self.encoder =  models.resnet50(pretrained=True)
        # num_ftrs = self.encoder.fc.in_features
        # self.encoder.fc = nn.Linear(num_ftrs, self.dim)

    def forward(self, input):
        out_space   = self.encoder(input) # latent space

        return out_space

class Decoder(nn.Module):

    def __init__(self,):
        super().__init__()
        self.dim = 256

        # Decoder based on DCGAN
        # self.fc_dec = nn.Linear(in_features=self.dim, out_features=4*4*128*8)

        self.conv0 = nn.ConvTranspose2d( self.dim, 128, 4, 1, 0, bias=False)
        self.bndc0 = nn.BatchNorm2d(128)

        self.conv1 = nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False)
        self.bndc1 = nn.BatchNorm2d(64)

        self.conv2 = nn.ConvTranspose2d( 64, 32, 4, 2, 1, bias=False)
        self.bndc2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.ConvTranspose2d( 32, 16, 4, 2, 1, bias=False)
        self.bndc3 = nn.BatchNorm2d(16)

        self.conv4 = nn.ConvTranspose2d( 16, 8, 4, 2, 1, bias=False)
        self.bndc4 = nn.BatchNorm2d(8)
        
        self.conv5 = nn.ConvTranspose2d( 8, 3, 4, 2, 1, bias=False)
        self.tan   = nn.Tanh()

        self.relu = nn.ReLU(True)

    def forward(self, input):
 
        resize_enc  = input.view(-1, input.size(1), 1, 1)
        dconv0      = self.relu(self.bndc0(self.conv0(resize_enc)))
        dconv1      = self.relu(self.bndc1(self.conv1(dconv0)))
        dconv2      = self.relu(self.bndc2(self.conv2(dconv1)))
        dconv3      = self.relu(self.bndc3(self.conv3(dconv2)))
        dconv4      = self.relu(self.bndc4(self.conv4(dconv3)))
        dconv5      = self.tan(self.conv5(dconv4))

        return dconv5
        
class Generator(nn.Module):

    def __init__(self, num_class):
        super().__init__()

        # Endoder
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.decoder.apply(weights_init)

    def forward(self, input):
        
        enc_output = self.encoder(input)
        dec_output = self.decoder(enc_output)

        return dec_output


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


