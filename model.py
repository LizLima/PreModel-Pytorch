import torch
import torch.nn as nn
import torchvision.models as models 

class Generator(nn.Module):

    def __init__(self, num_class):
        super().__init__()
        self.dim = 256

        # Endoder
        self.encoder =  models.vgg19(pretrained=True)
        # change last layer input(4096) -> output (256)
        self.n_infeatures = self.encoder.classifier._modules['6'].in_features
        self.encoder.classifier._modules['6'] = nn.Linear(self.n_infeatures, self.dim)
        # Layer to classification
        self.fc = nn.Linear(self.dim, num_class)


        # Decoder based on DCGAN
        self.fc_dec = nn.Linear(in_features=self.dim, out_features=4*4*128*8)

        self.conv0  = nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False)
        self.bn0    = nn.BatchNorm2d(512)
       
        self.conv1  = nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False)
        self.bn1    = nn.BatchNorm2d(256)
        
        self.conv2  = nn.ConvTranspose2d( 256, 128, 4, 2, 1, bias=False)
        self.bn2    = nn.BatchNorm2d(128)
        
        self.conv3  = nn.ConvTranspose2d( 128, 64, 4, 2, 1, bias=False)
        self.bn3    = nn.BatchNorm2d(64)
     
        self.conv4  = nn.ConvTranspose2d( 64, 3, 4, 2, 1, bias=False)
        self.tan    = nn.Tanh()

        self.relu = nn.ReLU(True)

    def forward(self, input):
        
        encoder     = self.encoder(input) # latent space
        fc          = self.fc(encoder) #num classses
        # decoder
        fc_dec      = self.fc_dec(encoder)
        reshape_fc  = fc_dec.view(-1, 1024, 4, 4)

        dconv0      = self.relu(self.bn0(self.conv0(reshape_fc)))
        dconv1      = self.relu(self.bn1(self.conv1(dconv0)))
        dconv2      = self.relu(self.bn2(self.conv2(dconv1)))
        dconv3      = self.relu(self.bn3(self.conv3(dconv2)))
        dconv4      = self.tan(self.conv4(dconv3))
        return fc, dconv4

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)