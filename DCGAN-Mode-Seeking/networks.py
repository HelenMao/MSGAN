import  torch
import torch.nn as nn

####################################################################
#------------------------- Generator -------------------------------
####################################################################
class generator(nn.Module):
    # initializers
    def __init__(self, opts, d=128):
        super(generator, self).__init__()

        self.deconv1 = nn.ConvTranspose2d(opts.nz+ opts.class_num, d*4, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d * 4)
        self.relu1 = nn.ReLU()
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.relu2 = nn.ReLU()
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn =nn.BatchNorm2d(d)
        self.relu3 = nn.ReLU()
        self.deconv4 = nn.ConvTranspose2d(d, 3, 4, 2, 1)
        self.tanh = nn.Tanh()

    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, input, label):
        x = torch.cat([input.unsqueeze(2).unsqueeze(3), label], 1)
        x = self.relu1(self.deconv1_bn(self.deconv1(x)))
        x = self.relu2(self.deconv2_bn(self.deconv2(x)))
        x = self.relu3(self.deconv3_bn(self.deconv3(x)))
        x = self.tanh(self.deconv4(x))
        return x

####################################################################
#------------------------- Discriminators --------------------------
####################################################################
class discriminator(nn.Module):
    # initializers
    def __init__(self, opts, d=128):
        super(discriminator, self).__init__()
        self.conv1= nn.Conv2d((3 + opts.class_num), d, 4, 2, 1)
        self.lrelu1 = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.lrelu2 = nn.LeakyReLU(0.2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.lrelu3 = nn.LeakyReLU(0.2)
        self.conv4 = nn.Conv2d(d*4, 1, 4, 1, 0)
        self.sigmoid = nn.Sigmoid()


    # weight_init
    def weight_init(self):
        for m in self._modules:
            gaussian_weights_init(self._modules[m])

    # forward method
    def forward(self, input, label):
        label = label.expand(label.shape[0], label.shape[1], input.shape[2], input.shape[3])
        x = torch.cat([input, label], 1)
        x = self.lrelu1(self.conv1(x))
        x = self.lrelu2(self.conv2_bn(self.conv2(x)))
        x = self.lrelu3(self.conv3_bn(self.conv3(x)))
        x = self.sigmoid(self.conv4(x))
        return x

####################################################################
#------------------------- Basic Functions -------------------------
####################################################################
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 and classname.find('Conv') == 0:
        m.weight.data.normal_(0.0, 0.02)
