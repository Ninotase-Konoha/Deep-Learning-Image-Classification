#!/usr/bin/env python3
"""
student.py

UNSW COMP9444 Neural Networks and Deep Learning

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
hw2main.py file unmodified, and you are only using the approved packages.

You have been given some default values for the variables train_val_split,
batch_size as well as the transform function.
You are encouraged to modify these to improve the performance of your model.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import instancenorm
from torch.nn.modules.linear import Linear
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

"""
   Answer to Question:

Briefly describe how your program works, and explain any design and training
decisions you made along the way.
"""

############################################################################
######     Specify transform(s) to be applied to the input images     ######
############################################################################
def transform(mode):
    """
    Called when loading the data. Visit this URL for more information:
    https://pytorch.org/vision/stable/transforms.html
    You may specify different transforms for training and testing
    """
    if mode == 'train':
        return transforms.ToTensor()
    elif mode == 'test':
        return transforms.ToTensor()


############################################################################
######   Define the Module to process the images and produce labels   ######
############################################################################




class Network(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=11, stride=3, padding=0)
        self.maxpooling1 = nn.MaxPool2d(kernel_size=3, stride=1)
        self.maxpooling2 = nn.MaxPool2d(kernel_size=5, stride=3)
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=7, stride=1, padding=2)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.linear1 = nn.Linear(in_features=3200, out_features=3200)
        self.linear2 = nn.Linear(in_features=3200, out_features=3200)
        self.linear3 = nn.Linear(in_features=3200, out_features=8)
        self.drop = nn.Dropout(p=0.5)


    def forward(self, input):

        # step 1~5
        y = self.maxpooling1(F.relu(self.conv1(input)))
        y = self.maxpooling1(F.relu(self.conv2(y)))
        y = F.relu(self.conv3(y))
        y = F.relu(self.conv4(y))
        y = self.maxpooling2(F.relu(self.conv5(y)))

        # step 6~8
        y = y.view(200, -1)
        y = self.drop(F.relu(self.linear1(y)))
        y = self.drop(F.relu(self.linear2(y)))
        y = self.linear3(y)

        return y

net = Network()
    
############################################################################
######      Specify the optimizer and loss function                   ######
############################################################################


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()
        self.lossF = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.lossF(output, target)

optimizer = optim.Adam(net.parameters(), lr=0.001)

loss_func = nn.CrossEntropyLoss()


############################################################################
######  Custom weight initialization and lr scheduling are optional   ######
############################################################################

# Normally, the default weight initialization and fixed learing rate
# should work fine. But, we have made it possible for you to define
# your own custom weight initialization and lr scheduler, if you wish.
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

    elif isinstance(m, Linear):
        nn.init.normal_(m.weight, mean=0.0, std= 1.0)
        nn.init.constant_(m.bias, 0)

    return m

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

############################################################################
#######              Metaparameters and training options              ######
############################################################################
dataset = "./data"
train_val_split = 0.8
batch_size = 200
epochs = 50
