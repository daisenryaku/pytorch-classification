import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from .BasicModule import BasicModule

class AlexNet(BasicModule):
    def __init__(self, in_channels=3, img_rows=224, num_classes=10):
        super(AlexNet, self).__init__()
        self.model_name = 'AlexNet'
        self.out_rows1 = (img_rows-7)/4+1
        self.out_rows2 = (self.out_rows1-3)/2+1
        self.out_rows3 = (self.out_rows2-3)/2+1
        self.out_rows  = (self.out_rows3-3)/2+1
        self.features = nn.Sequential(
                nn.Conv2d(in_channels, 96, 11, 4, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                
                nn.Conv2d(96, 256, 5, 1, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),

                nn.Conv2d(256, 384, 3, 1, 1),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(384, 384, 3, 1, 1),
                nn.ReLU(inplace=True),

                nn.Conv2d(384, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2),
                )
        self.classifier = nn.Sequential(
                nn.Dropout2d(),
                nn.Linear(256*self.out_rows*self.out_rows, 4096),
                nn.ReLU(),
                nn.Dropout2d(),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Linear(4096, num_classes),
                nn.Softmax(),
                )
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 256*self.out_rows*self.out_rows)
        x = self.classifier(x)
        return x
