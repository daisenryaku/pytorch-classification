import torch
import torch.nn as nn
from .BasicModule import BasicModule

class VGG16(BasicModule):
    def __init__(self, in_channels=3, img_rows=224, num_classes=2):
        super(VGG16, self).__init__()
        self.model_name = 'VGG16'
        self.out_rows = img_rows / 32
        self.feature = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                
                nn.Conv2d(256, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                )
        self.classifier = nn.Sequential(
                nn.Linear(512*self.out_rows*self.out_rows, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes)
                )
        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class VGG19(BasicModule):
    def __init__(self, in_channels=3, img_rows=224, num_classes=2):
        super(VGG19, self).__init__()
        self.model_name = 'VGG19'
        self.out_rows = img_rows / 32
        self.feture = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1),
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(64, 128, 3, 1, 1),
                nn.Conv2d(128, 128, 3, 1, 1),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(128, 256, 3, 1, 1),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(256, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.MaxPool2d(2, 2),

                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.Conv2d(512, 512, 3, 1, 1),
                nn.MaxPool2d(2, 2),
                )
        self.classifier = nn.Sequential(
                nn.Linear(512*self.out_rows*self.out_rows, 4096),
                nn.Linear(4096, 4096),
                nn.Linear(4096, num_classes)
                )
        self._initialize_weights()

    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

