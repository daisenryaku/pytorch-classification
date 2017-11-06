from .BasicModule import BasicModule
import torch.nn as nn


class LeNet(BasicModule):
    def __init__(self, img_rows=28, num_classes=10):
        super(LeNet, self).__init__()
        self.model_name = 'LeNet'
        self.out_rows = ((img_rows - 4)/2 - 4)/2 
        self.features = nn.Sequential(
                nn.Conv2d(1, 20, 5),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
                nn.Conv2d(20, 50, 5),
                nn.Dropout2d(),
                nn.MaxPool2d(2, 2),
                nn.ReLU(inplace=True),
                )
        self.classifier = nn.Sequential(
                nn.Linear(self.out_rows * self.out_rows * 50, 500),
                nn.ReLU(inplace=True),
                nn.Dropout2d(),
                nn.Linear(500, num_classes),
                nn.LogSoftmax(),
                )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, 800)
        x = self.classifier(x)
        return x
