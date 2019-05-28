import torchvision.models as models
import torch.nn as nn
import torch.nn.functional as F


def resnet18(numclasses, pretrained=False):
    model = models.resnet18(pretrained)
    conv1_out_channels = model.conv1.out_channels
    model.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=2)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, numclasses)
    return model


def resnet34(numclasses, pretrained=False):
    model = models.resnet34(pretrained)
    conv1_out_channels = model.conv1.out_channels
    model.conv1 = nn.Conv2d(1, conv1_out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False)
    model.maxpool = nn.MaxPool2d(kernel_size=2)
    fc_features = model.fc.in_features
    model.fc = nn.Linear(fc_features, numclasses)
    return model


class ConvNet(nn.Module):
    def __init__(self, numclasses):
        """
        args
        """
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512*3*3, 512),
            nn.Linear(512, numclasses)
        )

    def forward(self, net):
        net = self.layer1(net)
        net = self.layer2(net)
        net = self.layer3(net)
        net = net.view(net.size(0), -1)
        net = self.fc(net)
        return net


def convnet(numclasses):
    return ConvNet(numclasses)
