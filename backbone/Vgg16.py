import torchvision
from torch import nn
from backbone.BackBoneInterface import backboneinterface


class vgg16(backboneinterface):

    def __init__(self, pretrain:bool):
        super(vgg16).__init__(pretrain)

    def feature(self):
        vgg16 = torchvision.models.vgg16(pretrained=self.pretrained)

        child = list(vgg16.children())
        feature = child[:-1]

        for param in feature.parameters():
            param.requires_grad = False

        feature = nn.Sequential(*feature)
        return feature

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(128, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        # self.layer3 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(256, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.layer4 = nn.Sequential(
        #     nn.Conv2d(256, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.layer5 = nn.Sequential(
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.Conv2d(512, 512, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        #
        #     nn.MaxPool2d(kernel_size=2, stride=2)
        # )
        #
        # self.fc1 = nn.Sequential(
        #     nn.Linear(-1, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU()
        # )
        #
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU()
        # )
        #
        # self.fc3 = nn.Sequential(
        #     nn.Linear(4096, 1000),
        #     nn.BatchNorm1d(1000),
        #     nn.ReLU()
        # )






