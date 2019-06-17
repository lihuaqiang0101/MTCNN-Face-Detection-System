import torch
from torch import nn
import torch.nn.functional as F

class Pnet(nn.Module):
    def __init__(self):
        super(Pnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,10,kernel_size=3,padding=1,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(3,2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,16,3,stride=1),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(16,32,3,padding=0,stride=1),
            nn.PReLU()
        )
        self.confidence = nn.Conv2d(32,1,kernel_size=1,stride=1)
        self.offset = nn.Conv2d(32,4,kernel_size=1,stride=1)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        classification = self.confidence(x)
        bbox = self.offset(x)
        classification = F.sigmoid(classification)
        return classification,bbox



class Rnet(nn.Module):
    def __init__(self):
        super(Rnet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,28,3,padding=1,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(28,48,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(48,64,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.classification = nn.Conv2d(64,1,kernel_size=3)
        self.bbox = nn.Conv2d(64,4,3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        classification = self.classification(x)
        bbox = self.bbox(x)
        classification = F.sigmoid(classification)
        return classification.view(-1,1),bbox.view(-1,4)


class Onet(nn.Module):
    def __init__(self):
        super(Onet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,kernel_size=3,padding=1,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3,stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64,128,kernel_size=2,stride=1),
            nn.PReLU()
        )
        self.classification = nn.Conv2d(128,1,3)
        self.bbox = nn.Conv2d(128,4,3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        classification = F.sigmoid(self.classification(x))
        bbox = self.bbox(x)
        return classification.view(-1,1),bbox.view(-1,4)