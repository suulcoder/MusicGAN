import torch
from settings import *
from DiscriminatorNode import *


class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv0 = torch.nn.ModuleList([
            DiscriminatorNode(1, 16, (1, 1, 12), (1, 1, 12)) for _ in range(number_of_trakcs)
        ])
        self.conv1 = torch.nn.ModuleList([
            DiscriminatorNode(16, 16, (1, 4, 1), (1, 4, 1)) for _ in range(number_of_trakcs)
        ])
        self.conv2 = DiscriminatorNode(16 * 5, 64, (1, 1, 3), (1, 1, 1))
        self.conv3 = DiscriminatorNode(64, 64, (1, 1, 4), (1, 1, 4))
        self.conv4 = DiscriminatorNode(64, 128, (1, 4, 1), (1, 4, 1))
        self.conv5 = DiscriminatorNode(128, 128, (2, 1, 1), (1, 1, 1))
        self.conv6 = DiscriminatorNode(128, 256, (3, 1, 1), (3, 1, 1))
        self.dense = torch.nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(-1, number_of_trakcs, number_of_measures, measure_resolution, number_of_pitches)
        x = [conv(x[:, [i]]) for i, conv in enumerate(self.conv0)]
        x = torch.cat([conv(x_) for x_, conv in zip(x, self.conv1)], 1)
        x = self.conv2(x)
        x = self.conv3(x)          
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(-1, 256)
        x = self.dense(x)
        return x