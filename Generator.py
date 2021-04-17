import torch
from settings import *
from GeneratorNode import *

class Generator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.transconv0 = GeneratorNode(latent_dim, 256, (4, 1, 1), (4, 1, 1))
        self.transconv1 = GeneratorNode(256, 128, (1, 4, 1), (1, 4, 1))
        self.transconv2 = GeneratorNode(128, 64, (1, 1, 4), (1, 1, 4))
        self.transconv3 = GeneratorNode(64, 32, (1, 1, 3), (1, 1, 1))
        self.transconv4 = torch.nn.ModuleList([
            GeneratorNode(32, 16, (1, 4, 1), (1, 4, 1))
            for _ in range(number_of_trakcs)
        ])
        self.transconv5 = torch.nn.ModuleList([
            GeneratorNode(16, 1, (1, 1, 12), (1, 1, 12))
            for _ in range(number_of_trakcs)
        ])

    def forward(self, x):
        x = x.view(-1, latent_dim, 1, 1, 1)
        x = self.transconv0(x)
        x = self.transconv1(x)
        x = self.transconv2(x)
        x = self.transconv3(x)
        x = [transconv(x) for transconv in self.transconv4]
        x = torch.cat([transconv(x_) for x_, transconv in zip(x, self.transconv5)], 1)
        x = x.view(-1, number_of_trakcs, number_of_measures * measure_resolution, number_of_pitches)
        return x