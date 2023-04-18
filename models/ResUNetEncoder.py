
from torch import nn
import numpy as np
import models.utils as utils
from models.ResUNet import ResUNetDownBlock


class ResUNetEncoder(nn.Module):
    def __init__(self, in_channels=1, depth=4, wf=6):
        super(ResUNetEncoder, self).__init__()
        self.depth = depth
        self.wf = wf

        prev_channels = in_channels
        self.down_path = nn.ModuleList()

        self.down_path.append(ResUNetDownBlock(prev_channels, 2 ** (wf + 0), True))
        prev_channels = 2 ** (wf + 0)
        for i in range(1, depth):
            self.down_path.append(ResUNetDownBlock(prev_channels, 2 ** (wf + i)))
            prev_channels = 2 ** (wf + i)

        self.last = nn.Sequential(nn.Linear(prev_channels * 2 ** 6, 1), nn.Sigmoid())

        utils.init_weights(self.modules())

    def forward(self, x):
        for i, down in enumerate(self.down_path):
            x = down(x)
        x = x.reshape([x.shape[0], np.prod(x.shape[1:])])
        x = self.last(x)
        return x

