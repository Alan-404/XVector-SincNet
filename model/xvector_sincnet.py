import torch
import torch.nn as nn
from model.modules.net import SincNet
from model.utils.pooling import StatsPool

from typing import Optional

class XVectorSincNet(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        dimension: int = 512,
    ):
        super().__init__()

        self.sincnet = SincNet(sample_rate)
        in_channel = 60

        self.tdnns = nn.ModuleList()
        out_channels = [512, 512, 512, 512, 1500]
        self.kernel_size = [5, 3, 3, 1, 1]
        self.dilation = [1, 2, 3, 1, 1]
        self.padding = [0, 0, 0, 0, 0]
        self.stride = [1, 1, 1, 1, 1]

        for out_channel, kernel_size, dilation in zip(out_channels, self.kernel_size, self.dilation):
            self.tdnns.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channel,
                        out_channels=out_channel,
                        kernel_size=kernel_size,
                        dilation=dilation,
                    ),
                    nn.LeakyReLU(),
                    nn.BatchNorm1d(out_channel),
                ]
            )
            in_channel = out_channel

        self.stats_pool = StatsPool()

        self.embedding = nn.Linear(in_channel * 2, dimension)

    def forward(
        self, waveforms: torch.Tensor, weights: Optional[torch.Tensor] = None
    ) -> torch.Tensor:

        outputs = self.sincnet(waveforms).squeeze(dim=1)
        for tdnn in self.tdnns:
            outputs = tdnn(outputs)
        outputs = self.stats_pool(outputs, weights=weights)
        return self.embedding(outputs)