import torch.nn as nn


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64, num_layers: int = 4) -> None:
        super().__init__()
        layers = [nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.2, inplace=True)]
        channels = base_channels
        for _ in range(1, num_layers):
            next_channels = min(channels * 2, 512)
            layers.extend([
                nn.Conv2d(channels, next_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(next_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            channels = next_channels
        layers.append(nn.Conv2d(channels, 1, kernel_size=4, stride=1, padding=1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
