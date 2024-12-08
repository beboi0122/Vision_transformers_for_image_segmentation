import torch
import torch.nn as nn

class UNET2D(nn.Module):
    def __init__(self, in_channels, out_channels, chanel_list=[8, 16, 32, 64]):
        super(UNET2D, self).__init__()
        self.down_blocks = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        self.in_channels = in_channels
        self.out_channels = out_channels

        for out_channel in chanel_list:
            self.down_blocks.append(
                nn.Sequential(
                    nn.Conv2d(self.in_channels, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
                )
            )
            self.in_channels = out_channel

        for out_channel in reversed(chanel_list):
            self.up_blocks.append(
                nn.ConvTranspose2d(out_channel * 2, out_channel, 2, 2)
            )
            self.up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_channel * 2, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU()
                )
            )
            self.in_channels = out_channel

            self.bottleneck = nn.Sequential(
                nn.Conv2d(chanel_list[-1], chanel_list[-1]*2, 3, padding=1),
                nn.BatchNorm2d(chanel_list[-1]*2),
                nn.ReLU(),
                nn.Conv2d(chanel_list[-1]*2, chanel_list[-1]*2, 3, padding=1),
                nn.BatchNorm2d(chanel_list[-1]*2),
                nn.ReLU()
            )

            self.final_block = nn.Conv2d(chanel_list[0], out_channels, 1)

    def forward(self, x):
        skip_connections = []
        for block in self.down_blocks:
            x = block(x)
            skip_connections.append(x)
            x = nn.MaxPool2d(2, 2)(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]


        for idx in range(0, len(self.up_blocks), 2):
            x = self.up_blocks[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:], mode='bilinear')

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.up_blocks[idx + 1](concat_skip)

        return self.final_block(x)