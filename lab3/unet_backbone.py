from torch import nn


class UnetDownModule(nn.Module):

    """ U-Net downsampling block. """

    def __init__(self, in_channels, out_channels, downsample=True):
        super(UnetDownModule, self).__init__()

        # layers: optional downsampling, 2 x (conv + bn + relu)
        self.maxpool = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=5, stride=2, dilation=5) if downsample else None
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        if self.maxpool is not None:
            x = self.maxpool(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class UnetEncoder(nn.Module):

    """ U-Net encoder. https://arxiv.org/pdf/1505.04597.pdf """

    def __init__(self, num_channels):
        super(UnetEncoder, self,).__init__()
        self.module1 = UnetDownModule(num_channels, 64, downsample=False)
        self.module2 = UnetDownModule(64, 128)
        self.module3 = UnetDownModule(128, 256)
        self.module4 = UnetDownModule(256, 512)
        self.module5 = UnetDownModule(512, 1024)

    def forward(self, x):
        x = self.module1(x)
        x = self.module2(x)
        x = self.module3(x)
        x = self.module4(x)
        x = self.module5(x)
        return x