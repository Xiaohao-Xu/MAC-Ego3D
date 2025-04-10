import torch.nn as nn
import torch.nn.functional as F 
from utils import *
from inplace_abn import InPlaceABN 

class ConvBnReLU(nn.Module):
    '''
    unit net block
    '''
    def __init__(self, in_channels, out_channels,
                 kernel_size=3, stride=1, pad=1,
                 norm_act=InPlaceABN):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride=stride, padding=pad, bias=False)
        self.bn = norm_act(out_channels)


    def forward(self, x):
        return self.bn(self.conv(x))

class FeatureNet_multi(nn.Module):
    """
    2D feature network for neural point features
    """
    def __init__(self, intermediate=False, norm_act=InPlaceABN):
        super(FeatureNet_multi, self).__init__()


        self.conv0 = nn.Sequential(
                        ConvBnReLU(3, 8, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(8, 8, 3, 1, 1, norm_act=norm_act))

        self.conv1 = nn.Sequential(
                        ConvBnReLU(8, 16, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(16, 16, 3, 1, 1, norm_act=norm_act))

        self.conv2 = nn.Sequential(
                        ConvBnReLU(16, 32, 5, 2, 2, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act),
                        ConvBnReLU(32, 32, 3, 1, 1, norm_act=norm_act))

        self.toplayer = nn.Conv2d(32, 32, 1)
        self.intermediate = intermediate


    def forward(self, x):
        '''
        multi-level feature or not
        '''
        if self.intermediate:
            x1 = self.conv0(x)  # (B, 8, H, W)
            x2 = self.conv1(x1)  # (B, 16, H//2, W//2)
            x3 = self.conv2(x2)  # (B, 32, H//4, W//4)
            x3 = self.toplayer(x3)  # (B, 32, H//4, W//4)
            
            x2 = F.interpolate(x2, scale_factor=2, mode='bilinear', align_corners=False)
            x3 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            return [x1, x2, x3]
        else:
            # x: (B, 3, H, W)
            x = self.conv0(x) # (B, 8, H, W)
            x = self.conv1(x) # (B, 16, H//2, W//2)
            x = self.conv2(x) # (B, 32, H//4, W//4)
            x = self.toplayer(x) # (B, 32, H//4, W//4)

            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
            return [x]

class F_net(nn.Module):
    '''
    refer to Point-NeRF
    '''
    def __init__(self, input_channel, intermediate_channel, output_channel):
        super(F_net , self).__init__()
        self.embedding_one = nn.Sequential(
            nn.Linear(input_channel, intermediate_channel),
            nn.LeakyReLU(inplace=True)
        )
        self.embedding_two = nn.Sequential(
            nn.Linear(intermediate_channel, output_channel),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.embedding_one(x)
        x = self.embedding_two(x)
        return x

class F_net_radiance(nn.Module):
    '''
    refer to Point-NeRF
    '''  
    def __init__(self, input_channel, intermediate_channel, output_channel):
        super(F_net_radiance , self).__init__()
        self.embedding_one = nn.Sequential(
            nn.Linear(input_channel, intermediate_channel),
            nn.LeakyReLU(inplace=True)
        )
        self.embedding_two = nn.Sequential(
            nn.Linear(intermediate_channel, output_channel),
            nn.LeakyReLU(inplace=True)
        )
        
    def forward(self, x):
        x = self.embedding_one(x)
        x = self.embedding_two(x)
        return x