import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
'''
Reimplementation of the model proposed in 
Automating detection and localization of myocardialinfarction using
shallow and end-to-end deep neural networks.
link:https://www.sciencedirect.com/science/article/pii/S1568494620303239
Authors: Jafarian et al.
'''
class ResidualBlock2D(nn.Module):
    def __init__(self, channels, dilation1, dilation2):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation1, dilation=dilation1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=dilation2, dilation=dilation2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.second_layer(self.first_layer(x))
        #skip connection
        return out + x

class CNN_2D(nn.Module):
    def __init__(self, num_leads=12, filters=20, kernel_size=100, stride=50):
        super().__init__()
        hidden_channels = num_leads * filters
        #encoder
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_channels, kernel_size=kernel_size, stride=stride, groups=num_leads,bias=False),
            nn.ReLU(inplace=True)
        )
        #increase the depth with 2D conv
        self.first_conv = nn.Sequential(nn.Conv2d(num_leads, hidden_channels, kernel_size=3, padding=1, bias=False),
                                          nn.ReLU(inplace=True))
        #res blocks
        self.ResBlocks = nn.Sequential(
            ResidualBlock2D(hidden_channels, dilation1=1, dilation2=1),
            ResidualBlock2D(hidden_channels, dilation1=2, dilation2=2),
            ResidualBlock2D(hidden_channels, dilation1=4, dilation2=8),
        )

        self.last_conv = nn.Sequential(nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=16, dilation=16, bias=False),
                                       nn.BatchNorm2d(hidden_channels),
                                       nn.ReLU(inplace=True))
        self.pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(hidden_channels, 1)

    def forward(self,x):
        B = x.size(0)

        out = self.encoder(x)
        _, C, T = out.shape

        #reshape to fit into the 2D conv
        filter_size = C//x.size(1)
        out = out.view(B, x.size(1), filter_size, T).permute(0,1,3,2)

        out = self.first_conv(out)

        #res blocks
        out = self.ResBlocks(out)

        #final conv
        out = self.last_conv(out)

        #pooling and  fc
        out = self.pooling(out)
        #flatten
        out = out.view(B, -1)
        out = self.fc(out)

        return out