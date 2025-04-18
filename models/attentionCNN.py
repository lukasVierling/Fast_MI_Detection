import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LeadProcessingBlock1D(nn.Module):
    def __init__(self, channels, dilation):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )
        self.second_layer = nn.Sequential(
            nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.second_layer(self.first_layer(x))
        #skip connection
        return out + x
    
class LeadMixingBlock(nn.Module):
    def __init__(self, num_leads, filters_per_lead, num_heads):
        super().__init__()
        self.num_leads = num_leads
        self.filters_per_lead = filters_per_lead
        self.attn = nn.MultiheadAttention(embed_dim = filters_per_lead, num_heads = num_heads, batch_first=True)
    
    def forward(self,x):
        B,C,T = x.shape
        L = self.num_leads
        F = self.filters_per_lead
        #reshape the input to get the leads  (C=FT)
        out = x.view(B, L, F, T)

        # B*T attention probelms
        out = out.permute(0,3,1,2).reshape(B*T, L, F)

        attn_out, _ = self.attn(out, out, out)
        out = attn_out.reshape(B,T,L,F).permute(0,2,3,1)

        return out.reshape(B,C,T)
    
class CNN_1D(nn.Module):
    def __init__(self, num_leads=12, hidden_channels=64, filters_1d = 20, kernel_1d = 100, stride_1d = 50, attn_heads = 4, res_dilations = [1,2,4]):
        super().__init__()
        #encoder
        hidden_channels = num_leads * filters_1d
        self.encoder = nn.Sequential(
            nn.Conv1d(num_leads, hidden_channels, kernel_size=kernel_1d, stride=stride_1d, groups=num_leads,bias=False),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        self.LeadBlocks = nn.Sequential(
            LeadProcessingBlock1D(hidden_channels, dilation=res_dilations[0]),
            LeadMixingBlock(num_leads, filters_1d, attn_heads),
            LeadProcessingBlock1D(hidden_channels, dilation=res_dilations[1]),
            LeadMixingBlock(num_leads, filters_1d, attn_heads),
            LeadProcessingBlock1D(hidden_channels, dilation=res_dilations[2]),
            LeadMixingBlock(num_leads, filters_1d, attn_heads),
        )
        self.last_conv = nn.Sequential(nn.Conv1d(hidden_channels, hidden_channels, kernel_size=3, padding=16, dilation=16, bias=False),
                                       nn.BatchNorm1d(hidden_channels),
                                       nn.ReLU(inplace=True))
        self.pooling = nn.AdaptiveAvgPool2d((1))
        self.fc = nn.Linear(hidden_channels, 1)
    def forward(self, x):
        B = x.size(0)
        out = self.encoder(x)
        out = self.LeadBlocks(out)
        out = self.last_conv(out)
        out = self.pooling(out)
        #flatten 
        out = out.view(B,-1)
        out = self.fc(out)
        return out

