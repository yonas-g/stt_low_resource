import torch.nn as nn
import math


class InvertedResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, expand_ratio, first_stage=False, first_block=False):
        super().__init__()
        
        hidden_dim = out_channels * expand_ratio
        
        self.transform_conv = None
        if(not first_stage and first_block):
            self.transform_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=2)

        self.spatial_mixing = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size=7, padding=3, groups=in_channels, bias=False),
            nn.BatchNorm1d(out_channels),
        )

        self.feature_mixing = nn.Sequential(
            nn.Conv1d(out_channels, hidden_dim, kernel_size=1, stride=1),
            nn.GELU(),
        )

        self.bottleneck_channels = nn.Sequential(
            nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1),
        )

    def forward(self, x):
        if(self.transform_conv):
            x = self.transform_conv(x)

        out = self.spatial_mixing(x)
        out = self.feature_mixing(out)
        out = self.bottleneck_channels(out)

        return x + out

class ConvNext(nn.Module):
    
    def __init__(self):
        super().__init__()

        
        self.stem = nn.Sequential(
            nn.Conv1d(256, 96, kernel_size=4, stride=4, bias=False),
            nn.BatchNorm1d(96),
            nn.GELU(),
        )

        self.stage_cfgs = [
            [4,  96, 3]
        ]

        in_channels = 96

        layers = []
        for idx, curr_stage in enumerate(self.stage_cfgs):
            expand_ratio, out_channels, num_blocks = curr_stage
            for block_idx in range(num_blocks):
                
                block = InvertedResidualBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    expand_ratio=expand_ratio,
                    first_stage=True if idx == 0 else False,
                    first_block=True if block_idx == 0 else False
                )
                layers.append(block)
                
                in_channels = out_channels 
            
        self.layers = nn.Sequential(*layers)

        self.final_block = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=1, padding=0, stride=1, bias=False),
            nn.BatchNorm1d(256),
            nn.GELU()
        )

        

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.stem(x)
        out = self.layers(out)
        feats = self.final_block(out)

        return feats