import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

@MODELS.register_module()
class FusionNeck(nn.Module):
    def __init__(self, in_channels, out_channels, num_outs):
        super(FusionNeck, self).__init__()
        # Assume in_channels is a list of channels from each backbone feature level
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_outs = num_outs

        # Implement fusion layers
        self.fuse_layers = nn.ModuleList()
        for i in range(len(in_channels)):
            self.fuse_layers.append(
                nn.Conv2d(in_channels[i], out_channels, kernel_size=1)
            )
        
        # Implement extra layers if the number of outs is more than the inputs
        self.extra_fuse_layers = nn.ModuleList()
        if num_outs > len(in_channels):
            for i in range(num_outs - len(in_channels)):
                self.extra_fuse_layers.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
                )

    def forward(self, inputs):
        # Apply fuse layers
        fused_feats = [self.fuse_layers[i](feature) for i, feature in enumerate(inputs)]
        
        # If num_outs is greater than the number of inputs, apply extra layers
        if len(fused_feats) < self.num_outs:
            for extra_layer in self.extra_fuse_layers:
                fused_feats.append(extra_layer(fused_feats[-1]))

        # Up-sample and concatenate features (naive example, replace with your actual fusion logic)
        out = []
        for feature in fused_feats:
            out.append(F.interpolate(feature, size=fused_feats[0].shape[-2:], mode='nearest'))
        
        return tuple(out)
