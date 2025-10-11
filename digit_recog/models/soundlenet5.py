import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from einops import rearrange

from models.modules import LeNet5, SNetPluse
from models.layer_new import GraphConstructor, mixprop


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel * reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel * reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x.transpose(1,2)).view(b, c)
        y = self.fc(y).view(b, c, 1)
        y = F.softmax(y, dim=1)
        return y


class InterModule(nn.Module):
    def __init__(self, dim_list, freeze, channels=256, num_layers=3, num_nodes=2, tanhalpha=3, gcn_depth=2, dropout=0.3, propalpha=0.05):
        super().__init__()
        num_nodes = len(dim_list)
        # align the channel dimension
        self.bottle_conv = nn.ModuleList()
        for i in range(num_nodes):
            self.bottle_conv.append(nn.Sequential(
                nn.Linear(dim_list[i], channels),
                # nn.InstanceNorm2d(channels),
                nn.BatchNorm1d(channels),
                nn.GELU(),
            ))

        self.gc = GraphConstructor(nnodes=num_nodes, dim=channels, alpha=tanhalpha, freeze=freeze)

        self.num_layers = num_layers
        self.gconv1, self.gconv2, self.norm = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        for _ in range(num_layers):
            self.gconv1.append(mixprop(channels, channels, gcn_depth, dropout, propalpha))
            self.gconv2.append(mixprop(channels, channels, gcn_depth, dropout, propalpha))
            self.norm.append(nn.InstanceNorm1d(channels))

        self.skipE = nn.Linear(channels, channels)
        self.conv = nn.Linear(channels, channels)

        # self.weighting = SELayer(num_nodes)

    def forward(self, fts):
        """
        :param fts: modality fts. list
        :return:
        """

        reg_fts = []
        for idx, ft in enumerate(fts):
            reg_fts.append(self.bottle_conv[idx](ft))
        features = torch.stack(reg_fts, dim=1)
        # features size: b, m, l
        Adj = self.gc(features)
        
        x = features
        b, m, l = features.shape
        for i in range(self.num_layers):
            residuals = self.gconv1[i](x, Adj) + self.gconv2[i](x, Adj.transpose(-1,-2))
            x = x + residuals
            x = self.norm[i](x)

        features = features + self.skipE(x)
        features = F.relu(features)
        features = self.conv(features)
        features = features.view(b, m, l)
        
        normalized_weight = self.att_forward(features, l)
        fused_features = features * normalized_weight
        fused_features = torch.sum(fused_features, dim=1).unsqueeze(1)

        features = torch.cat([features, fused_features], dim=1)
        return features

    def att_forward(self, value, C):
        score = torch.bmm(value, value.transpose(1, 2)) / np.sqrt(C)
        attn = F.softmax(score, -1)
        context = torch.bmm(attn, value)
        return F.softmax(context, dim=1)



class SoundLenet5(nn.Module):
    """docstring forLenet5 Sound"""
    def __init__(self, freeze=False, freeze_encoder=False):
        super(SoundLenet5, self).__init__()

        self.img_extractor = LeNet5()
        self.sound_extractor = SNetPluse()

        self.inter = InterModule(dim_list=[160, 320], channels=256, freeze=freeze)

        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 10)
        self.dropout = nn.Dropout()
        self.relu = nn.ReLU(inplace=True)
        self.softplus = nn.Softplus()
        self.softmax = nn.Softmax(dim=1)

        if freeze_encoder:
            for p in self.img_extractor.parameters():
                p.requires_grad = False
            for p in self.sound_extractor.parameters():
                p.requires_grad = False

    def forward(self, image, sound=None, only_image=False):
        _, img_feature = self.img_extractor(image)
        img_feature = img_feature.view(img_feature.size(0), -1)  
        # image -> torch.Size([b, 160])
        
        if not only_image:
            assert sound is not None
            _, sound_feature = self.sound_extractor(sound)
            sound_feature = sound_feature.view(sound_feature.size(0), -1)
            # sound -> torch.Size([b, 320])
            x = self.inter([img_feature, sound_feature])
        
        else:
            x = self.inter([img_feature])        

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
