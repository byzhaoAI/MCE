import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.models.resnet import resnet18
from efficientnet_pytorch import EfficientNet


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip

class Decoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )

    def forward(self, x, batch_size=1, bev_flip_indices=None):
        b, c, h, w = x.shape

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            # x = x.view(batch_size, int(b//batch_size), c, h, w)
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])
            # x = x.view(b, c, h, w)

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x

class Encoder_eff(nn.Module):
    def __init__(self, C, version='b4'):
        super().__init__()
        self.C = C
        self.downsample = 8
        self.version = version

        if self.version == 'b0':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        elif self.version == 'b4':
            self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
        self.delete_unused_layers()

        if self.downsample == 16:
            if self.version == 'b0':
                upsampling_in_channels = 320 + 112
            elif self.version == 'b4':
                upsampling_in_channels = 448 + 160
            upsampling_out_channels = 512
        elif self.downsample == 8:
            if self.version == 'b0':
                upsampling_in_channels = 112 + 40
            elif self.version == 'b4':
                upsampling_in_channels = 160 + 56
            upsampling_out_channels = 128
        else:
            raise ValueError(f'Downsample factor {self.downsample} not handled.')

        self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
        self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

    def delete_unused_layers(self):
        indices_to_delete = []
        for idx in range(len(self.backbone._blocks)):
            if self.downsample == 8:
                if self.version == 'b0' and idx > 10:
                    indices_to_delete.append(idx)
                if self.version == 'b4' and idx > 21:
                    indices_to_delete.append(idx)

        for idx in reversed(indices_to_delete):
            del self.backbone._blocks[idx]

        del self.backbone._conv_head
        del self.backbone._bn1
        del self.backbone._avg_pooling
        del self.backbone._dropout
        del self.backbone._fc

    def get_features(self, x):
        # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()

        # Stem
        x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self.backbone._blocks):
            drop_connect_rate = self.backbone._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.backbone._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

            if self.downsample == 8:
                if self.version == 'b0' and idx == 10:
                    break
                if self.version == 'b4' and idx == 21:
                    break

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        if self.downsample == 16:
            input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
        elif self.downsample == 8:
            input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
        # print('input_1', input_1.shape)
        # print('input_2', input_2.shape)
        x = self.upsampling_layer(input_1, input_2)
        # print('x', x.shape)
        return x

    def forward(self, x):
        x = self.get_features(x)  # get feature vector
        x = self.depth_layer(x)  # feature and depth head
        return x
