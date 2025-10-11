import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
sys.path.append("..")

import utils.geom
import utils.vox
import utils.misc
import utils.basic

from imml_nets.module import Encoder_res101, Encoder_res50, Encoder_eff, Decoder


EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None, 
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 use_camera=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101",
                 device='cuda'):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])
        self.device = device

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_camera = use_camera
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()
        
        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # BEV compressor
        cat_dim = 0
        if self.use_camera:
            cat_dim += feat2d_dim * Y

        if self.use_radar:
            if self.use_metaradar:
                cat_dim += 16 * Y
            else:
                cat_dim += 1
                
        if self.use_lidar:
            cat_dim += Y
        
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(cat_dim, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )

        # Decoder
        self.decoder = Decoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        # set_bn_momentum(self, 0.1)

        if vox_util is not None:
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, mask, label_tuple, loss_fn,
                rgb_camXs, pix_T_cams, cam0_T_camXs, 
                vox_util, lid_occ_mem0=None, rad_occ_mem0=None, 
                modal_weight=[1,1,1], is_training=True):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if lid_occ_mem0 is not None:
                lid_occ_mem0[self.bev_flip1_index] = torch.flip(lid_occ_mem0[self.bev_flip1_index], [-1])
                lid_occ_mem0[self.bev_flip2_index] = torch.flip(lid_occ_mem0[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        feat_bev_ = []
        if self.use_camera:
            rgb_bev = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            feat_bev_.append(rgb_bev * mask[:, :, 0:1].unsqueeze(-1))
        
        if self.use_lidar:
            assert(lid_occ_mem0 is not None)
            lid_bev = lid_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_.append(lid_bev * mask[:, :, 1:2].unsqueeze(-1))

        if self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
            else:
                rad_bev = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
            feat_bev_.append(rad_bev * mask[:, :, 2:3].unsqueeze(-1))

        feat_bev_ = torch.cat(feat_bev_, dim=1)
        feat_bev_ = self.bev_compressor(feat_bev_)

        # bev decoder
        out_dict = self.decoder(feat_bev_, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        total_loss, metrics = self.cal_loss(out_dict, label_tuple, loss_fn)
        if is_training:
            return total_loss, metrics
        return metrics, torch.zeros(1).to(device), torch.zeros(1).to(device)
    

    def cal_loss(self, out_dict, label_tuple, loss_fn):
        total_loss = torch.zeros(1).to(self.device)

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_bev_e = out_dict['segmentation']
        center_bev_e = out_dict['instance_center']
        offset_bev_e = out_dict['instance_offset']

        seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g = label_tuple

        ce_loss = loss_fn(seg_bev_e, seg_bev_g, valid_bev_g)
        center_loss = balanced_mse_loss(center_bev_e, center_bev_g)
        offset_loss = torch.abs(offset_bev_e-offset_bev_g).sum(dim=1, keepdim=True)
        offset_loss = utils.basic.reduce_masked_mean(offset_loss, seg_bev_g*valid_bev_g)

        ce_factor = 1 / torch.exp(self.ce_weight)
        ce_loss = 10.0 * ce_loss * ce_factor
        ce_uncertainty_loss = 0.5 * self.ce_weight

        center_factor = 1 / (2*torch.exp(self.center_weight))
        center_loss = center_factor * center_loss
        center_uncertainty_loss = 0.5 * self.center_weight

        offset_factor = 1 / (2*torch.exp(self.offset_weight))
        offset_loss = offset_factor * offset_loss
        offset_uncertainty_loss = 0.5 * self.offset_weight

        total_loss += ce_loss
        total_loss += center_loss
        total_loss += offset_loss
        total_loss += ce_uncertainty_loss
        total_loss += center_uncertainty_loss
        total_loss += offset_uncertainty_loss

        seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum(dim=[1,2,3])
        union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum(dim=[1,2,3])
        iou = (intersection/(1e-4 + union)).mean()

        metrics = dict()
        metrics['ce_loss'] = ce_loss.item()
        metrics['center_loss'] = center_loss.item()
        metrics['offset_loss'] = offset_loss.item()
        metrics['ce_weight'] = self.ce_weight.item()
        metrics['center_weight'] = self.center_weight.item()
        metrics['offset_weight'] = self.offset_weight.item()
        metrics['iou'] = iou.item()
        return total_loss, metrics

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = utils.basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = utils.basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss