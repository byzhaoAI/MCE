import sys
sys.path.append("..")
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import utils.geom
import utils.vox
import utils.misc
import utils.basic
from imml_nets.transformer import Transformer
from imml_nets.module import Encoder_res101, Encoder_res50, Encoder_eff, Decoder


EPS = 1e-4
num_modals = 3

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
                 device='cpu'):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])
        self.device = device
        self.modal_set = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]

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
        
        self.feat2d_dim = feat2d_dim = latent_dim
        # RGB Encoder
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        elif encoder_type == "effb0":
            self.encoder = Encoder_eff(feat2d_dim, version='b0')
        else:
            # effb4
            self.encoder = Encoder_eff(feat2d_dim, version='b4')

        # Aligner
        self.rgb_aligner = nn.Sequential(
            nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(feat2d_dim),
            nn.GELU(),
        )
        self.lidar_aligner = nn.Sequential(
            nn.Conv2d(Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(feat2d_dim),
            nn.GELU(),
        )
        self.radar_aligner = nn.Sequential(
            nn.Conv2d(16*Y if self.use_metaradar else 1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(feat2d_dim),
            nn.GELU(),
        )
        
        # Transformer: cross attention
        self.rgb_pos = nn.Parameter(torch.zeros(1, 1, feat2d_dim))
        self.lid_pos = nn.Parameter(torch.zeros(1, 1, feat2d_dim))
        self.rad_pos = nn.Parameter(torch.zeros(1, 1, feat2d_dim))
        self.netInter = Transformer(feat2d_dim, mlp_dim=512)

        # Compressor
        self.bev_compressor = nn.Sequential(
            nn.Conv2d(feat2d_dim*3, latent_dim, kernel_size=3, padding=1, stride=1, bias=False),
            nn.InstanceNorm2d(latent_dim),
            nn.GELU(),
        )

        # Decoder
        self.decoder = Decoder(in_channels=latent_dim, n_classes=1, predict_future_flow=False)
        self.rgb_decoder = Decoder(in_channels=latent_dim, n_classes=1, predict_future_flow=False)
        self.lidar_decoder = Decoder(in_channels=latent_dim, n_classes=1, predict_future_flow=False)
        self.radar_decoder = Decoder(in_channels=latent_dim, n_classes=1, predict_future_flow=False)

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
        
    def forward(self, mask, modal_weight, label_tuple, loss_fn,
                rgb_camXs, pix_T_cams, cam0_T_camXs, 
                vox_util, rad_occ_mem0=None, lid_occ_mem0=None,
                is_training=False):
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
        Z, Y, X = self.Z, self.Y, self.X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
        
        # RGB encoding
        rgb_bev = self.camera_feat(rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util)
        if self.rand_flip:
            rgb_bev[self.bev_flip1_index] = torch.flip(rgb_bev[self.bev_flip1_index], [-1])
            rgb_bev[self.bev_flip2_index] = torch.flip(rgb_bev[self.bev_flip2_index], [-3])
        rgb_bev = rgb_bev.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
        rgb_bev = self.rgb_aligner(rgb_bev)

        # Lidar encoding
        if self.rand_flip:
            lid_occ_mem0[self.bev_flip1_index] = torch.flip(lid_occ_mem0[self.bev_flip1_index], [-1])
            lid_occ_mem0[self.bev_flip2_index] = torch.flip(lid_occ_mem0[self.bev_flip2_index], [-3])
        lid_bev = lid_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
        lid_bev = self.lidar_aligner(lid_bev)

        # Radar encoding
        if self.rand_flip:
            rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
            rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])
        if self.use_metaradar:
            rad_bev = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
        else:
            rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1)
        rad_bev = self.radar_aligner(rad_bev)

        # Cross attention
        rgb_bev_rec, lid_bev_rec, rad_bev_rec = self.cross_attention(rgb_bev, lid_bev, rad_bev, B, Z, X, mask)

        # bev compressing
        feat_bev = torch.cat((rgb_bev_rec, lid_bev_rec, rad_bev_rec), dim=1)
        feat_bev = self.bev_compressor(feat_bev)
        
        # bev decoder
        compute_output = self.compute_loss(
            self.decoder(feat_bev, B, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None),
            label_tuple, loss_fn, self.device, is_training=is_training
        )

        rgb_dict = self.rgb_decoder(rgb_bev_rec, B, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        lid_dict = self.lidar_decoder(lid_bev_rec, B, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)
        rad_dict = self.radar_decoder(rad_bev_rec, B, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        # sample level [B, num_modal] ×
        # batch level [num_modal] √
        independent_contrib = torch.stack([
            (((rgb_dict['segmentation'].sigmoid().round() == label_tuple[0]) * label_tuple[1] * mask[:,:,0].unsqueeze(-1).unsqueeze(-1)).sum(dim=[1,2,3]) / torch.sum(label_tuple[1], dim=[1,2,3])).sum().float(),
            (((lid_dict['segmentation'].sigmoid().round() == label_tuple[0]) * label_tuple[1] * mask[:,:,1].unsqueeze(-1).unsqueeze(-1)).sum(dim=[1,2,3]) / torch.sum(label_tuple[1], dim=[1,2,3])).sum().float(),
            (((rad_dict['segmentation'].sigmoid().round() == label_tuple[0]) * label_tuple[1] * mask[:,:,2].unsqueeze(-1).unsqueeze(-1)).sum(dim=[1,2,3]) / torch.sum(label_tuple[1], dim=[1,2,3])).sum().float()
        ], dim=0)

        bs_subsets = self.gen_bs_subsets(mask[:,0], self.device)
        bs_individual_contrib = dict()
        bs_multimodal_contrib = dict()
        for value in self.modal_set:
            if sum(value) > 1:
                bs_multimodal_contrib[value] = 0
            else:
                bs_individual_contrib[value] = 0

        if is_training:
            # fusion loss
            rgb_loss, _ = self.compute_loss(rgb_dict, label_tuple, loss_fn, self.device, is_training=True)
            lid_loss, _ = self.compute_loss(lid_dict, label_tuple, loss_fn, self.device, is_training=True)
            rad_loss, _ = self.compute_loss(rad_dict, label_tuple, loss_fn, self.device, is_training=True)

            # separate loss
            sep_loss = torch.zeros(3).float().to(self.device)
            sep_loss[0:1] += rgb_loss
            sep_loss[1:2] += lid_loss
            sep_loss[2:3] += rad_loss

            # subset loss
            rec_loss = torch.zeros(B,3).float().to(self.device)
            subfuse_loss = torch.zeros(B,1).float().to(self.device)

            # compute shapley value and shapley loss                
            # for each sample
            for b in range(B):
                # shape: subsets - (len_subsets, len_modal)
                subsets = bs_subsets[b]

                rec_loss_rgb, rec_loss_lid, rec_loss_rad = torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device)
                for _subset in subsets:
                    # Reconstruction
                    rgb_bev_rec, lid_bev_rec, rad_bev_rec = self.cross_attention(
                        rgb_bev[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                        lid_bev[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                        rad_bev[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                        1, Z, X, _subset.view(1,1,-1)
                    )
                    # bev compressing
                    feat_bev = torch.cat((rgb_bev_rec, lid_bev_rec, rad_bev_rec), dim=1)
                    feat_bev = self.bev_compressor(feat_bev)
                    
                    # bev decoder
                    b_dict = self.decoder(feat_bev, 1, (self.bev_flip1_index[b:b+1], self.bev_flip2_index[b:b+1]) if self.rand_flip else None)
                    b_label_tuple = [b_seg[b:b+1] for b_seg in label_tuple]
                    b_fuse_loss, b_fuse_metrics = self.compute_loss(b_dict, b_label_tuple, loss_fn, self.device, is_training=True)
                    subfuse_loss[b:b+1] += b_fuse_loss #*self.ce_weight
                        
                    # compute auxiliary completion loss
                    rec_mask = (mask[b][0] - _subset)
                    rec_loss_rgb += F.l1_loss(rgb_bev[b:b+1], rgb_bev_rec) * rec_mask[0]
                    rec_loss_lid += F.l1_loss(lid_bev[b:b+1], lid_bev_rec) * rec_mask[1]
                    rec_loss_rad += F.l1_loss(rad_bev[b:b+1], rad_bev_rec) * rec_mask[2]

                    # calculate the positions where the elements of the two tensors are equal, and sum to get the total number of equal elements
                    if (_subset).sum() > 1:
                        bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (((b_dict['segmentation'].sigmoid().round() == b_label_tuple[0]) * b_label_tuple[1]).sum(dim=[1,2,3]) / b_label_tuple[1].sum(dim=[1,2,3])).sum().float()
                    else:
                        bs_individual_contrib[tuple(_subset.cpu().numpy())] += (((b_dict['segmentation'].sigmoid().round() == b_label_tuple[0]) * b_label_tuple[1]).sum(dim=[1,2,3]) / b_label_tuple[1].sum(dim=[1,2,3])).sum().float()
                rec_loss[b] = torch.cat([rec_loss_rgb, rec_loss_lid, rec_loss_rad], dim=0)

            # torch.Size([num_modal])
            bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)
            shap_mask = (bs_shap_value < independent_contrib).int().to(self.device)
            shap_loss = (independent_contrib - bs_shap_value) * shap_mask

            rec_loss = rec_loss * modal_weight * shap_loss
            sep_loss = sep_loss * modal_weight * shap_loss

            return compute_output[1], compute_output[0], sep_loss, shap_loss, \
                rec_loss / (2**mask[:,0].sum(dim=1, keepdim=True)-1), subfuse_loss / (2**mask[:,0].sum(dim=1, keepdim=True)-1)
                   
        # for each sample
        for b in range(B):
            # shape: subsets - (len_subsets, len_modal)
            subsets = bs_subsets[b]

            rec_loss_rgb, rec_loss_lid, rec_loss_rad = torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device)
            for _subset in subsets:
                # Reconstruction
                rgb_bev_rec, lid_bev_rec, rad_bev_rec = self.cross_attention(
                    rgb_bev[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                    lid_bev[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                    rad_bev[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat2d_dim, Z, X)).to(self.device),
                    1, Z, X, _subset.view(1,1,-1)
                )
                # bev compressing
                feat_bev = torch.cat((rgb_bev_rec, lid_bev_rec, rad_bev_rec), dim=1)
                feat_bev = self.bev_compressor(feat_bev)
            
                # bev decoder
                b_dict = self.decoder(feat_bev, 1, (self.bev_flip1_index[b:b+1], self.bev_flip2_index[b:b+1]) if self.rand_flip else None)
                b_label_tuple = [b_seg[b:b+1] for b_seg in label_tuple]

                # calculate the positions where the elements of the two tensors are equal, and sum to get the total number of equal elements
                if (_subset).sum() > 1:
                    bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (((b_dict['segmentation'].sigmoid().round() == b_label_tuple[0]) * b_label_tuple[1]).sum(dim=[1,2,3]) / b_label_tuple[1].sum(dim=[1,2,3])).sum().float()
                else:
                    bs_individual_contrib[tuple(_subset.cpu().numpy())] += (((b_dict['segmentation'].sigmoid().round() == b_label_tuple[0]) * b_label_tuple[1]).sum(dim=[1,2,3]) / b_label_tuple[1].sum(dim=[1,2,3])).sum().float()
        
        # torch.Size([num_modal])
        bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)
        shap_mask = (bs_shap_value < independent_contrib).int().to(self.device)
        shap_loss = (independent_contrib - bs_shap_value) * shap_mask
        return compute_output, independent_contrib, bs_shap_value

    def camera_feat(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util):
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
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(self.device)) / self.std.to(self.device)
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
            xyz_camA = self.xyz_camA.to(self.device).repeat(B*S,1,1)
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
        return feat_mem
    
    def cross_attention(self, rgb_bev, lid_bev, rad_bev, B, Z, X, mask):
        feat_bev = torch.stack((
            rgb_bev.permute(0, 2, 3, 1).contiguous().view(B, -1, self.feat2d_dim) * mask[:, :, 0:1],
            lid_bev.permute(0, 2, 3, 1).contiguous().view(B, -1, self.feat2d_dim) * mask[:, :, 1:2],
            rad_bev.permute(0, 2, 3, 1).contiguous().view(B, -1, self.feat2d_dim) * mask[:, :, 2:3]
        ), dim=2)
        multimodal_pos = torch.stack((self.rgb_pos, self.lid_pos, self.rad_pos), dim=2)

        feat_bev = self.netInter(feat_bev, multimodal_pos)
        rgb_bev_rec, lid_bev_rec, rad_bev_rec = feat_bev[:,:,0], feat_bev[:,:,1], feat_bev[:,:,2]

        rgb_bev_rec = rgb_bev_rec.permute(0, 2, 1).contiguous().view(B, self.feat2d_dim, Z, X)
        lid_bev_rec = lid_bev_rec.permute(0, 2, 1).contiguous().view(B, self.feat2d_dim, Z, X)
        rad_bev_rec = rad_bev_rec.permute(0, 2, 1).contiguous().view(B, self.feat2d_dim, Z, X)
        return rgb_bev_rec, lid_bev_rec, rad_bev_rec

    def compute_loss(self, out_dict, label_tuple, loss_fn, device, is_training=False):
        total_loss = torch.zeros(1).to(device)
        # raw_e = out_dict['raw_feat']
        # feat_e = out_dict['feat']
        seg_bev_e = out_dict['segmentation']
        seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g = label_tuple

        # compute iou metrics
        # (B)
        seg_bev_e_round = torch.sigmoid(seg_bev_e).round()
        intersection = (seg_bev_e_round*seg_bev_g*valid_bev_g).sum(dim=[1,2,3])
        union = ((seg_bev_e_round+seg_bev_g)*valid_bev_g).clamp(0,1).sum(dim=[1,2,3])
        batch_iou = intersection / (1e-4 + union)
        # (1)
        iou= (intersection / (1e-4 + union)).mean()

        metrics = dict()
        # metrics['ce_loss'] = ce_loss.item()
        # metrics['center_loss'] = center_loss.item()
        # metrics['offset_loss'] = offset_loss.item()
        # metrics['ce_weight'] = self.ce_weight.item()
        # metrics['center_weight'] = self.center_weight.item()
        # metrics['offset_weight'] = self.offset_weight.item()
        

        metrics['batch_intersection'] = intersection
        metrics['batch_union'] = union
        metrics['batch_iou'] = batch_iou
        metrics['iou'] = iou.item()

        if is_training:
            center_bev_e = out_dict['instance_center']
            offset_bev_e = out_dict['instance_offset']

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
            return total_loss, metrics
        return metrics
    
    def gen_subsets(self, lst):
        # compute the total number of subsets (2 to the power of the length of the list minus 1, because the empty set is not included)
        num_subsets = 2 ** len(lst) - 1
        # use bit manipulation to generate all subsets
        subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
        return subsets

    def indices_to_tensor(self, indices_list, L, device):
        B = len(indices_list)
        # create a tensor filled with False, dtype=torch.bool
        tensor = torch.zeros((B, L), dtype=torch.int)
        # use scatter_ to set True based on indices
        for i, indices in enumerate(indices_list):
            if indices:  # ensure indices is not empty
                tensor[i].scatter_(0, torch.tensor(indices, dtype=torch.long), 1)
        return tensor.to(device)
    
    def gen_bs_subsets(self, modal_mask, device):
        # modal_mask shape: B, len_modal
        B, modal_num = modal_mask.shape
        # for each sample
        bs_subsets = []
        for b in range(B):
            # use nonzero to acquire the indices of non-zero elements, and convert it from tensor to numpy and then to list
            indices = self.gen_subsets(modal_mask[b].nonzero(as_tuple=True)[0].tolist())
            bs_subsets.append(self.indices_to_tensor(indices, num_modals, device))
        return bs_subsets

    def shapley_value(self, multimodal_contrib, individual_contrib, device):
        values = torch.zeros((num_modals)).to(device)
            
        for idx in range(num_modals):
            # the number of avaliable modalitites
            L = len(self.modal_set)

            # retreive i-th modality individual contribution
            idx_list = [0] * num_modals
            idx_list[idx] = 1
            contrib_idx = individual_contrib[tuple(idx_list)]

            # compute marginal contribution
            # check if the i-th modality is avaliable and not only avaliable independently
            for key in multimodal_contrib:
                if key[idx] == 0:
                    continue
                # (contribution with i-th modality) - (contribution only i-th modality)
                contrib_diff = multimodal_contrib[key] - contrib_idx
                cardinal = sum(key) - 1
                assert cardinal > 0
                weight = (math.factorial(cardinal) * math.factorial(L-cardinal-1) / math.factorial(L)) # Weight = |S|!(n-|S|-1)!/n!
                values[idx] += weight * contrib_diff
        
            # Add the term corresponding to the empty set
            values[idx] += contrib_idx / L
        return values

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
