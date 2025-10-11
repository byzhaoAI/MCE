import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from models.modules import LeNet5, SNetPluse
from models.transformer import Transformer

num_modals = 2


class SoundLenet5(nn.Module):
    modal_set = [(1, 0), (0, 1), (1, 1)]

    """batch level shap for Lenet5 Sound"""
    def __init__(self, freeze=False, freeze_encoder=False):
        super(SoundLenet5, self).__init__()

        self.img_extractor = LeNet5()
        self.sound_extractor = SNetPluse()

        # align multimodal features
        self.aligner = nn.Linear(160, 320)

        # inter attention module
        self.image_pos = nn.Parameter(torch.zeros(1, 1, 320))
        self.sound_pos = nn.Parameter(torch.zeros(1, 1, 320))
        self.inter = Transformer()

        self.fuse_dec = nn.Sequential(
            nn.Linear(640, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 10)
        )

        self.img_dec = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 10)
        )

        self.sound_dec = nn.Sequential(
            nn.Linear(320, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(64, 10)
        )

        self.criterion = nn.CrossEntropyLoss()

        self.uni_img = None
        self.uni_sound = None

    def unimodal_update(self, model_dict, device):
        self.uni_img = model_dict['image']
        for key, value in self.uni_img.items():
            for name, param in value.named_parameters():
                param.requires_grad = False
            value.to(device).eval()
        self.uni_sound = model_dict['sound']
        for key, value in self.uni_sound.items():
            for name, param in value.named_parameters():
                param.requires_grad = False
            value.to(device).eval()

    def reset_eval(self):
        """reset the model to eval mode"""
        for key, module in self.uni_img.items():
            module.eval()
        for key, module in self.uni_sound.items():
            module.eval()
    
    def unimodal_forward(self, x, modal_name, is_train=False):
        if not is_train:
            with torch.no_grad():
                if modal_name == 'image':
                    assert self.uni_img is not None, 'Please load unimodal model first'
                    _, feature = self.uni_img['img_enc'](x)
                    feature = feature.flatten(1)
                    feature = self.uni_img['aligner'](feature)
                    pred = self.uni_img['img_dec'](feature)
                elif modal_name == 'sound':
                    assert self.uni_sound is not None, 'Please load unimodal model first'
                    _, feature = self.uni_sound['sound_enc'](x)
                    feature = feature.flatten(1)
                    pred = self.uni_sound['sound_dec'](feature)
                else:
                    raise ValueError('modal_name should be image or sound')
            return pred

        if modal_name == 'image':
            _, feature = self.img_extractor(x)
            feature = feature.flatten(1)
            feature = self.aligner(feature)
            pred = self.img_dec(feature)
            model_dict = {
                'img_enc': self.img_extractor,
                'aligner': self.aligner,
                'img_dec': self.img_dec
            }
        elif modal_name == 'sound':
            _, feature = self.sound_extractor(x)
            feature = feature.flatten(1)
            pred = self.sound_dec(feature)
            model_dict = {
                'sound_enc': self.sound_extractor,
                'sound_dec': self.sound_dec
            }
        else:
            raise ValueError('modal_name should be image or sound')
        
        return pred, model_dict

    def forward(self, image, sound=None, mask=None, label=None, modal_weight=None, training=False):
        B = image.shape[0]
        device = image.device

        # image -> torch.Size([b, 10, 16])
        _, img_feature = self.img_extractor(image)
        img_feature = img_feature.flatten(1)
        # -> [b, 320]
        img_feature = self.aligner(img_feature)
        
        # sound -> torch.Size([b, 20, 16])
        _, sound_feature = self.sound_extractor(sound)
        # -> [b, 320]
        sound_feature = sound_feature.flatten(1)

        # attention reconstruction
        # -> [1, 2, 320]
        multimodal_pos = torch.cat((self.image_pos, self.sound_pos), dim=1)
        fused_feature = self.inter(
            torch.stack((img_feature, sound_feature), dim=1),
            multimodal_pos
        )
        img_feature, sound_feature = torch.chunk(fused_feature, 2, dim=1)
        img_feature = img_feature.squeeze(1)
        sound_feature = sound_feature.squeeze(1)
        x = self.fuse_dec(torch.cat([img_feature, sound_feature], dim=1))

        if training:
            # TODO
            assert mask is not None
            assert label is not None

            img_pred = self.img_dec(img_feature)
            sound_pred = self.sound_dec(sound_feature)

            sep_loss = torch.zeros(B,2).float().to(device)
            # subset loss
            rec_loss = torch.zeros(B,2).float().to(device)
            subfuse_loss = torch.zeros(B,1).float().to(device)

            img_logits = torch.max(self.unimodal_forward(image, 'image'), 1)[1]
            sound_logits = torch.max(self.unimodal_forward(sound, 'sound'), 1)[1]
            # img_logits = torch.max(img_pred, 1)[1]
            # sound_logits = torch.max(sound_pred, 1)[1]

            # sample level [B, num_modal] ×
            # batch level [num_modal] √
            independent_contrib = torch.stack([
                ((img_logits == label) * mask[:, 0]).sum().float(),
                ((sound_logits == label) * mask[:, 1]).sum().float()
            ], dim=0)

            bs_subsets = self.gen_bs_subsets(mask, device)
            bs_individual_contrib = dict()
            bs_multimodal_contrib = dict()
            for value in self.modal_set:
                if sum(value) > 1:
                    bs_multimodal_contrib[value] = 0
                else:
                    bs_individual_contrib[value] = 0
                
            # for each sample
            for b in range(B):
                # shape: subsets - (len_subsets, len_modal)
                subsets = bs_subsets[b]

                for _subset in subsets:
                    if torch.equal(_subset, mask[b].int()):
                        fused_pred = x[b:b+1]
                    else:
                        img_f = img_feature[b:b+1] if _subset[0] == 1 else torch.zeros((1, img_feature.shape[1])).to(img_feature.device)
                        sound_f = sound_feature[b:b+1] if _subset[1] == 1 else torch.zeros((1, sound_feature.shape[1])).to(sound_feature.device)

                        f_rec = self.inter(torch.stack((img_f, sound_f), dim=1), multimodal_pos)
                        img_f_rec, sound_f_rec = torch.chunk(f_rec, 2, dim=1)
                        img_f_rec = img_f_rec.squeeze(1)
                        sound_f_rec = sound_f_rec.squeeze(1)
                        # Decoder
                        fused_pred = self.fuse_dec(torch.cat([img_f_rec, sound_f_rec], dim=1))
                        # compute subset fused loss
                        subfuse_loss[b:b+1] += self.criterion(fused_pred, label[b:b+1])
                        
                        # compute auxiliary completion loss
                        ################################ New ################################
                        # rec_mask = (mask[b] - _subset).view(1,-1, 1)
                        # _modal_weight = modal_weight if modal_weight is not None else torch.ones(mask[b].shape).to(device)
                        # _modal_weight = _modal_weight.view(1,-1,1)
                        # rec_loss[b] += F.l1_loss(
                        #     torch.stack([img_feature[b:b+1], sound_feature[b:b+1]], dim=1) * _modal_weight * rec_mask,
                        #     torch.stack([img_f_rec, sound_f_rec], dim=1) * _modal_weight * rec_mask
                        # )
                        rec_mask = mask[b] - _subset
                        _modal_weight = modal_weight if modal_weight is not None else torch.ones(mask[b].shape).to(device)
                        rec_loss[b][0] = F.l1_loss(img_feature[b:b+1], img_f_rec) * _modal_weight[0] * rec_mask[0]
                        rec_loss[b][1] = F.l1_loss(sound_feature[b:b+1], sound_f_rec) * _modal_weight[1] * rec_mask[1]
                        ################################ New ################################

                    # calculate the positions where the elements of the two tensors are equal, and sum to get the total number of equal elements
                    fused_pred = torch.max(fused_pred, 1)[1]
                    if (_subset).sum() > 1:
                        bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (fused_pred == label[b:b+1]).sum().float()
                    else:
                        bs_individual_contrib[tuple(_subset.cpu().numpy())] += (fused_pred == label[b:b+1]).sum().float()
            
            # torch.Size([2])
            bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, device)
            shap_mask = (bs_shap_value < independent_contrib).int().to(device)
            shap_loss = (independent_contrib - bs_shap_value) * shap_mask / mask.sum(dim=0)
            
            # torch.Size([128, 1]) torch.Size([2]) torch.Size([128, 2]) torch.Size([128, 1])
            # print(rec_loss.shape, shap_loss.shape, sep_loss.shape, subfuse_loss.shape)

            rec_loss = rec_loss * shap_loss
            
            sep_loss[:,0] = self.criterion(img_pred, label)
            sep_loss[:,1] = self.criterion(sound_pred, label)
            sep_loss = sep_loss * modal_weight * shap_loss * mask.float()

            return x, rec_loss / (2**mask.sum(dim=1, keepdim=True)-1), shap_loss, sep_loss, subfuse_loss / (2**mask.sum(dim=1, keepdim=True)-1)
        return x, img_feature, sound_feature

    def gen_subsets(self, lst):
        # calculate the total number of subsets (2 to the power of the list length minus 1, because the empty set is not included)
        num_subsets = 2 ** len(lst) - 1
        # use bit manipulation to generate all subsets
        subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
        return subsets

    def indices_to_tensor(self, indices_list, L, device):
        B = len(indices_list)
        # create a tensor filled with False, dtype is torch.bool
        tensor = torch.zeros((B, L), dtype=torch.int)
        # use scatter_ method to set True according to the indices
        for i, indices in enumerate(indices_list):
            if indices:  # ensure indices is not an empty list
                tensor[i].scatter_(0, torch.tensor(indices, dtype=torch.long), 1)
        return tensor.to(device)
    
    def gen_bs_subsets(self, modal_mask, device):
        # modal_mask shape: B, len_modal
        B, modal_num = modal_mask.shape
        # for each sample
        bs_subsets = []
        for b in range(B):
            # use nonzero to get the indices of non-zero elements, and convert it from tensor to numpy then to list
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
