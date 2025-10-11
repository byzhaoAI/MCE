
import math
import torch
import torch.nn as nn
import os
import json
from collections import OrderedDict
import torch.nn.functional as F
from models.base_model import BaseModel
from models.networks.fc import FcEncoder
from models.networks.lstm import LSTMEncoder
from models.networks.textcnn import TextCNN
from models.transformer import Transformer
from models.networks.classifier import FcClassifier
from models.networks.autoencoder import ResidualAE
from models.utt_fusion_model import UttFusionModel
from .utils.config import OptConfig

num_modals = 3


class MCEModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--input_dim_a', type=int, default=130, help='acoustic input dim')
        parser.add_argument('--input_dim_l', type=int, default=1024, help='lexical input dim')
        parser.add_argument('--input_dim_v', type=int, default=384, help='lexical input dim')
        parser.add_argument('--embd_size_a', default=128, type=int, help='audio model embedding size')
        parser.add_argument('--embd_size_l', default=128, type=int, help='text model embedding size')
        parser.add_argument('--embd_size_v', default=128, type=int, help='visual model embedding size')
        parser.add_argument('--embd_method_a', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='audio embedding method,last,mean or atten')
        parser.add_argument('--embd_method_v', default='maxpool', type=str, choices=['last', 'maxpool', 'attention'], \
            help='visual embedding method,last,mean or atten')
        parser.add_argument('--AE_layers', type=str, default='128,64,32', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--n_blocks', type=int, default=3, help='number of AE blocks')
        parser.add_argument('--cls_layers', type=str, default='128,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--dropout_rate', type=float, default=0.3, help='rate of dropout')
        parser.add_argument('--bn', action='store_true', help='if specified, use bn layers in FC')
        parser.add_argument('--pretrained_path', type=str, help='where to load pretrained encoder network')
        parser.add_argument('--ce_weight', type=float, default=1.0, help='weight of ce loss')
        parser.add_argument('--mse_weight', type=float, default=1.0, help='weight of mse loss')
        parser.add_argument('--cycle_weight', type=float, default=1.0, help='weight of cycle loss')
        parser.add_argument('--rec_weight', type=float, default=1.0, help='weight of reconstruction loss')
        parser.add_argument('--sep_weight', type=float, default=1.0, help='weight of single modality loss')
        parser.add_argument('--sub_weight', type=float, default=1.0, help='weight of sub fuse loss')
        parser.add_argument('--share_weight', action='store_true', help='share weight of forward and backward autoencoders')
        parser.add_argument('--shap_weight', type=float, default=1.0, help='weight of shapley loss')
        parser.add_argument('--use_a', action='store_true', help='use factor a')
        parser.add_argument('--use_b', action='store_true', help='use factor b')
        return parser
    
    def update_unimodal(self, model_dict):
        self.uni_A = model_dict['A']
        self.uni_L = model_dict['L']
        self.uni_V = model_dict['V']
        for network in [self.uni_A, self.uni_L, self.uni_V]:
            for name, param in network['encoder'].named_parameters():
                param.requires_grad = False
            for name, param in network['classifier'].named_parameters():
                param.requires_grad = False
            network['encoder'].to(self.device).eval()
            network['classifier'].to(self.device).eval()

    def unimodal_forward(self, x, modal_name):
        if modal_name == 'A':
            encoder, classifier = self.uni_A['encoder'], self.uni_A['classifier']
        elif modal_name == 'L':
            encoder, classifier = self.uni_L['encoder'], self.uni_L['classifier']
        elif modal_name == 'V':
            encoder, classifier = self.uni_V['encoder'], self.uni_V['classifier']
        else: raise
        with torch.no_grad():
            output = encoder(x)
            output, _ = classifier(output)
        return output

    def __init__(self, opt):
        """Initialize the LSTM autoencoder class
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt)
        # our expriment is on 10 fold setting, teacher is on 5 fold setting, the train set should match
        self.loss_names = ['CE', 'rec', 'shap', 'sep', 'subfuse']
        self.model_names = ['A', 'V', 'L', 'Inter', 'Cls', 'Cls_a', 'Cls_v', 'Cls_l']
        self.modal_set = [(1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,1,1)]

        # acoustic model
        self.netA = LSTMEncoder(opt.input_dim_a, opt.embd_size_a, embd_method=opt.embd_method_a)
        # lexical model
        self.netL = TextCNN(opt.input_dim_l, opt.embd_size_l)
        # visual model
        self.netV = LSTMEncoder(opt.input_dim_v, opt.embd_size_v, opt.embd_method_v)

        # cross attention
        assert opt.embd_size_a == opt.embd_size_l == opt.embd_size_v
        self.netA_pos = nn.Parameter(torch.zeros(1, opt.embd_size_a))
        self.netL_pos = nn.Parameter(torch.zeros(1, opt.embd_size_l))
        self.netV_pos = nn.Parameter(torch.zeros(1, opt.embd_size_v))
        self.netInter = Transformer(opt.embd_size_a, mlp_dim=2048)

        # classifier model
        cls_layers = list(map(lambda x: int(x), opt.cls_layers.split(',')))
        self.netCls = FcClassifier(opt.embd_size_a + opt.embd_size_l + opt.embd_size_v, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_a = FcClassifier(opt.embd_size_a, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_l = FcClassifier(opt.embd_size_l, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)
        self.netCls_v = FcClassifier(opt.embd_size_v, cls_layers, output_dim=opt.output_dim, dropout=opt.dropout_rate, use_bn=opt.bn)


        if self.isTrain:
            self.criterion_ce = torch.nn.CrossEntropyLoss()
            self.criterion_abs = torch.nn.L1Loss()
            self.criterion_mse = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            # specials = ['A_pos', 'L_pos', 'V_pos']
            params = []
            for net in self.model_names:
                # if net not in specials:
                params.append({'params': getattr(self, 'net'+net).parameters()})
            if 'A_pos' in self.model_names:
                params.append({'params': self.netA_pos})
            if 'L_pos' in self.model_names:
                params.append({'params': self.netL_pos})
            if 'V_pos' in self.model_names:
                params.append({'params': self.netV_pos})
            self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
            self.output_dim = opt.output_dim
            self.ce_weight = opt.ce_weight
            self.mse_weight = opt.mse_weight
            self.cycle_weight = opt.cycle_weight
            self.rec_weight = opt.rec_weight
            self.shap_weight = opt.shap_weight
            self.sep_weight = opt.sep_weight
            self.sub_weight = opt.sub_weight
            self.use_a = opt.use_a
            self.use_b = opt.use_b

        # modify save_dir
        self.save_dir = os.path.join(self.save_dir, str(opt.cvNo))
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
    
    def post_process(self):
        pass
        
    def load_from_opt_record(self, file_path):
        opt_content = json.load(open(file_path, 'r'))
        opt = OptConfig()
        opt.load(opt_content)
        return opt

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        """
        acoustic = input['A_feat'].float().to(self.device)
        lexical = input['L_feat'].float().to(self.device)
        visual = input['V_feat'].float().to(self.device)
        self.label = input['label'].to(self.device)
        self.missing_index = input['missing_index'].long().to(self.device)
        if self.isTrain:
            self.modal_weight = torch.from_numpy(input['modal_weight']).float().to(self.device)
        # A modality
        self.A_miss_index = self.missing_index[:, 0].unsqueeze(1).unsqueeze(2)
        self.A_miss = acoustic * self.A_miss_index
        # self.A_reverse = acoustic * -1 * (self.A_miss_index - 1)
        # L modality
        self.L_miss_index = self.missing_index[:, 2].unsqueeze(1).unsqueeze(2)
        self.L_miss = lexical * self.L_miss_index
        # self.L_reverse = lexical * -1 * (self.L_miss_index - 1)
        # V modality
        self.V_miss_index = self.missing_index[:, 1].unsqueeze(1).unsqueeze(2)
        self.V_miss = visual * self.V_miss_index
        # self.V_reverse = visual * -1 * (self.V_miss_index - 1)

    def pretrain_unimodal(self, unimodal):
        # get utt level representattion
        if unimodal == 'A':
            need_train = True if self.missing_index[:, 0].sum() > 0 else False
            self.feat_A_r = self.netA(self.A_miss[self.missing_index[:, 0].bool()])
            self.logits_a, _ = self.netCls_a(self.feat_A_r)
            loss = self.criterion_ce(self.logits_a, self.label[self.missing_index[:, 0].bool()])
        elif unimodal == 'L':
            need_train = True if self.missing_index[:, 2].sum() > 0 else False
            self.feat_L_r = self.netL(self.L_miss[self.missing_index[:, 2].bool()])
            self.logits_l, _ = self.netCls_l(self.feat_L_r)
            loss = self.criterion_ce(self.logits_l, self.label[self.missing_index[:, 2].bool()])
        elif unimodal == 'V':
            need_train = True if self.missing_index[:, 1].sum() > 0 else False
            self.feat_V_r = self.netV(self.V_miss[self.missing_index[:, 1].bool()])
            self.logits_v, _ = self.netCls_v(self.feat_V_r)
            loss = self.criterion_ce(self.logits_v, self.label[self.missing_index[:, 1].bool()])

        if need_train:
            self.loss_CE = loss
            self.optimizer.zero_grad()  
            self.loss_CE.backward()
            self.optimizer.step()
            for model in self.model_names:
                torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 1.0)
        else:
            self.loss_CE = 0
        self.loss_rec = 0
        self.loss_shap = 0
        self.loss_sep = 0
        self.loss_subfuse = 0

        if unimodal == 'A':
            modal_model_dict = {'encoder': self.netA, 'classifier': self.netCls_a}
        elif unimodal == 'L':
            modal_model_dict = {'encoder': self.netL, 'classifier': self.netCls_l}
        elif unimodal == 'V':
            modal_model_dict = {'encoder': self.netV, 'classifier': self.netCls_v}
        return need_train, modal_model_dict

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        # get utt level representattion
        self.feat_A_miss = self.netA(self.A_miss)
        self.feat_L_miss = self.netL(self.L_miss)
        self.feat_V_miss = self.netV(self.V_miss)

        self.feat_fusion_miss = torch.stack([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=1)
        multimodal_pos = torch.stack((self.netA_pos, self.netL_pos, self.netV_pos), dim=1)
        self.feat_fusion_rec = self.netInter(self.feat_fusion_miss, multimodal_pos)

        # fusion miss
        # self.feat_fusion_miss = torch.cat([self.feat_A_miss, self.feat_L_miss, self.feat_V_miss], dim=-1)
        self.feat_A_r, self.feat_L_r, self.feat_V_r = self.feat_fusion_rec[:, 0], self.feat_fusion_rec[:, 1], self.feat_fusion_rec[:, 2]
        self.feat_fusion_r = torch.cat([self.feat_A_r, self.feat_L_r, self.feat_V_r], dim=-1)
        self.logits, _ = self.netCls(self.feat_fusion_r)
        self.pred = F.softmax(self.logits, dim=-1)

        # if self.isTrain:
            

    def backward(self):
        """Calculate the loss for back propagation"""
        B = self.logits.shape[0]
        multimodal_pos = torch.stack((self.netA_pos, self.netL_pos, self.netV_pos), dim=1)
        
        self.logits_a, _ = self.netCls_a(self.feat_A_r)
        self.logits_l, _ = self.netCls_l(self.feat_L_r)
        self.logits_v, _ = self.netCls_v(self.feat_V_r)

        sep_loss = torch.zeros(3).float().to(self.device)
        sep_loss[0] = self.criterion_ce(self.logits_a*self.missing_index[:,0:1], self.label*self.missing_index[:,0]) #*self.ce_weight
        sep_loss[1] = self.criterion_ce(self.logits_l*self.missing_index[:,1:2], self.label*self.missing_index[:,1]) #*self.ce_weight
        sep_loss[2] = self.criterion_ce(self.logits_v*self.missing_index[:,2:3], self.label*self.missing_index[:,2]) #*self.ce_weight

        # subset loss
        rec_loss = torch.zeros(B,3).float().to(self.device)
        subfuse_loss = torch.zeros(B,1).float().to(self.device)
        
        # individual prediction
        # [B]
        # a_pred = F.softmax(self.logits_a, dim=-1).argmax(dim=1)
        # l_pred = F.softmax(self.logits_l, dim=-1).argmax(dim=1)
        # v_pred = F.softmax(self.logits_v, dim=-1).argmax(dim=1)
        a_pred = F.softmax(self.unimodal_forward(self.A_miss, 'A'), dim=-1).argmax(dim=1)
        l_pred = F.softmax(self.unimodal_forward(self.L_miss, 'L'), dim=-1).argmax(dim=1)
        v_pred = F.softmax(self.unimodal_forward(self.V_miss, 'V'), dim=-1).argmax(dim=1)

        # sample level [B, num_modal] ×
        # batch level [num_modal] √
        independent_contrib = torch.stack([
            ((a_pred == self.label) * self.missing_index[:, 0]).sum().float(),
            ((l_pred == self.label) * self.missing_index[:, 1]).sum().float(),
            ((v_pred == self.label) * self.missing_index[:, 2]).sum().float()
        ], dim=0)

        bs_subsets = self.gen_bs_subsets(self.missing_index, self.device)
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

            rec_loss_a, rec_loss_l, rec_loss_v = torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device), torch.zeros(1).float().to(self.device)
            for _subset in subsets:
                if torch.equal(_subset, self.missing_index[b].int()):
                    fused_pred = self.pred[b:b+1]
                else:
                    a_f = self.feat_A_r[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat_A_r.shape[1])).to(self.device)
                    l_f = self.feat_L_r[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat_L_r.shape[1])).to(self.device)
                    v_f = self.feat_V_r[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat_V_r.shape[1])).to(self.device)

                    f_rec = self.netInter(torch.stack((a_f, l_f, v_f), dim=1), multimodal_pos)
                    a_f_rec, l_f_rec, v_f_rec = f_rec[:, 0], f_rec[:, 1], f_rec[:, 2]
                    # Decoder
                    fused_pred, _ = self.netCls(torch.cat([a_f_rec, l_f_rec, v_f_rec], dim=-1))
                    # compute subset fused loss
                    subfuse_loss[b:b+1] += self.criterion_ce(fused_pred, self.label[b:b+1]) #*self.ce_weight
                    
                    # compute auxiliary completion loss
                    # rec_mask&_modal_weight shape: (1,modal_num)
                    # rec_mask = (self.missing_index[b] - _subset).view(1,-1)
                    # _modal_weight = self.modal_weight.view(1,-1)
                    # rec_loss[b] += F.l1_loss(
                    #     torch.stack([self.feat_A_r[b:b+1], self.feat_L_r[b:b+1], self.feat_V_r[b:b+1]], dim=1) * _modal_weight * rec_mask,
                    #     torch.stack([a_f_rec, l_f_rec, v_f_rec], dim=1) * _modal_weight * rec_mask
                    # )
                    rec_mask = (self.missing_index[b] - _subset)
                    rec_loss_a += F.l1_loss(self.feat_A_r[b:b+1], a_f_rec) * rec_mask[0]
                    rec_loss_l += F.l1_loss(self.feat_L_r[b:b+1], l_f_rec) * rec_mask[1]
                    rec_loss_v += F.l1_loss(self.feat_V_r[b:b+1], v_f_rec) * rec_mask[2]
                
                # 计算两个张量中元素相等的位置，并求和得到相等元素的总数
                pred = F.softmax(fused_pred, dim=-1).argmax(dim=1)
                if (_subset).sum() > 1:
                    bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
                else:
                    bs_individual_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
            rec_loss[b] = torch.cat([rec_loss_a, rec_loss_l, rec_loss_v], dim=0)

        # torch.Size([num_modal])
        bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)
        shap_mask = (bs_shap_value < independent_contrib).int().to(self.device)
        shap_loss = (independent_contrib - bs_shap_value) * shap_mask / self.missing_index.sum(dim=0)

        # sample_mod_count&sample_exist_mask shape (b, 1)
        sample_subset_count = 2**self.missing_index.sum(dim=1, keepdim=True) - 1
        sample_exist_mask = (sample_subset_count > 0).float()
        sample_mod_count = sample_subset_count * sample_exist_mask + (1 - sample_exist_mask)
        
        # torch.Size([128, 1]) torch.Size([2]) torch.Size([128, 2]) torch.Size([128, 1])
        # print(rec_loss.shape, shap_loss.shape, sep_loss.shape, subfuse_loss.shape)

        self.loss_CE = self.criterion_ce(self.logits, self.label) #*self.ce_weight

        if self.use_a:
            rec_loss = rec_loss * self.modal_weight.view(1,-1)
            sep_loss = sep_loss * self.modal_weight
        if self.use_b:
            rec_loss = rec_loss * shap_loss.view(1,-1)
            sep_loss = sep_loss * shap_loss
        
        # rec_loss = rec_loss * shap_loss.view(1,-1)
        self.loss_rec = (rec_loss / sample_mod_count).sum() / sample_exist_mask.sum()

        self.loss_shap = shap_loss.sum()
        # sep_loss = sep_loss * shap_loss

        self.loss_sep = sep_loss.mean()
        
        self.loss_subfuse = (subfuse_loss / sample_mod_count).sum() / sample_exist_mask.sum()

        # self.loss_rec = 0
        # self.loss_shap = 0
        # self.loss_sep = 0
        # self.loss_subfuse = 0

        # loss = self.loss_CE + self.loss_rec*self.rec_weight \
        #     + self.loss_shap*self.shap_weight \
        #     + self.loss_sep*self.sep_weight \
        #     + self.loss_subfuse*self.sub_weight
        loss = self.loss_CE \
            + self.loss_rec * self.rec_weight \
            + self.loss_sep * self.sep_weight \
            + self.loss_subfuse * self.sub_weight

        loss.backward()
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 1.0)
        # print(self.missing_index.sum(dim=0), self.missing_index.sum(dim=1), independent_contrib, bs_individual_contrib, bs_multimodal_contrib, bs_shap_value, shap_loss)
        self.loggers = [self.missing_index.sum(dim=0), bs_individual_contrib, bs_multimodal_contrib, independent_contrib, bs_shap_value, shap_mask, shap_loss]

    def cal_shap(self):
        """Calculate the loss for back propagation"""
        B = self.logits.shape[0]
        multimodal_pos = torch.stack((self.netA_pos, self.netL_pos, self.netV_pos), dim=1)
        
        self.logits_a, _ = self.netCls_a(self.feat_A_r)
        self.logits_l, _ = self.netCls_l(self.feat_L_r)
        self.logits_v, _ = self.netCls_v(self.feat_V_r)
        
        # individual prediction
        # [B]
        # a_pred = F.softmax(self.logits_a, dim=-1).argmax(dim=1)
        # l_pred = F.softmax(self.logits_l, dim=-1).argmax(dim=1)
        # v_pred = F.softmax(self.logits_v, dim=-1).argmax(dim=1)
        a_pred = F.softmax(self.unimodal_forward(self.A_miss, 'A'), dim=-1).argmax(dim=1)
        l_pred = F.softmax(self.unimodal_forward(self.L_miss, 'L'), dim=-1).argmax(dim=1)
        v_pred = F.softmax(self.unimodal_forward(self.V_miss, 'V'), dim=-1).argmax(dim=1)

        # sample level [B, num_modal] ×
        # batch level [num_modal] √
        self.independent_contrib = torch.stack([
            ((a_pred == self.label) * self.missing_index[:, 0]).sum().float(),
            ((l_pred == self.label) * self.missing_index[:, 1]).sum().float(),
            ((v_pred == self.label) * self.missing_index[:, 2]).sum().float()
        ], dim=0)

        bs_subsets = self.gen_bs_subsets(self.missing_index, self.device)
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
                if torch.equal(_subset, self.missing_index[b].int()):
                    fused_pred = self.pred[b:b+1]
                else:
                    a_f = self.feat_A_r[b:b+1] if _subset[0] == 1 else torch.zeros((1, self.feat_A_r.shape[1])).to(self.device)
                    l_f = self.feat_L_r[b:b+1] if _subset[1] == 1 else torch.zeros((1, self.feat_L_r.shape[1])).to(self.device)
                    v_f = self.feat_V_r[b:b+1] if _subset[2] == 1 else torch.zeros((1, self.feat_V_r.shape[1])).to(self.device)

                    f_rec = self.netInter(torch.stack((a_f, l_f, v_f), dim=1), multimodal_pos)
                    a_f_rec, l_f_rec, v_f_rec = f_rec[:, 0], f_rec[:, 1], f_rec[:, 2]
                    # Decoder
                    fused_pred, _ = self.netCls(torch.cat([a_f_rec, l_f_rec, v_f_rec], dim=-1))
                
                # 计算两个张量中元素相等的位置，并求和得到相等元素的总数
                pred = F.softmax(fused_pred, dim=-1).argmax(dim=1)
                if (_subset).sum() > 1:
                    bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
                else:
                    bs_individual_contrib[tuple(_subset.cpu().numpy())] += (pred == self.label[b:b+1]).sum().float()
        
        # torch.Size([num_modal])
        self.bs_shap_value = self.shapley_value(bs_multimodal_contrib, bs_individual_contrib, self.device)
        
        
    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()   
        # backward
        self.optimizer.zero_grad()  
        self.backward()            
        self.optimizer.step()

    def gen_subsets(self, lst):
        # 计算子集的总数（2的列表长度次方减去1，因为不包括空集）
        num_subsets = 2 ** len(lst) - 1
        # 使用位运算生成所有子集  
        subsets = [[lst[i] for i, subset_bit in enumerate(bin(subset)[2:].zfill(len(lst))) if subset_bit == '1'] for subset in range(1, num_subsets + 1)]
        return subsets

    def indices_to_tensor(self, indices_list, L, device):
        B = len(indices_list)
        # 创建一个全False的tensor，dtype为torch.bool
        tensor = torch.zeros((B, L), dtype=torch.int)
        # 使用scatter_方法根据索引设置True
        for i, indices in enumerate(indices_list):
            if indices:  # 确保indices不是空列表
                tensor[i].scatter_(0, torch.tensor(indices, dtype=torch.long), 1)
        return tensor.to(device)
    
    def gen_bs_subsets(self, modal_mask, device):
        # modal_mask shape: B, len_modal
        B, modal_num = modal_mask.shape
        # for each sample
        bs_subsets = []
        for b in range(B):
            # 使用nonzero获取非零元素的索引，并将其从tensor转为numpy然后转为list
            indices = self.gen_subsets(modal_mask[b].nonzero(as_tuple=True)[0].tolist())
            bs_subsets.append(self.indices_to_tensor(indices, num_modals, device))
        return bs_subsets

    def shapley_value(self, multimodal_contrib, individual_contrib, device):
        values = torch.zeros((num_modals)).to(device)
            
        for idx in range(num_modals):
            # the number of avaliable modalitites
            L = torch.sum((torch.sum(self.missing_index, dim=0) > 0).float())

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
