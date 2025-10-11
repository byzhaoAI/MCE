import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np
import copy

import torch
# 禁用cuDNN以排除问题
torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

# from models.soundlenet5_exgat import SoundLenet5  # can work
from models.soundlenet5_shap_new import SoundLenet5     # can work
# from models.soundlenet5_pii import SoundLenet5
from models.loss import KDFeatureLoss, KDFeatureLossTwo, KDLossAlignTwo

from dataset.partial_training_dataset import MetaTrSouMNIST
# from dataset.meta_training_dataset import MetaTrSouMNIST
# from dataset.meta_testing_dataset import MetaTeSouMNIST
from dataset.soundmnist import SoundMNIST
from dataset.mnist import MNIST

from utils.misc import AverageMeter, AvgF1, save_ckpt, save_ckpt_inferNet, save_ckpt_classifier

from metann import ProtoModule
import saverloader


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch training script')
    parser.add_argument('-i', '--image_root', default = 'soundmnist/mnist/', type = str, help='data root' )
    parser.add_argument('-s', '--sound_root', default = 'soundmnist/sound_450/', type = str, help='data root' )
    parser.add_argument('--checkpoint', default='save/', type=str, help='checkpoint directory')
    parser.add_argument('--load_path', default='', type=str, help='load model from this path')
    parser.add_argument('--load_step', default=False, type=bool, help='load step')
    parser.add_argument('--load_optimizer', default=False, type=bool, help='load optimizer')
    
    parser.add_argument('--only_image_test', default=False, type=bool, help='whether to test only with image')

    # model related parm
    parser.add_argument('-f', '--freeze', default = False, type = bool, help='freeze adjacent matrix' )
    parser.add_argument('--freeze_encoder', default = False, type = bool, help='freeze encoders' )    
    parser.add_argument('-b', '--batch_size', default = 128, type = int, help='batch size' )
    parser.add_argument('--per_class_num', default = 51, type = int, help='per_class_num' )
    # 15 * 10 = 150 sound data available, total 1500 sound data
    # 5% -> 6
    # 10% -> 12
    # 15% -> 15 -
    # 20% -> 21 -
    # 50% -> 51 -
    # 70% -> 75 -
    # 100% -> 105 -
    parser.add_argument('--iterations', default = 4000 , type = int, help='num of epoch' )
    parser.add_argument('--eval_interval', default = 50, type = int, help='eval interval' )
    parser.add_argument('--lr', default = 1e-4, type = float, help='initial learning rate' )
    parser.add_argument('--vis_device', default='0', type=str, help='set visiable device')

    parser.add_argument('--rec_weight', default = 1, type = float, help='' )
    parser.add_argument('--shap_weight', default = 1, type = float, help='' )
    parser.add_argument('--sub_weight', default = 1, type = float, help='' )
    parser.add_argument('--sep_weight', default = 1, type = float, help='' )

    args = parser.parse_args()

    return args

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    for exist_rate in range(1, 11):
        per_class_num = int(exist_rate * 10.5)
        if per_class_num % 3 != 0:
            per_class_num = (per_class_num // 3 + 1) * 3
        args.per_class_num = per_class_num
        checkpoint_path = args.checkpoint + 'per' + str(exist_rate*10) + '/'
        print('per_class_num:', args.per_class_num)
        
        # train step dataset
        meta_train_dataset = MetaTrSouMNIST(img_root=args.image_root, sound_root=args.sound_root, per_class_num=args.per_class_num, meta_split='mtr')
        meta_train_loader = DataLoader(meta_train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=4, pin_memory=True)
        print('train data size:', len(meta_train_dataset))

        # test dataset
        meta_test_dataset = SoundMNIST(img_root=args.image_root,sound_root=args.sound_root, per_class_num=args.per_class_num, train=False)
        meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=4, pin_memory=True)
        print('test data size:',len(meta_test_dataset))

        modal_weight = torch.tensor([
            1, 
            105 / args.per_class_num
        ]).float().to(device)
        print('modal_weight:', modal_weight)

        #creat model 
        print('==> model creating....')
        global_step = 1

        image_sound_extractor = SoundLenet5(freeze=args.freeze, freeze_encoder=args.freeze_encoder)
        image_sound_extractor = image_sound_extractor.to(device) # multimodal fusion model
        parameters = list(image_sound_extractor.parameters())
        print("==> Total parameters (reference): {:.2f}M".format(sum(p.numel() for p in image_sound_extractor.parameters()) / 1000000.0))

        if args.load_path != '':
            if args.load_step and args.load_optimizer:
                global_step = saverloader.load(args.load_path, image_sound_extractor, optimizer, ignore_load=None)
            elif args.load_step:
                global_step = saverloader.load(args.load_path, image_sound_extractor, ignore_load=None)
            else:
                _ = saverloader.load(args.load_path, image_sound_extractor, ignore_load=None)
                global_step = 1
        # requires_grad(parameters, True)
        image_sound_extractor.train()

        image_dict, sound_dict = None, None
        unimodal_path = 'save/unimodal_ckpts/best'
        model_pths = os.listdir(unimodal_path)
        print(model_pths)
        for model_pth in model_pths:
            if 'image' in model_pth:
                image_dict = torch.load(path.join(unimodal_path, model_pth), map_location='cpu')
                print('==> load image extractor from: ', path.join(unimodal_path, model_pth))
            elif 'sound' in model_pth and ('_'+str(exist_rate)+'_') in model_pth:
                sound_dict = torch.load(path.join(unimodal_path, model_pth), map_location='cpu')
                print('==> load sound extractor from: ', path.join(unimodal_path, model_pth))
        assert image_dict is not None, 'image extractor is None'
        assert sound_dict is not None, 'sound extractor is None'
        
        image_model = {
            'img_enc': copy.deepcopy(image_sound_extractor.img_extractor),
            'aligner': copy.deepcopy(image_sound_extractor.aligner),
            'img_dec': copy.deepcopy(image_sound_extractor.img_dec),
        }
        sound_model = {
            'sound_enc': copy.deepcopy(image_sound_extractor.sound_extractor),
            'sound_dec': copy.deepcopy(image_sound_extractor.sound_dec),
        }
        image_model['img_enc'].load_state_dict(image_dict['img_enc'], strict=True)
        image_model['aligner'].load_state_dict(image_dict['aligner'], strict=True)
        image_model['img_dec'].load_state_dict(image_dict['img_dec'], strict=True)
        sound_model['sound_enc'].load_state_dict(sound_dict['sound_enc'], strict=True)
        sound_model['sound_dec'].load_state_dict(sound_dict['sound_dec'], strict=True)
        model_dict = {
            'image': image_model,
            'sound': sound_model
        }
        image_sound_extractor.unimodal_update(model_dict, device)
        print('==> model has been created')

        criterion_meta_train = nn.CrossEntropyLoss().to(device)
        criterion_meta_val = KDLossAlignTwo(alpha = 0.01, beta = 0.01).to(device)

        optimizer_image_sound = torch.optim.Adam(filter(lambda p: p.requires_grad, image_sound_extractor.parameters()), lr = args.lr, weight_decay = 1e-4)
        scheduler_image_sound = torch.optim.lr_scheduler.StepLR(optimizer_image_sound, step_size = 5000, gamma = 0.1)  

        print('start training')
        best_image_acc = -999
        best_full_acc = -999
        for iterate in range(args.iterations):
            # Train for one iteration
            meta_batch = next(iter(meta_train_loader))
            '''
            # Train for one iteration
            if np.random.rand() >= args.per_class_num/150:
                meta_batch = next(iter(meta_train_loader))
            else:
                meta_batch = next(iter(meta_val_loader))
            '''

            # 一阶段，使用全模态学习邻接矩阵
            global_step = run_train(args, meta_batch, modal_weight, image_sound_extractor, criterion_meta_train, optimizer_image_sound, device, global_step, args.iterations)
            scheduler_image_sound.step()

            # save model checkpoint
            if np.mod(global_step, args.eval_interval)==0:
                saverloader.save(checkpoint_path, optimizer_image_sound, image_sound_extractor, global_step, keep_latest=-1)
                # print("Adjacent matrix is: ", image_sound_extractor.inter.gc.adj)

            if (global_step) % args.eval_interval == 0 or global_step >= 1200:
                # test_acc = 0.
                # for i in range(30):
                #     test_acc += run_eval(meta_test_loader, image_sound_extractor, device)
                # test_acc = test_acc / 30
                image_sound_extractor.eval()
                if args.only_image_test:
                    test_image_acc = run_eval(meta_test_loader, image_sound_extractor, True, device)
                    print('Iteration:[{}/{}], image-only:{},  test acc:{:.6f}'.format(global_step, args.iterations, True, test_image_acc))
                    if test_image_acc > best_image_acc:
                        print('Best image acc update from', best_image_acc, ' to ', test_image_acc, ' at ', global_step)
                        best_image_acc = test_image_acc
                        saverloader.save(checkpoint_path + 'best_image', optimizer_image_sound, image_sound_extractor, global_step, keep_latest=1)
                else:
                    test_image_acc = run_eval(meta_test_loader, image_sound_extractor, True, device)
                    print('Iteration:[{}/{}], image-only:{},  test acc:{:.6f}'.format(global_step, args.iterations, True, test_image_acc))
                    if test_image_acc > best_image_acc:
                        print('Best image acc update from', best_image_acc, ' to ', test_image_acc, ' at ', global_step)
                        best_image_acc = test_image_acc
                        saverloader.save(checkpoint_path + 'best_image', optimizer_image_sound, image_sound_extractor, global_step, keep_latest=1)
                    
                    test_full_acc = run_eval(meta_test_loader, image_sound_extractor, False, device)
                    print('Iteration:[{}/{}], image-only:{},  test acc:{:.6f}'.format(global_step, args.iterations, False, test_full_acc))
                    if test_full_acc > best_full_acc:
                        print('Best full acc update from', best_full_acc, ' to ', test_full_acc, ' at ', global_step)
                        best_full_acc = test_full_acc
                        saverloader.save(checkpoint_path + 'best_full', optimizer_image_sound, image_sound_extractor, global_step, keep_latest=1)

                image_sound_extractor.train()
                image_sound_extractor.reset_eval()
            
            global_step += 1
    

def run_train(args, batch, modal_weight, image_sound_extractor, criterion, optimizer_image_sound, device, iterate, total_iterate):
    ''' train one epoch'''
    # torch.set_grad_enabled(True)
    batch_size = batch[0].shape[0]

    # sampled from both image and sound modality
    image = batch[0].to(device)
    sound = batch[1].to(device)
    mask = batch[2].to(device)
    label = batch[3].to(device)

    # meta-training 
    loss_meta_train = 0.
    loss_meta_val = 0.

    optimizer_image_sound.zero_grad()

    predictions, rec_loss, shap_loss, sep_loss, subfuse_loss = image_sound_extractor(image, sound, mask, label, modal_weight=modal_weight, training=True)
    bs_rec_loss = rec_loss.mean()
    bs_shap_loss = shap_loss.sum()
    bs_sep_loss = sep_loss.mean()
    bs_subfuse_loss = subfuse_loss.mean()
    
    task_loss = criterion(predictions, label)
    loss = task_loss + bs_rec_loss * args.rec_weight \
        + bs_sep_loss * args.sep_weight \
        + bs_subfuse_loss * args.sub_weight
        # + bs_shap_loss * args.shap_weight \

    torch.autograd.set_detect_anomaly(True)

    loss.backward()
    optimizer_image_sound.step()
    torch.cuda.empty_cache()

    print('Iteration [{}/{}], Task: {:.4f}, Rec: {:.4f}, Shap: {:.4f}, Sep: {:.4f}, Subfuse: {:.4f}' .format(
        iterate, total_iterate, task_loss.item(), bs_rec_loss.item(), bs_shap_loss.item(), bs_sep_loss.item(), bs_subfuse_loss.item()
    ))
    return iterate


def run_eval(test_loader, image_sound_extractor, image_only, device):
    # Switch to evaluate mode
    # torch.set_grad_enabled(False)
    correct = 0
    total = 0

    upperbound = torch.zeros(2).float().to(device)
    single_modal = torch.zeros(2).float().to(device)
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch[0].to(device)
            sounds = batch[1].to(device)
            mask = torch.ones((images.shape[0],2)).to(device) if not image_only else torch.tensor([1, 0]*images.shape[0]).to(device)
            labels = batch[2].to(device)

            if image_only:
                sound = torch.zeros(sounds.shape).to(device)
            outputs, img_feature, sound_feature = image_sound_extractor(images, sounds, mask, labels, modal_weight=None, training=False)
            # outputs, img_feature, sound_feature = image_sound_extractor(images, sounds)
            # outputs = image_sound_extractor(images, sounds, image_only)[:,0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            comparison = (predicted == labels)
            correct += (predicted == labels).int().sum().item()

            if not image_only:
                independent_contrib, shap_value = cal_contrib(outputs, images, sounds, img_feature, sound_feature, mask, labels, image_sound_extractor, device)
                upperbound += independent_contrib
                single_modal += shap_value

    if not image_only:
        print('upperbound:', upperbound)
        print('single_modal:', single_modal)
        print('Avg upperbound:', upperbound / 450)
        print('Avg single_modal:', single_modal / 450)

    test_acc = 100 * correct / total   
    
    return test_acc

def cal_contrib(outputs, images, sounds, img_feature, sound_feature, mask, label, image_sound_extractor, device):
    # img_pred = image_sound_extractor.img_dec(img_feature)
    # sound_pred = image_sound_extractor.sound_dec(sound_feature)
    img_pred = image_sound_extractor.unimodal_forward(images, 'image')
    sound_pred = image_sound_extractor.unimodal_forward(sounds, 'sound')
    img_pred = torch.max(img_pred, 1)[1]
    sound_pred = torch.max(sound_pred, 1)[1]

    # sample level [B, num_modal] ×
    # batch level [num_modal] √
    independent_contrib = torch.stack([
        ((img_pred == label) * mask[:, 0]).sum().float(),
        ((sound_pred == label) * mask[:, 1]).sum().float()
    ], dim=0)

    bs_subsets = image_sound_extractor.gen_bs_subsets(mask, device)
    bs_individual_contrib = dict()
    bs_multimodal_contrib = dict()
    for value in image_sound_extractor.modal_set:
        if sum(value) > 1:
            bs_multimodal_contrib[value] = 0
        else:
            bs_individual_contrib[value] = 0
    
    multimodal_pos = torch.cat((image_sound_extractor.image_pos, image_sound_extractor.sound_pos), dim=1)
    # for each sample
    for b in range(outputs.shape[0]):
        # shape: subsets - (len_subsets, len_modal)
        subsets = bs_subsets[b]

        for _subset in subsets:
            if torch.equal(_subset, mask[b].int()):
                fused_pred = outputs[b:b+1]
            else:
                img_f = img_feature[b:b+1] if _subset[0] == 1 else torch.zeros((1, img_feature.shape[1])).to(img_feature.device)
                sound_f = sound_feature[b:b+1] if _subset[1] == 1 else torch.zeros((1, sound_feature.shape[1])).to(sound_feature.device)

                f_rec = image_sound_extractor.inter(torch.stack((img_f, sound_f), dim=1), multimodal_pos)
                img_f_rec, sound_f_rec = torch.chunk(f_rec, 2, dim=1)
                img_f_rec = img_f_rec.squeeze(1)
                sound_f_rec = sound_f_rec.squeeze(1)
                # Decoder
                fused_pred = image_sound_extractor.fuse_dec(torch.cat([img_f_rec, sound_f_rec], dim=1))
            
            # 计算两个张量中元素相等的位置，并求和得到相等元素的总数
            fused_pred = torch.max(fused_pred, 1)[1]
            if (_subset).sum() > 1:
                bs_multimodal_contrib[tuple(_subset.cpu().numpy())] += (fused_pred == label[b:b+1]).sum().float()
            else:
                bs_individual_contrib[tuple(_subset.cpu().numpy())] += (fused_pred == label[b:b+1]).sum().float()
    
    # torch.Size([2])
    shap_value = image_sound_extractor.shapley_value(bs_multimodal_contrib, bs_individual_contrib, device)
    
    return independent_contrib, shap_value


if __name__ == '__main__':
    main(parse_args())