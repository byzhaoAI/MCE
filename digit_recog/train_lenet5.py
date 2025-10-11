import os
import time
import math
import datetime
import argparse
import os.path as path
import numpy as np

import torch
# 禁用cuDNN以排除问题
# torch.backends.cudnn.enabled = False

import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from models.modules import LeNet5
# from models.soundlenet5_pii import SoundLenet5
from models.loss import KDFeatureLoss, KDFeatureLossTwo, KDLossAlignTwo

from dataset.meta_training_dataset import MetaTrSouMNIST
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
    parser.add_argument('--checkpoint', default='save/per50/', type=str, help='checkpoint directory')
    parser.add_argument('--load_path', default='', type=str, help='load model from this path')
    parser.add_argument('--load_step', default=False, type=bool, help='load step')
    parser.add_argument('--load_optimizer', default=False, type=bool, help='load optimizer')
    
    parser.add_argument('--only_image_test', default=False, type=bool, help='whether to test only with image')

    # model related parm
    parser.add_argument('-f', '--freeze', default = False, type = bool, help='freeze adjacent matrix' )
    parser.add_argument('--freeze_encoder', default = False, type = bool, help='freeze encoders' )    
    parser.add_argument('-b', '--batch_size', default = 64, type = int, help='batch size' )
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

    args = parser.parse_args()

    return args

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag

def main(args):
    cudnn.benchmark = True
    os.environ["CUDA_VISIBLE_DEVICES"] = args.vis_device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # train step dataset
    if args.per_class_num < 105:
        meta_train_dataset = MetaTrSouMNIST(img_root=args.image_root, sound_root=args.sound_root, per_class_num=args.per_class_num, meta_split='mtr')
        meta_train_loader = DataLoader(meta_train_dataset, batch_size = args.batch_size, shuffle = True, num_workers=4, pin_memory=True)
        print('train data size:', len(meta_train_dataset))

    if args.per_class_num > 0:
        meta_val_dataset = MetaTrSouMNIST(img_root=args.image_root, sound_root=args.sound_root, per_class_num=args.per_class_num, meta_split='mval')
        meta_val_loader = DataLoader(meta_val_dataset, batch_size = args.batch_size, shuffle = True, num_workers=4, pin_memory=True)
        print('val data size:', len(meta_val_dataset))
    
    # test dataset
    meta_test_dataset = SoundMNIST(img_root=args.image_root,sound_root=args.sound_root, per_class_num=args.per_class_num, train=False)
    meta_test_loader = DataLoader(meta_test_dataset, batch_size = args.batch_size, shuffle = False, num_workers=4, pin_memory=True)
    print('test data size:',len(meta_test_dataset))


    #creat model 
    print('==> model creating....')
    global_step = 1

    image_sound_extractor = LeNet5()
    # image_sound_extractor = SoundLenet5(freeze=args.freeze, freeze_encoder=args.freeze_encoder)
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
        if args.per_class_num == 0:
            meta_batch = next(iter(meta_train_loader))
        elif args.per_class_num < 105:
            if np.random.rand() >= 0.5:
                meta_batch = next(iter(meta_train_loader))
            else:
                meta_batch = next(iter(meta_val_loader))
        else:
            meta_batch = next(iter(meta_val_loader))
        '''
        # Train for one iteration
        if np.random.rand() >= args.per_class_num/150:
            meta_batch = next(iter(meta_train_loader))
        else:
            meta_batch = next(iter(meta_val_loader))
        '''

        # 一阶段，使用全模态学习邻接矩阵
        global_step = run_train(args, meta_batch, image_sound_extractor, criterion_meta_train, optimizer_image_sound, device, global_step, args.iterations)
        scheduler_image_sound.step()

        # save model checkpoint
        if np.mod(global_step, args.eval_interval)==0:
            saverloader.save(args.checkpoint, optimizer_image_sound, image_sound_extractor, global_step, keep_latest=-1)
            # print("Adjacent matrix is: ", image_sound_extractor.inter.gc.adj)

        if (global_step) % args.eval_interval == 0 or global_step >= 1200:
            # test_acc = 0.
            # for i in range(30):
            #     test_acc += run_eval(meta_test_loader, image_sound_extractor, device)
            # test_acc = test_acc / 30
            image_sound_extractor.eval()
            test_image_acc = run_eval(meta_test_loader, image_sound_extractor, True, device)
            print('Iteration:[{}/{}], image-only:{},  test acc:{:.6f}'.format(global_step, args.iterations, True, test_image_acc))
            if test_image_acc > best_image_acc:
                print('Best image acc update from', best_image_acc, ' to ', test_image_acc, ' at ', global_step)
                best_image_acc = test_image_acc
                saverloader.save(args.checkpoint + 'best_image', optimizer_image_sound, image_sound_extractor, global_step, keep_latest=1)

            image_sound_extractor.train()
        
        global_step += 1
    

def run_train(args, batch, image_sound_extractor, criterion, optimizer_image_sound, device, iterate, total_iterate):
    ''' train one epoch'''
    # torch.set_grad_enabled(True)
    batch_size = batch[0].shape[0]

    # sampled from both image and sound modality
    if len(batch) == 3:
        image = batch[0].to(device)
        sound = batch[1].to(device)
        label = batch[2].to(device)
        image_only = False
    else:
        image = batch[0].to(device)
        sound = None
        label = batch[1].to(device)
        image_only = True

    # meta-training 
    loss_meta_train = 0.
    loss_meta_val = 0.

    optimizer_image_sound.zero_grad()

    predictions, _ = image_sound_extractor(image)
    loss = criterion(predictions, label)

    torch.autograd.set_detect_anomaly(True)

    loss.backward()
    optimizer_image_sound.step()
    torch.cuda.empty_cache()

    print('Iteration [{}/{}], Loss: {:.4f}' .format(iterate, total_iterate, loss.item()))
    return iterate


def run_eval(test_loader, image_sound_extractor, image_only, device):
    # Switch to evaluate mode
    # torch.set_grad_enabled(False)
    correct = 0
    total = 0

    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            images = batch[0].to(device)
            sounds = batch[1].to(device)
            labels = batch[2].to(device)

            outputs, _ = image_sound_extractor(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total   
    
    return test_acc


if __name__ == '__main__':
    main(parse_args())