import os
import time
import argparse
import numpy as np
import saverloader
from fire import Fire
# from imml_nets.segnet_imml import Segnet
from imml_nets.segnet import Segnet
import utils.misc
import utils.improc
import utils.vox
import random
import nuscenesdataset
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# from my_train import run_model

import logging

random.seed(125)
np.random.seed(125)


# the scene centroid is defined wrt a reference camera,
# which is usually random
scene_centroid_x = 0.0
scene_centroid_y = 1.0 # down 1 meter
scene_centroid_z = 0.0

scene_centroid_py = np.array([scene_centroid_x,
                              scene_centroid_y,
                              scene_centroid_z]).reshape([1, 3])
scene_centroid = torch.from_numpy(scene_centroid_py).float()

XMIN, XMAX = -50, 50
ZMIN, ZMAX = -50, 50
YMIN, YMAX = -5, 5
bounds = (XMIN, XMAX, YMIN, YMAX, ZMIN, ZMAX)

Z, Y, X = 200, 8, 200

def requires_grad(parameters, flag=True):
    for p in parameters:
        p.requires_grad = flag
        
def fetch_optimizer(lr, wdecay, epsilon, num_steps, params):
    """ Create the optimizer and learning rate scheduler """
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wdecay, eps=epsilon)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, lr, num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = utils.basic.reduce_masked_mean(loss, valid)
        return loss

def run_model(model, loss_fn, d, device='cuda:0', mask=None, is_training=False, modal_weight=[1.,1.,1.], balanced_weights=[1.,1.,1.]):
    masks, imgs, rots, trans, intrins, pts0, extra0, pts, extra, lrtlist_velo, vislist, tidlist, scorelist, seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g, radar_data, egopose = d

    B0,T,S,C,H,W = imgs.shape
    assert(T==1)

    # eliminate the time dimension
    imgs = imgs[:,0]
    rots = rots[:,0]
    trans = trans[:,0]
    intrins = intrins[:,0]
    pts0 = pts0[:,0]
    extra0 = extra0[:,0]
    pts = pts[:,0]
    extra = extra[:,0]
    lrtlist_velo = lrtlist_velo[:,0]
    vislist = vislist[:,0]
    tidlist = tidlist[:,0]
    scorelist = scorelist[:,0]
    seg_bev_g = seg_bev_g[:,0]
    valid_bev_g = valid_bev_g[:,0]
    center_bev_g = center_bev_g[:,0]
    offset_bev_g = offset_bev_g[:,0]
    radar_data = radar_data[:,0]
    egopose = egopose[:,0]
    
    origin_T_velo0t = egopose.to(device) # B,T,4,4
    lrtlist_velo = lrtlist_velo.to(device)
    scorelist = scorelist.to(device)

    rgb_camXs = imgs.float().to(device)
    rgb_camXs = rgb_camXs - 0.5 # go to -0.5, 0.5

    seg_bev_g = seg_bev_g.to(device)
    valid_bev_g = valid_bev_g.to(device)
    center_bev_g = center_bev_g.to(device)
    offset_bev_g = offset_bev_g.to(device)

    xyz_velo0 = pts.to(device).permute(0, 2, 1)
    rad_data = radar_data.to(device).permute(0, 2, 1) # B, R, 19
    xyz_rad = rad_data[:,:,:3]
    meta_rad = rad_data[:,:,3:]

    B, S, C, H, W = rgb_camXs.shape
    B, V, D = xyz_velo0.shape

    __p = lambda x: utils.basic.pack_seqdim(x, B)
    __u = lambda x: utils.basic.unpack_seqdim(x, B)

    mag = torch.norm(xyz_velo0, dim=2)
    xyz_velo0 = xyz_velo0[:,mag[0]>1]
    xyz_velo0_bak = xyz_velo0.clone()

    intrins_ = __p(intrins)
    pix_T_cams_ = utils.geom.merge_intrinsics(*utils.geom.split_intrinsics(intrins_)).to(device)
    pix_T_cams = __u(pix_T_cams_)

    velo_T_cams = utils.geom.merge_rtlist(rots, trans).to(device)
    cams_T_velo = __u(utils.geom.safe_inverse(__p(velo_T_cams)))
    
    cam0_T_camXs = utils.geom.get_camM_T_camXs(velo_T_cams, ind=0)
    camXs_T_cam0 = __u(utils.geom.safe_inverse(__p(cam0_T_camXs)))
    cam0_T_camXs_ = __p(cam0_T_camXs)
    camXs_T_cam0_ = __p(camXs_T_cam0)
    
    xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_velo0)
    rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_rad)

    lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:,0], lrtlist_velo)

    vox_util = utils.vox.Vox_util(
        Z, Y, X,
        scene_centroid=scene_centroid.to(device),
        bounds=bounds,
        assert_cube=False)
    
    # V = xyz_velo0.shape[1]

    occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
    rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
    metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

    if model.module.use_metaradar and not model.module.use_radar:
        assert(False) # cannot use_metaradar without use_radar

    # radar input
    if model.module.use_metaradar:
        _rad_occ_mem0 = metarad_occ_mem0
    else:
        _rad_occ_mem0 = rad_occ_mem0

    # lidar input
    lid_occ_mem0 = occ_mem0

    # camera input
    cam0_T_camXs = cam0_T_camXs
    lrtlist_cam0_g = lrtlist_cam0

    _masks = masks.to(device) if is_training else torch.tensor(mask, device=device).unsqueeze(0).unsqueeze(0).repeat(B,1,1)
    # model forward
    if is_training:
        output_tuple = model(
            mask=_masks,
            modal_weight=modal_weight,
            label_tuple=(seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g),
            loss_fn=loss_fn,
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=vox_util,
            rad_occ_mem0=_rad_occ_mem0,
            lid_occ_mem0=lid_occ_mem0,
            is_training=True
        )
        if len(output_tuple) > 2:
            # Specialize for shap imml model
            metrics, fuse_loss, sep_loss, shap_loss, rec_loss, subfuse_loss = output_tuple
            rec_loss = rec_loss.sum(dim=1).mean()
            sep_loss = sep_loss.sum() / B
            subfuse_loss = subfuse_loss.mean()
            logging.info(f'fuse_loss: {fuse_loss.item()}, shap_weight: {shap_loss}, rec_loss: {rec_loss.item()}, sep_loss: {sep_loss.item()}, subfuse_loss: {subfuse_loss.item()}')

            sep_weight, rec_weight, sub_weight = balanced_weights[0], balanced_weights[1], balanced_weights[2]
            total_loss = fuse_loss + sep_loss * sep_weight \
                + subfuse_loss * sub_weight \
                + rec_loss * rec_weight 
        else:
            total_loss, metrics= output_tuple
        return total_loss, metrics
    
    return model(
            mask=_masks,
            modal_weight=modal_weight,
            label_tuple=(seg_bev_g, valid_bev_g, center_bev_g, offset_bev_g),
            loss_fn=loss_fn,
            rgb_camXs=rgb_camXs,
            pix_T_cams=pix_T_cams,
            cam0_T_camXs=cam0_T_camXs,
            vox_util=vox_util,
            rad_occ_mem0=_rad_occ_mem0,
            lid_occ_mem0=lid_occ_mem0,
            is_training=False
        )

    
def main(
        exp_name='clr_imml',
        # training
        max_iters=140000,
        log_freq=5000,#1000
        shuffle=True,
        dset='trainval',
        do_val=False,
        val_freq=100,
        save_freq=1000,
        batch_size=1,
        grad_acc=5,
        lr=3e-4,
        use_scheduler=True,
        weight_decay=1e-7,
        nworkers=16,
        # data/log/save/load directories
        data_dir='nuScenes/',
        log_dir='pii/logs/',#'my_logs',
        ckpt_dir='pii/ckpts/',#'my_ckpts/',
        keep_latest=2,
        init_dir='',#'my_ckpts/1x5_3e-4s_ilr+gd_00:46:47',
        ignore_load=None,
        load_step=True,
        load_optimizer=False,
        # data
        res_scale=2,
        rand_flip=True,
        rand_crop_and_resize=True,
        ncams=6,
        nsweeps=5,
        # model
        encoder_type='res101',
        use_camera=True,
        use_radar=True,
        use_radar_filters=False,
        use_lidar=True,
        use_metaradar=True,
        do_rgbcompress=True,
        do_shuffle_cams=True,
        modality=0,
        missing_id=0,
        # cuda
        device_ids=[0],
        balanced_weights=[1.,1.,1.]
    ):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    _missing_rates = [
        [0.2, 0.5, 0.8],
        [0.2, 0.8, 0.5],
        [0.5, 0.2, 0.8],
        [0.5, 0.8, 0.2],
        [0.8, 0.2, 0.5],
        [0.8, 0.5, 0.2]
    ]
    missing_rates = [_missing_rates[missing_id]] if missing_id < len(_missing_rates) else [_missing_rates[0]]
    print(_missing_rates, missing_rates)

    full_mode_list = [[1,0,0], [0,1,0], [0,0,1], [1,1,0], [1,0,1], [0,1,1], [1,1,1]]
    print('use_camera, use_radar, use_metaradar, use_lidar: ', use_camera, use_radar, use_metaradar, use_lidar)
    print('balanced_weights: ', balanced_weights)

    B = batch_size
    assert(B % len(device_ids) == 0) # batch size must be divisible by number of gpus
    if grad_acc > 1:
        print('effective batch size: ', B*grad_acc)
    device = 'cuda:%d' % device_ids[0]

    # autogen a name
    model_name = "%d" % B
    if grad_acc > 1:
        model_name += "x%d" % grad_acc
    lrn = "%.1e" % lr # e.g., 5.0e-04
    lrn = lrn[0] + lrn[3:5] + lrn[-1] # e.g., 5e-4
    model_name += "_%s" % lrn
    if use_scheduler:
        model_name += "s"
    # model_name += "_%s" % exp_name 
    import datetime
    model_date = datetime.datetime.now().strftime('%Y%m%d')
    # model_date = datetime.datetime.now().strftime('%H:%M:%S')
    model_name = model_name + '_' + model_date
    print('model_name: ', model_name)

    # set up ckpt and logging 
    ckpt_dir = os.path.join(ckpt_dir, model_name)
    
    # set up dataloaders
    final_dim = (int(224 * res_scale), int(400 * res_scale))
    print('resolution: ', final_dim)

    if rand_crop_and_resize:
        resize_lim = [0.8,1.2]
        crop_offset = int(final_dim[0]*(1-resize_lim[0]))
    else:
        resize_lim = [1.0,1.0]
        crop_offset = 0
    
    data_aug_conf = {
        'crop_offset': crop_offset,
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'H': 900, 'W': 1600,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'ncams': ncams,
    }
    
    for missing_rate in missing_rates:
        id_list = [int(i*10) for i in missing_rate]
        ckpt_path = os.path.join(ckpt_dir, f'ckpt_{id_list[0]}{id_list[1]}{id_list[2]}')
        logging.basicConfig(
            level = logging.INFO,
            format = '%(asctime)s - %(levelname)s - %(message)s',
            filename = os.path.join(log_dir, f'{model_name}_{id_list[0]}{id_list[1]}{id_list[2]}_{balanced_weights[0]}{balanced_weights[1]}{balanced_weights[2]}.log'),  # 指定文件名
            filemode = 'a'         # 模式：'w' 覆盖写，'a' 追加写（默认）
        )

        modal_weight = 1 / (1 - torch.tensor(missing_rate).to(device)) / batch_size
        logging.info('current missing_rate: %s, modal_weight: %s', missing_rate, modal_weight)

        train_dataloader, val_dataloader = nuscenesdataset.compile_data(
            dset,
            data_dir,
            data_aug_conf=data_aug_conf,
            centroid=scene_centroid_py,
            bounds=bounds,
            res_3d=(Z,Y,X),
            bsz=B,
            nworkers=nworkers,
            shuffle=shuffle,
            use_radar_filters=use_radar_filters,
            seqlen=1, # we do not load a temporal sequence here, but that can work with this dataloader
            nsweeps=nsweeps,
            do_shuffle_cams=do_shuffle_cams,
            get_tids=True,
            missing_rate=missing_rate,
        )
        train_iterloader = iter(train_dataloader)
        val_iterloader = iter(val_dataloader)

        vox_util = utils.vox.Vox_util(
            Z, Y, X,
            scene_centroid=scene_centroid.to(device),
            bounds=bounds,
            assert_cube=False)

        # set up model & seg loss
        seg_loss_fn = SimpleLoss(2.13).to(device) # value from lift-splat
        model = Segnet(Z, Y, X, vox_util, use_camera=use_camera, use_radar=use_radar, use_lidar=use_lidar, use_metaradar=use_metaradar, do_rgbcompress=do_rgbcompress, encoder_type=encoder_type, rand_flip=rand_flip, device=device)
        model = model.to(device)
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        
        parameters = list(model.parameters())
        if use_scheduler:
            optimizer, scheduler = fetch_optimizer(lr, weight_decay, 1e-8, max_iters, model.parameters())
        else:
            optimizer = torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logging.info('total_params: %s', total_params)

        # load checkpoint
        global_step = 0
        if init_dir:
            print(init_dir, load_step, load_optimizer)
            if load_step and load_optimizer:
                global_step = saverloader.load(init_dir, model.module, optimizer, ignore_load=ignore_load)
            elif load_step:
                global_step = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
            else:
                _ = saverloader.load(init_dir, model.module, ignore_load=ignore_load)
                global_step = 0
        requires_grad(parameters, True)
        model.train()

        # set up running logging pools
        n_pool = 10
        loss_pool_t = utils.misc.SimplePool(n_pool, version='np')
        time_pool_t = utils.misc.SimplePool(n_pool, version='np')
        iou_pool_t = utils.misc.SimplePool(n_pool, version='np')
        ce_pool_t = utils.misc.SimplePool(n_pool, version='np')
        center_pool_t = utils.misc.SimplePool(n_pool, version='np')
        offset_pool_t = utils.misc.SimplePool(n_pool, version='np')
        ce_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
        center_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
        offset_weight_pool_t = utils.misc.SimplePool(n_pool, version='np')
        
        best_iou = [0] * 7
        # training loop
        while global_step < max_iters:
            global_step += 1

            iter_start_time = time.time()
            iter_read_time = 0.0
            for internal_step in range(grad_acc):
                # read sample
                read_start_time = time.time()
                try:
                    sample = next(train_iterloader)
                except StopIteration:
                    train_iterloader = iter(train_dataloader)
                    sample = next(train_iterloader)
                read_time = time.time()-read_start_time
                iter_read_time += read_time

                # run training iteration
                total_loss, metrics = run_model(model, seg_loss_fn, sample, device, is_training=True, modal_weight=modal_weight, balanced_weights=balanced_weights)
                total_loss.backward()
            
            # if global_step % grad_acc == 0:
            torch.nn.utils.clip_grad_norm_(parameters, 5.0)
            optimizer.step()
            if use_scheduler:
                scheduler.step()
            torch.cuda.empty_cache()
            optimizer.zero_grad()
            
            # save model checkpoint
            if np.mod(global_step, save_freq)==0:
                saverloader.save(ckpt_path, optimizer, model.module, global_step, keep_latest=keep_latest)

            # run val
            if global_step <= 40000:
                interval = log_freq
            else:
                interval = 5000
            # interval = log_freq
            
            if global_step % interval == 0: # log_freq = 1000
                logging.info('validate if best ...')
                for _idx, _mode in enumerate(full_mode_list):
                    iou, UB, SM = [], [], []
                    torch.cuda.empty_cache()
                    model.eval()
                    for _, sample in enumerate(val_dataloader): 
                        with torch.no_grad():
                            metrics, upperbound, single_modal = run_model(model, seg_loss_fn, sample, device, mask=_mode, is_training=False)
                        UB.append(upperbound.cpu().numpy())
                        SM.append(single_modal.cpu().numpy())
                        iou.append(metrics['iou'])
                        # intersection += metrics['intersection'].mean()
                        # union += metrics['union'].mean()
                    # iou = np.mean(intersection / union
                    UB = np.stack(UB, axis=0)
                    SM = np.stack(SM, axis=0)
                    sum_UB, avg_UB = np.sum(UB, axis=0), np.mean(UB, axis=0) / B
                    sum_SM, avg_SM = np.sum(SM, axis=0), np.mean(SM, axis=0) / B
                    iou = np.mean(iou)
                    logging.info('%s mode %s IoU is %s', global_step, _mode, iou)
                    logging.info('upperbound: (sum %s), (avg %s); single modal: (sum %s), (avg %s)', sum_UB, avg_UB, sum_SM, avg_SM)

                    if iou > best_iou[_idx]:
                        logging.info('Best IoU update from %s to %s at %s', best_iou[_idx], iou, global_step)
                        best_iou[_idx] = iou
                        best_ckpt_path = os.path.join(ckpt_path + '_best', f'{_mode[0]}{_mode[1]}{_mode[2]}')
                        saverloader.save(best_ckpt_path, optimizer, model.module, global_step, keep_latest=keep_latest)

                logging.info(f'Avg IoU: {np.mean(best_iou)}, Individual IoU: {best_iou}')
                model.train()
            
            # log time
            iter_time = time.time()-iter_start_time
            logging.info('%s; step %06d/%d; rtime %.2f; itime %.2f; loss %.5f; iou_t %.1f' % (
                model_name, global_step, max_iters, iter_read_time, iter_time,
                total_loss.item(), 100*iou_pool_t.mean()))
            

if __name__ == '__main__':
    Fire(main)

