import os
import time
import random
import copy
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from opts.get_opts import Options
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger, ResultRecorder
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from missing_index import missing_pattern


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def eval(model, val_iter, is_save=False, phase='test', logger=None):
    model.eval()
    total_pred = []
    total_label = []
    total_miss_type = []
    total_missing_index = []

    upperbound = torch.zeros(3).float()
    single_modal = torch.zeros(3).float()
    final_capability = torch.zeros(3).float()
    sample_count = 0

    for _iter, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)            # unpack data from dataset and apply preprocessing
        model.test()
        pred = model.pred.argmax(dim=1).detach().cpu().numpy()
        label = data['label']
        miss_type = np.array(data['miss_type'])
        total_pred.append(pred)
        total_label.append(label)
        total_miss_type.append(miss_type)
        total_missing_index.append(data['missing_index'])

        if logger is not None:
            model.cal_shap()
            sample_count += pred.shape[0]
            upperbound += model.independent_contrib.detach().cpu()
            single_modal += model.bs_shap_value.detach().cpu()
            final_capability += model.bs_individual_capability.detach().cpu()

    # calculate metrics
    total_pred = np.concatenate(total_pred)
    total_label = np.concatenate(total_label)
    total_miss_type = np.concatenate(total_miss_type)
    acc = accuracy_score(total_label, total_pred)
    uar = recall_score(total_label, total_pred, average='macro')
    f1 = f1_score(total_label, total_pred, average='macro')
    cm = confusion_matrix(total_label, total_pred)

    if logger is not None:
        total_missing_index = torch.cat(total_missing_index)
        logger.info(f'{phase}, sample_count: {sample_count}, modality_count: {total_missing_index.sum(dim=0)}')
        logger.info(f'{phase}, upperbound: {upperbound}')
        logger.info(f'{phase}, single_modal: {single_modal}')
        logger.info(f'{phase}, final capability: {final_capability}')
        logger.info(f'{phase}, Avg upperbound: {upperbound / total_missing_index.sum(dim=0).to(upperbound.device)}')
        logger.info(f'{phase}, Avg single_modal: {single_modal / total_missing_index.sum(dim=0).to(single_modal.device)}')
        logger.info(f'{phase}, Avg final capability: {final_capability / total_missing_index.sum(dim=0).to(single_modal.device)}')
    
    if is_save:
        # save test whole results
        save_dir = model.save_dir
        np.save(os.path.join(save_dir, '{}_pred.npy'.format(phase)), total_pred)
        np.save(os.path.join(save_dir, '{}_label.npy'.format(phase)), total_label)
    
        # save part results
        for part_name in ['azz', 'zvz', 'zzl', 'avz', 'azl', 'zvl', 'nnn']:
            part_index = np.where(total_miss_type == part_name)
            part_pred = total_pred[part_index]
            part_label = total_label[part_index]
            acc_part = accuracy_score(part_label, part_pred)
            uar_part = recall_score(part_label, part_pred, average='macro')
            f1_part = f1_score(part_label, part_pred, average='macro')
            np.save(os.path.join(save_dir, '{}_{}_pred.npy'.format(phase, part_name)), part_pred)
            np.save(os.path.join(save_dir, '{}_{}_label.npy'.format(phase, part_name)), part_label)
            if phase == 'test':
                recorder_lookup[part_name].write_result_to_tsv({
                    'acc': acc_part,
                    'uar': uar_part,
                    'f1': f1_part
                }, cvNo=opt.cvNo)

    model.train()

    return acc, uar, f1, cm

def clean_chekpoints(expr_name, store_epoch):
    root = os.path.join(opt.checkpoints_dir, expr_name)
    for checkpoint in os.listdir(root):
        if not checkpoint.startswith(str(store_epoch)+'_') and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))

if __name__ == '__main__':
    ##########setting seed
    # set_seed(1037)
    seed_list = [22, 333, 1067, 89, 1, 510, 68, 90, 100, 999]
    cudnn.benchmark = False
    cudnn.deterministic = True

    opt = Options().parse()                             # get training options
    opt.ext = 1.5

    opt.mse_weight = 0.15
    opt.ii = 3
    opt.beta = 0.7
    opt.eta = 0.1

    set_seed(seed_list[opt.select_seed])
    missing_rates = [[0.2, 0.5, 0.8]]#, [0.2, 0.8, 0.5], [0.5, 0.2, 0.8], [0.5, 0.8, 0.2], [0.8, 0.2, 0.5], [0.8, 0.5, 0.2]]
    print('seed: ', seed_list[opt.select_seed])
    print(missing_rates)

    for missing_rate in missing_rates:
        modal_weight = 1 / (1 - np.array(missing_rate))

        logger_path = os.path.join(opt.log_dir, opt.name, str(opt.cvNo)) # get logger path
        if not os.path.exists(logger_path):                 # make sure logger path exists
            os.mkdir(logger_path)
        
        result_dir = os.path.join(opt.log_dir, opt.name, 'results')
        if not os.path.exists(result_dir):                  # make sure result path exists
            os.mkdir(result_dir)
        
        total_cv = 10 if opt.corpus_name == 'IEMOCAP' else 12
        name_miss_rate = ''
        for _rate in missing_rate:
            name_miss_rate += f'{int(_rate*10)}'
        recorder_lookup = {                                 # init result recoreder
            "total": ResultRecorder(os.path.join(result_dir, f'result_total_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "azz": ResultRecorder(os.path.join(result_dir, f'result_azz_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "zvz": ResultRecorder(os.path.join(result_dir, f'result_zvz_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "zzl": ResultRecorder(os.path.join(result_dir, f'result_zzl_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "avz": ResultRecorder(os.path.join(result_dir, f'result_avz_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "azl": ResultRecorder(os.path.join(result_dir, f'result_azl_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "zvl": ResultRecorder(os.path.join(result_dir, f'result_zvl_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
            "nnn": ResultRecorder(os.path.join(result_dir, f'result_avl_{opt.rate_indx}_{name_miss_rate}.tsv'), total_cv=total_cv),
        }

        suffix = '_'.join([opt.model, opt.dataset_mode])    # get logger suffix
        logger = get_logger(logger_path, suffix)            # get logger
        logger.info(f'Missing rate: {missing_rate}, Modal weight: {modal_weight}') # log missing rate and modal weight

        if opt.has_test:                                    # create a dataset given opt.dataset_mode and other options
            dataset, val_dataset, tst_dataset = create_dataset_with_args(opt, set_name=['trn', 'val', 'tst'])  
        else:
            dataset, val_dataset = create_dataset_with_args(opt, set_name=['trn', 'val'])
        dataset_size = len(dataset)    # get the number of images in the dataset.
        logger.info('The number of training samples = %d' % dataset_size)
        logger.info('The number of samples in train/val/test set = %d/%d/%d' % (len(dataset), len(val_dataset), len(tst_dataset)))

        opt.total_iters = (opt.niter + opt.niter_decay + 1) * dataset_size
        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt)               # regular setup: load and print networks; create schedulers
        total_iters = 0                # the total number of training iterations
        best_eval_epoch = -1           # record the best eval epoch
        best_eval_acc, best_eval_uar, best_eval_f1 = 0, 0, 0

        mp = missing_pattern(3, dataset_size, missing_rate)
        #mp = missing_pattern(3, dataset_size, [0.5, 0.5, 0.5])    ##################

        perf = []

        unimodal_dict = dict()
        base_ckpts_path = './unimodal_checkpoints/'
        for _idx, _rate in enumerate(missing_rate):
            if _idx == 0:
                ckpt_path = os.path.join(base_ckpts_path, f'A_{int(_rate*10)}.pth')
                encoder = copy.deepcopy(model.netA)
                classifier = copy.deepcopy(model.netCls_a)
            elif _idx == 1:
                ckpt_path = os.path.join(base_ckpts_path, f'L_{int(_rate*10)}.pth')
                encoder = copy.deepcopy(model.netL)
                classifier = copy.deepcopy(model.netCls_l)
            elif _idx == 2:
                ckpt_path = os.path.join(base_ckpts_path, f'V_{int(_rate*10)}.pth')
                encoder = copy.deepcopy(model.netV)
                classifier = copy.deepcopy(model.netCls_v)
            else: raise
            model_dict = torch.load(ckpt_path, map_location='cpu')
            encoder.load_state_dict(model_dict['encoder'], strict=True)
            classifier.load_state_dict(model_dict['classifier'], strict=True)
            if _idx == 0:
                unimodal_dict['A'] = {'encoder': encoder, 'classifier': classifier}
            elif _idx == 1:
                unimodal_dict['L'] = {'encoder': encoder, 'classifier': classifier}
            elif _idx == 2:
                unimodal_dict['V'] = {'encoder': encoder, 'classifier': classifier}
        model.update_unimodal(unimodal_dict)

        for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
            epoch_start_time = time.time()  # timer for entire epoch
            iter_data_time = time.time()    # timer for data loading per iteration
            epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch

            for i, data in enumerate(dataset):  # inner loop within one epoch
                iter_start_time = time.time()   # timer for computation per iteration
                total_iters += 1                # opt.batch_size
                epoch_iter += opt.batch_size
                data['missing_index'] = mp[opt.batch_size * i : opt.batch_size * (i+1), :]   #################
                data['modal_weight'] = modal_weight
                model.set_input(data)           # unpack data from dataset and apply preprocessing
                model.optimize_parameters(epoch)   # calculate loss functions, get gradients, update network weights
                    
                if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                    losses = model.get_current_losses()
                    t_comp = (time.time() - iter_start_time) / opt.batch_size
                    logger.info('Cur epoch {}'.format(epoch) + ' loss ' + 
                            ' '.join(map(lambda x:'{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))

                iter_data_time = time.time()
                
            if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
                logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
                model.save_networks('latest')
                model.save_networks(epoch)

            logger.info('End of training epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
            model.update_learning_rate(logger)                     # update learning rates at the end of every epoch.
            
            # # eval
            acc, uar, f1, cm = eval(model, val_dataset)
            # acc, uar, f1, cm = eval(model, val_dataset, phase='val', logger=logger)
            logger.info('Val result of epoch %d / %d acc %.4f uar %.4f f1 %.4f' % (epoch, opt.niter + opt.niter_decay, acc, uar, f1))
            logger.info('\n{}'.format(cm))
            
            # show test result for debugging
            if opt.has_test and opt.verbose:
                acc, uar, f1, cm = eval(model, tst_dataset)
                # acc, uar, f1, cm = eval(model, tst_dataset, phase='test', logger=logger)
                logger.info('Tst result of epoch %d acc %.4f uar %.4f f1 %.4f' % (epoch, acc, uar, f1))
                logger.info('\n{}'.format(cm))

            perf.append([acc, uar, f1])

            # record epoch with best result
            if opt.corpus_name == 'IEMOCAP':
                if f1 > best_eval_f1:  
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            elif opt.corpus_name == 'MSP':
                if f1 > best_eval_f1:
                    best_eval_epoch = epoch
                    best_eval_uar = uar
                    best_eval_acc = acc
                    best_eval_f1 = f1
                select_metric = 'f1'
                best_metric = best_eval_f1
            else:
                raise ValueError(f'corpus name must be IEMOCAP or MSP, but got {opt.corpus_name}')

        logger.info('Best eval epoch %d found with %s %f' % (best_eval_epoch, select_metric, best_metric))



        # test
        if opt.has_test:
            logger.info('Loading best model found on val set: epoch-%d' % best_eval_epoch)
            model.load_networks(best_eval_epoch)
            _ = eval(model, val_dataset, is_save=True, phase='val')
            acc, uar, f1, cm = eval(model, tst_dataset, is_save=True, phase='test', logger=logger)
            logger.info('Tst result acc %.4f uar %.4f f1 %.4f' % (acc, uar, f1))
            logger.info('\n{}'.format(cm))
            recorder_lookup['total'].write_result_to_tsv({
                'acc': acc,
                'uar': uar,
                'f1': f1
            }, cvNo=opt.cvNo)
        else:
            recorder_lookup['total'].write_result_to_tsv({
                'acc': best_eval_acc,
                'uar': best_eval_uar,
                'f1': best_eval_f1
            }, cvNo=opt.cvNo)

        clean_chekpoints(opt.name + '/' + str(opt.cvNo), best_eval_epoch)
        logger.info('The number of training samples = %d' % dataset_size)



    