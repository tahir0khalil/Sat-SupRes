import logging
import torch
from os import path as osp

from basicsr.data import create_dataloader, create_dataset
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str 

# name = 'EO' 
# lq_path = '/NAS2/tahir/EO_SAR_val_100/EO/LR'
# model_path = './exp/NAFNetSR-B_x2_EO_2_blur/models/net_g_100000.pth'  
# save_path = '/NAS2/tahir/EO_SAR_val_100/EO/model_output' 

name = 'SAR' 
lq_path = '/NAS2/tahir/EO_SAR_val_100/SAR/LR'
model_path = './exp/NAFNetSR-B_x2_SAR_2_blur/models/net_g_100000.pth' 
save_path = '/NAS2/tahir/EO_SAR_val_100/SAR/model_output'

CE_mode = 5

opt = {
    'name': name, 
    'dataset_opt': {'name':name, 'type':'PairedStereoImageDataset', 'dataroot_lq':lq_path, 'io_backend':{'type':'disk'}, 'phase':'test', 'scale': 2},
    'network_g': {'type': 'NAFSSR', 'up_scale': 2, 'width': 96, 'num_blks': 64},
    'path': {'pretrain_network_g': model_path},
    'save_path': save_path, 
    'strict_load_g': True,
    'scale': 2,
    'is_train': False,
    'num_gpu': 1,
    'manual_seed': 10,
    'dist': False, 
    'ce_mode': CE_mode
} 

dataset_opt = opt['dataset_opt']#.items() 
test_set = create_dataset(dataset_opt)
test_loader = create_dataloader(
            test_set,
            dataset_opt,
            num_gpu=opt['num_gpu'],
            dist=opt['dist'],
            sampler=None,
            seed=opt['manual_seed']) 

model = ImageRestorationModel(opt)  

test_set_name = test_loader.dataset.opt['name'] 
model.validation(
    test_loader,
    current_iter=opt['name'],
    # ce_mode=opt['ce_mode'], 
    tb_logger=None,
    save_img=True,
    rgb2bgr=True, use_image=True)  

