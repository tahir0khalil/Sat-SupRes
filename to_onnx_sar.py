import torch
import cv2
import numpy as np
from os import path as osp
from basicsr.train import parse_options
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str 


opt = {
    'network_g': {'type': 'NAFSSR', 'up_scale': 2, 'width': 96, 'num_blks': 64},
    'path': {'pretrain_network_g': './exp/NAFNetSR-B_x2_SAR_2_blur/models/net_g_100000.pth'},
    'strict_load_g': True,
    'scale': 2,
    'is_train': False,
    'num_gpu': 1,
    'manual_seed': 10,
    'dist': False
} 

model = ImageRestorationModel(opt)
model.net_g.eval()
img = 'IE_240729115046657_000011.png'

img_tensor = (
    torch
    .from_numpy(cv2.imread(img)[..., ::-1].astype(np.float32) / 255.)  # Convert BGR to RGB and normalize
    .permute(2, 0, 1)  # Convert HWC to CHW format
    .unsqueeze(0)  # Add batch dimension
    .to('cuda')  # Move tensor to GPU
)


torch.onnx.export(model.net_g,              
                  img_tensor,               
                  "./ONNX_files/NF_SAR.onnx",
                  export_params=True,       
                  opset_version=11,         
                  do_constant_folding=True, 
                  input_names = ['input'],  
                  output_names = ['output'],
                  dynamic_axes={'input' : {0 : 'batch_size'}, 'output' : {0 : 'batch_size'}}) 





# no. of parameters -> 4344588
