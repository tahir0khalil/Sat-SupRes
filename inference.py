import torch
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np
# from os import path as osp
from basicsr.train import parse_options
from basicsr.data import create_dataloader, create_dataset
from basicsr.models import create_model
from basicsr.models.image_restoration_model import ImageRestorationModel
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs)
from basicsr.utils.options import dict2str  
from tqdm import tqdm 

from PIL import Image, ImageEnhance
import os 
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask



which_model = 'EO' 
lq_imgs_path = '/NAS2/tahir/EO_SAR_val_100/EO/LR' 
save_img_path = '/NAS2/tahir/EO_SAR_val_100/EO/model_output' 
ce_mode = 1

if which_model == 'EO': 
    model_path = './exp/NAFNetSR-B_x2_EO_2_blur/models/net_g_100000.pth'
elif which_model == 'SAR': 
    model_path = './exp/NAFNetSR-B_x2_SAR_2_blur/models/net_g_100000.pth'
else: 
    raise ValueError("Invalid model selection, select either EO or SAR")
min_max=(0, 1)
######
def contrast_enhancement(img, mode, img_name):

    sharpening_kernel = np.array([[-1, -1, -1], 
                              [-1,  9, -1], 
                              [-1, -1, -1]])
    mild_sharpening_kernel = np.array([[0, -1,  0],
                                    [-1,  5, -1],
                                    [0, -1,  0]])
    strong_sharpening_kernel = np.array([[-1, -2, -1],
                                        [-2, 13, -2],
                                        [-1, -2, -1]])
    k = 1.5  # Boost factor (high_boost_kernel=identity_matrix×k−low-pass_kernel)
    high_boost_kernel = np.array([[-1, -1, -1],
                                [-1,  k+8, -1],
                                [-1, -1, -1]])

    if mode == 1:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_high_boost = cv2.filter2D(img_np, -1, high_boost_kernel)
        x = img_name[0:-4] + '_1_highbooast' + '.png'
        cv2.imwrite(os.path.join(save_img_path, x), img_np_high_boost) 

    elif mode == 2:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_mild_sharp = cv2.filter2D(img_np, -1, mild_sharpening_kernel)
        x = img_name[0:-4] + '_2_mildSharp' + '.png'
        cv2.imwrite(os.path.join(save_img_path, x), img_np_mild_sharp)

    elif mode == 3:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_sharp = cv2.filter2D(img_np, -1, sharpening_kernel)
        x = img_name[0:-4] + '_3_Sharp' + '.png'
        cv2.imwrite(os.path.join(save_img_path, x), img_np_sharp)

    elif mode == 4:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_np_strong_sharp = cv2.filter2D(img_np, -1, strong_sharpening_kernel)
        x = img_name[0:-4] + '_4_strongSharp' + '.png'
        cv2.imwrite(os.path.join(save_img_path, x), img_np_strong_sharp)

    elif mode == 5:
        img_enhancer = ImageEnhance.Sharpness(img)
        img_enhanced_image = img_enhancer.enhance(3.0)
        img_name = img_name[0:-4] + '_5_low_sharp' + '.png'
        img_enhanced_image.save(os.path.join(save_img_path, x)) 

    elif mode == 6:
        img_np = np.array(img)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        img_float = img_as_float(img_np) 
        sharpened_img = unsharp_mask(img_float, radius=2.0, amount=1.5) 
        sharp_img_unshapmask = img_as_ubyte(sharpened_img) 
        x = img_name[0:-4] + '_6_unsharpMask' + '.png'
        io.imsave(os.path.join(save_img_path, x), sharp_img_unshapmask)

    else: 
        x = img_name[0:-4] + '_no_enhancement' + '.png'
        img.save(os.path.join(save_img_path, x)) 
######

opt = {
    'network_g': {'type': 'NAFSSR', 'up_scale': 2, 'width': 96, 'num_blks': 64},
    'path': {'pretrain_network_g': model_path},
    'strict_load_g': True,
    'scale': 2,
    'is_train': False,
    'num_gpu': 1,
    'manual_seed': 10,
    'dist': False
} 

model = ImageRestorationModel(opt)
model.net_g.eval()

img_names = os.listdir(lq_imgs_path) 

# for imgs in tqdm(img_names):
for imgs in tqdm(img_names):    

    img = os.path.join(lq_imgs_path, imgs) 

    img_tensor = (
        torch
        .from_numpy(cv2.imread(img)[..., ::-1].astype(np.float32) / 255.)  # Convert BGR to RGB and normalize
        .permute(2, 0, 1)  # Convert HWC to CHW format
        .unsqueeze(0)  # Add batch dimension
        .to('cuda')  # Move tensor to GPU
    ) 
    print(f"img_tensor.shape: {img_tensor.shape}")
    print(f"img_tensor.max: {img_tensor.max()}")
    print(f"img_tensor.min: {img_tensor.min()}") 

    with torch.no_grad():
        pred = model.net_g(img_tensor) 

    pred = pred.squeeze(0).float().detach().cpu().clamp_(*min_max)
    tensor_image = (pred - min_max[0]) / (min_max[1] - min_max[0])



    # img_np = tensor_image.numpy()
    img_np = img_tensor.squeeze(0).float().detach().cpu().numpy()
    img_np = img_np.transpose(1, 2, 0)
    if img_np.shape[2] == 1:  # gray image
        img_np = np.squeeze(img_np, axis=2)
    elif img_np.shape[2] == 3:
        # if rgb2bgr:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_np = (img_np * 255.0).round()
    img_np = img_np.astype(np.uint8) 

    print(f"img_np.shape: {img_np.shape}")
    print(f"img_np.max: {img_np.max()}")
    print(f"img_np.min: {img_np.min()}") 

    x = imgs[0:-4] + '_aaa_sk' + '.png'
    cv2.imwrite(os.path.join(save_img_path, x), img_np)
    # tensor_image = pred.squeeze(0)  # Remove batch dimension, shape becomes [3, 1024, 1024]
    # tensor_image = (tensor_image - tensor_image.min()) / (tensor_image.max() - tensor_image.min())
    # print(f"img_tensor.shape: {tensor_image.shape}")
    # print(f"img_tensor.max: {tensor_image.max()}")
    # print(f"img_tensor.min: {tensor_image.min()}")
    # image = to_pil_image(tensor_image)
    # image = to_pil_image(img_tensor.squeeze(0).float().detach().cpu().clamp_(*min_max))
    # print(f"image.max: {max(image)}")
    # print(f"image.size: {image.size}")
    # x = imgs[0:-4] + '_aaa_sk' + '.png'
    # image.save(os.path.join(save_img_path, x)) 
    # enhanced_img = contrast_enhancement(image, ce_mode, imgs)



''' 
use test.py format but have an opt file there and it will be enough
for contrast enhancement, add a function in image_restoration_model which takes the newly formed image -> sr_image and returns a transformed image


''' 