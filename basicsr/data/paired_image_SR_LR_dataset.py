# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
from torch.utils import data as data
from torchvision import datasets, models, transforms
from torchvision.transforms.functional import normalize, resize

from basicsr.data.data_util import (paired_paths_from_folder,
                                    paired_paths_from_lmdb,
                                    paired_paths_from_meta_info_file)
from basicsr.data.transforms import augment, paired_random_crop, paired_random_crop_hw
from basicsr.utils import FileClient, imfrombytes, img2tensor, padding
import os as osp
import numpy as np
from PIL import Image 
import torch 
from torch import nn 
import cv2 

class ModifiedResNet50(nn.Module):
    def __init__(self, base_model):
        super(ModifiedResNet50, self).__init__()
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])  # Keep all layers except the final fc
        
        # Define the new layers
        self.fc1 = nn.Linear(2048, 512)
        self.fc2 = nn.Linear(512, 6)
        self.sigmoid = nn.Sigmoid()  # Sigmoid to ensure outputs are between 0 and 1
        
    def forward(self, x):
        x = self.base_model(x)  # Forward pass through ResNet50 backbone
        x = torch.flatten(x, 1)  # Flatten the output from the base model
        x = self.fc1(x)  # First linear layer (2048 -> 512)
        x = nn.ReLU()(x)  # Apply ReLU activation
        x = self.fc2(x)  # Second linear layer (512 -> 6)
        x = self.sigmoid(x)  # Sigmoid activation to bound the output between 0 and 1
        return x 
    
class PairedImageSRLRDataset(data.Dataset):
    """Paired image dataset for image restoration.

    Read LQ (Low Quality, e.g. LR (Low Resolution), blurry, noisy, etc) and
    GT image pairs.

    There are three modes:
    1. 'lmdb': Use lmdb files.
        If opt['io_backend'] == lmdb.
    2. 'meta_info_file': Use meta information file to generate paths.
        If opt['io_backend'] != lmdb and opt['meta_info_file'] is not None.
    3. 'folder': Scan folders to generate paths.
        The rest.

    Args:
        opt (dict): Config for train datasets. It contains the following keys:
            dataroot_gt (str): Data root path for gt.
            dataroot_lq (str): Data root path for lq.
            meta_info_file (str): Path for meta information file.
            io_backend (dict): IO backend type and other kwarg.
            filename_tmpl (str): Template for each filename. Note that the
                template excludes the file extension. Default: '{}'.
            gt_size (int): Cropped patched size for gt patches.
            use_flip (bool): Use horizontal flips.
            use_rot (bool): Use rotation (use vertical flip and transposing h
                and w for implementation).

            scale (bool): Scale, which will be added automatically.
            phase (str): 'train' or 'val'.
    """

    def __init__(self, opt):
        super(PairedImageSRLRDataset, self).__init__()
        self.opt = opt
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.dir_clean = [] 
        self.dir_scan = [] 

        for x in osp.listdir(self.gt_folder): 
            if 'scanner03' in x or 'scanner04' in x: 
                continue
            if osp.path.isfile(osp.path.join(self.gt_folder, x)) and osp.path.isfile(osp.path.join(self.lq_folder, x)): 
                self.dir_clean.append(osp.path.join(self.gt_folder, x))
                self.dir_scan.append(osp.path.join(self.lq_folder, x))
                # print("=======================")
                # print(f"img_name: {x}")
                # print("=======================") 

        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'
        self.Transforms = transforms.Compose([transforms.ToTensor(),
                    # transforms.RandomCrop(256)
                    transforms.Resize(512) #256
                    ]) 
        # R34
        # self.color_encoder = models.resnet34(pretrained=False)
        # self.color_encoder.fc = nn.Linear(in_features=512, out_features=6, bias=False)
        # self.color_encoder = self.color_encoder.to(self.device)
        # R50
        # model = models.resnet50(pretrained=False, progress=True)
        # model.fc = nn.Linear(in_features=2048, out_features=6)
        model = ModifiedResNet50(models.resnet50(pretrained=False, progress=True)) 

        # class ModifiedResNet50(nn.Module):
        #     def __init__(self, base_model):
        #         super(ModifiedResNet50, self).__init__()
        #         self.base_model = base_model
        #         self.sigmoid = nn.Sigmoid()  # Sigmoid to ensure outputs are between 0 and 1

        #     def forward(self, x):
        #         x = self.base_model(x)
        #         x = self.sigmoid(x)
        #         return x
        # modified_model = ModifiedResNet50(model) 
        # self.color_encoder = modified_model.to(self.device) 
        self.color_encoder = model.to(self.device) 

        # #512
        # encoder_checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/weights/Color_Encoder.h5', map_location=self.device) 
        # self.color_encoder.load_state_dict(encoder_checkpoint)
        # #512-R50
        # encoder_checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_512_20240911/50.pth', map_location=self.device)
        # self.color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        # #256
        # encoder_checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_256_2/30.pth', map_location=self.device)
        # self.color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        # #256-R50
        # encoder_checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_512_retrain_20240919/140.pth', map_location=self.device)
        encoder_checkpoint = torch.load('/home/tahir/workspace/descan_extension/AAAI_Github_Code_Descan/Train_color_encoder/weights/Color_encoder_r50_256_20240912/50.pth', map_location=self.device)
        self.color_encoder.load_state_dict(encoder_checkpoint['model_state_dict'])
        
        
        
        # nums_lq = len(osp.listdir(self.lq_folder))
        # nums_gt = len(osp.listdir(self.gt_folder))
        nums_lq = len(self.dir_scan)
        nums_gt = len(self.dir_clean)

            # nums_lq = sorted(nums_lq)
            # nums_gt = sorted(nums_gt)

            # print('lq gt ... opt')
            # print(nums_lq, nums_gt, opt)
        assert nums_gt == nums_lq

        self.nums = nums_lq
            # {:04}_L   {:04}_R


            # self.paths = paired_paths_from_folder(
            #     [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            #     self.filename_tmpl)

    def __getitem__(self, index):
        img_name = self.dir_clean[index].split("/")[-1].split(".")[0]
        img_clean = Image.open(self.dir_clean[index])
        img_scan = Image.open(self.dir_scan[index]) 

        # img_clean = np.array(img_clean.convert('RGB')).astype(np.uint8)
        # img_scan = np.array(img_scan.convert('RGB')).astype(np.uint8)
        # <train> ------------------------------------------------------------
        # img_clean = self.Transforms(np.array(img_clean))
        # img_scan = self.Transforms(np.array(img_scan))

        # mean_clean = torch.mean(img_clean, dim=[1, 2])
        # std_clean = torch.std(img_clean, dim=[1, 2])

        # mean_scan = torch.mean(img_scan, dim=[1, 2])
        # std_scan = torch.std(img_scan, dim=[1, 2])

        # normalized_scan = (img_scan - mean_scan[:, None, None]) / (std_scan[:, None, None] + 1e-6)
        # shifted_scan = normalized_scan * std_clean[:, None, None] + mean_clean[:, None, None]
        # shifted_scan = shifted_scan - shifted_scan.min()
        # shifted_scan = shifted_scan / shifted_scan.max()
        # </train> ------------------------------------------------------------
        # <test> ------------------------------------------------------------
        img_clean = self.Transforms(np.array(img_clean))
        img_scan = self.Transforms(np.array(img_scan)) 
        img_scan_c = img_scan.to(self.device)

        output_dist = self.color_encoder(img_scan_c[None, ...])
        mean_pred, std_pred = output_dist[0][:3].cpu(), output_dist[0][3:].cpu()

        mean_scan = torch.mean(img_scan, dim=[1, 2])
        std_scan = torch.std(img_scan, dim=[1, 2])
        
        normalized_scan = (img_scan - mean_scan[:, None, None]) / (std_scan[:, None, None] + 1e-6)
        shifted_scan = normalized_scan * std_pred[:, None, None] + mean_pred[:, None, None]
        shifted_scan = shifted_scan - shifted_scan.min()
        shifted_scan = shifted_scan / shifted_scan.max()

        # </test> ------------------------------------------------------------
        #--------------------------------
        # if self.file_client is None:
        #     self.file_client = FileClient(
        #         self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # scale = self.opt['scale']

        # # Load gt and lq images. Dimension order: HWC; channel order: BGR;
        # # image range: [0, 1], float32.
        # # gt_path = self.paths[index]['gt_path']

        # gt_path_L = os.path.join(self.gt_folder, '{:04}_L.png'.format(index + 1))
        # gt_path_R = os.path.join(self.gt_folder, '{:04}_R.png'.format(index + 1))


        # # print('gt path,', gt_path)
        # img_bytes = self.file_client.get(gt_path_L, 'gt')
        # try:
        #     img_gt_L = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path_L))

        # img_bytes = self.file_client.get(gt_path_R, 'gt')
        # try:
        #     img_gt_R = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path_R))


        # lq_path_L = os.path.join(self.lq_folder, '{:04}_L.png'.format(index + 1))
        # lq_path_R = os.path.join(self.lq_folder, '{:04}_R.png'.format(index + 1))

        # # lq_path = self.paths[index]['lq_path']
        # # print(', lq path', lq_path)
        # img_bytes = self.file_client.get(lq_path_L, 'lq')
        # try:
        #     img_lq_L = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path_L))

        # img_bytes = self.file_client.get(lq_path_R, 'lq')
        # try:
        #     img_lq_R = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("lq path {} not working".format(lq_path_R))



        # img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        # img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)

        # # augmentation for training
        # if self.opt['phase'] == 'train':
        #     gt_size = self.opt['gt_size']
        #     # padding
        #     img_gt, img_lq = padding(img_gt, img_lq, gt_size)

        #     # random crop
        #     img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
        #                                         gt_path_L)
        #     # flip, rotation
        #     img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'],
        #                              self.opt['use_rot'])

        # # TODO: color space transform
        # # BGR to RGB, HWC to CHW, numpy to tensor
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        # # normalize
        # if self.mean is not None or self.std is not None:
        #     normalize(img_lq, self.mean, self.std, inplace=True)
        #     normalize(img_gt, self.mean, self.std, inplace=True)

        # # if scale != 1:
        # #     c, h, w = img_lq.shape
        # #     img_lq = resize(img_lq, [h*scale, w*scale])
        #     # print('img_lq .. ', img_lq.shape, img_gt.shape)


        return {
            'lq': shifted_scan,
            'gt': img_clean,
            'lq_path': img_name,
            'gt_path': img_name,
        }
    #---------------------------------------------

    def __len__(self):
        return self.nums #// 2


class PairedStereoImageDataset(data.Dataset):
    '''
    Paired dataset for stereo SR (Flickr1024, KITTI, Middlebury)
    '''
    def __init__(self, opt):
        super(PairedStereoImageDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None

        # self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        self.lq_folder = opt['dataroot_lq']
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        assert self.io_backend_opt['type'] == 'disk'
        import os
        self.lq_files = os.listdir(self.lq_folder)
        # self.gt_files = os.listdir(self.gt_folder) ####GT

        # self.nums = len(self.gt_files)
        self.nums = len(self.lq_files)

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        # gt_path_L = os.path.join(self.gt_folder, self.gt_files[index], 'hr0.png')
        # gt_path_R = os.path.join(self.gt_folder, self.gt_files[index], 'hr1.png')

        # gt_path_L = osp.path.join(self.gt_folder, self.gt_files[index]) ###GT
        # gt_path_R = osp.path.join(self.gt_folder, self.gt_files[index])

        ###GT
        # img_bytes = self.file_client.get(gt_path_L, 'gt')
        # try:
        #     img_gt_L = imfrombytes(img_bytes, float32=True)
        # except:
        #     raise Exception("gt path {} not working".format(gt_path_L))
        ###GT
        
        # blurred_image = cv2.GaussianBlur(image, (5, 5), sigmaX=2)
        # print("========================")
        # print(f"type(img_gt_L): {type(img_gt_L)}") 
        # print(f"img_gt_L.shape: {img_gt_L.shape}")
        # img_bytes = self.file_client.get(gt_path_R, 'gt')
        # try:
            # img_gt_R = imfrombytes(img_bytes, float32=True)
        # except:
            # raise Exception("gt path {} not working".format(gt_path_R))

        # lq_path_L = os.path.join(self.lq_folder, self.lq_files[index], 'lr0.png')
        # lq_path_R = os.path.join(self.lq_folder, self.lq_files[index], 'lr1.png')

        lq_path_L = osp.path.join(self.lq_folder, self.lq_files[index])
        # lq_path_R = osp.path.join(self.lq_folder, self.lq_files[index])

        # lq_path = self.paths[index]['lq_path']
        # print(', lq path', lq_path)
        img_bytes = self.file_client.get(lq_path_L, 'lq')
        try:
            img_lq_L = imfrombytes(img_bytes, float32=True)
        except:
            raise Exception("lq path {} not working".format(lq_path_L))
        
        img_lq_L = cv2.GaussianBlur(img_lq_L, (7, 7), sigmaX=1.5)

        # print(f"type(img_lq_L): {type(img_lq_L)}") 
        # print(f"img_lq_L.shape: {img_lq_L.shape}")
        # print("========================")
        # img_bytes = self.file_client.get(lq_path_R, 'lq')
        # try:
            # img_lq_R = imfrombytes(img_bytes, float32=True)
        # except:
            # raise Exception("lq path {} not working".format(lq_path_R))

        # img_gt = np.concatenate([img_gt_L, img_gt_R], axis=-1)
        # img_lq = np.concatenate([img_lq_L, img_lq_R], axis=-1)
        
        # img_gt = img_gt_L ###GT
        img_lq = img_lq_L 

        scale = self.opt['scale']
        # augmentation for training
        if self.opt['phase'] == 'train':
            if 'gt_size_h' in self.opt and 'gt_size_w' in self.opt:
                gt_size_h = int(self.opt['gt_size_h'])
                gt_size_w = int(self.opt['gt_size_w'])
            else:
                gt_size = int(self.opt['gt_size'])
                gt_size_h, gt_size_w = gt_size, gt_size

            if 'flip_RGB' in self.opt and self.opt['flip_RGB']:
                idx = [
                    [0, 1, 2, 3, 4, 5],
                    [0, 2, 1, 3, 5, 4],
                    [1, 0, 2, 4, 3, 5],
                    [1, 2, 0, 4, 5, 3],
                    [2, 0, 1, 5, 3, 4],
                    [2, 1, 0, 5, 4, 3],
                ][int(np.random.rand() * 6)]

                img_gt = img_gt[:, :, idx]
                img_lq = img_lq[:, :, idx]

            # random crop
            img_gt, img_lq = img_gt.copy(), img_lq.copy()
            img_gt, img_lq = paired_random_crop_hw(img_gt, img_lq, gt_size_h, gt_size_w, scale,
                                                'gt_path_L_and_R')
            # flip, rotation
            imgs, status = augment([img_gt, img_lq], self.opt['use_hflip'],
                                    self.opt['use_rot'], vflip=self.opt['use_vflip'], return_status=True)


            img_gt, img_lq = imgs

        ###GT 
        # img_gt, img_lq = img2tensor([img_gt, img_lq],
        #                             bgr2rgb=True,
        #                             float32=True)
        img_lq = img2tensor(img_lq, bgr2rgb=True, float32=True) 
        
        # print("========================================")
        # print(f"img_gt.shape: {img_gt.shape}")
        # print(f"img_lq.shape: {img_lq.shape}")
        # print("========================================")
        # normalize
        if self.mean is not None or self.std is not None:
            normalize(img_lq, self.mean, self.std, inplace=True)
            # normalize(img_gt, self.mean, self.std, inplace=True) ###GT

        # return {
        #     'lq': img_lq,
        #     'gt': img_gt,
        #     'lq_path': osp.path.join(self.lq_folder, self.lq_files[index]),
        #     'gt_path': osp.path.join(self.gt_folder, self.gt_files[index]),
        # }
        return {
            'lq': img_lq,
            
            'lq_path': osp.path.join(self.lq_folder, self.lq_files[index]),
        
        } 

    def __len__(self):
        return self.nums