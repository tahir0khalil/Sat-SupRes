# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from BasicSR (https://github.com/xinntao/BasicSR)
# Copyright 2018-2020 BasicSR Authors
# ------------------------------------------------------------------------
import importlib
import torch
import torch.nn.functional as F
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
import time 
from FDL_pytorch import FDL_loss
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
import torchvision.transforms as transforms 
import torchvision 
import numpy as np 
import cv2 
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask 

loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')

class ImageRestorationModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageRestorationModel, self).__init__(opt)

        # define network
        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

        self.scale = int(opt['scale'])

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            if pixel_type == 'FDL_loss': 
                self.cri_pix = FDL_loss().cuda()
            else:
                cri_pix_cls = getattr(loss_module, pixel_type)
                self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                    self.device)
        else:
            self.cri_pix = None

        # if train_opt.get('perceptual_opt'):
        #     percep_type = train_opt['perceptual_opt'].pop('type')
        #     cri_perceptual_cls = getattr(loss_module, percep_type)
        #     self.cri_perceptual = cri_perceptual_cls(
        #         **train_opt['perceptual_opt']).to(self.device)
        # else:
        #     self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
        #         if k.startswith('module.offsets') or k.startswith('module.dcns'):
        #             optim_params_lowlr.append(v)
        #         else:
                optim_params.append(v)
            # else:
            #     logger = get_root_logger()
            #     logger.warning(f'Params {k} will not be optimized.')
        # print(optim_params)
        # ratio = 0.1

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam([{'params': optim_params}],
                                                **train_opt['optim_g'])
        elif optim_type == 'SGD':
            self.optimizer_g = torch.optim.SGD(optim_params,
                                               **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW([{'params': optim_params}],
                                                **train_opt['optim_g'])
            pass
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_data(self, data, is_val=False):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def grids(self):
        b, c, h, w = self.gt.size()
        self.original_size = (b, c, h, w)

        assert b == 1
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)


        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale
        #adaptive step_i, step_j
        num_row = (h - 1) // crop_size_h + 1
        num_col = (w - 1) // crop_size_w + 1

        import math
        step_j = crop_size_w if num_col == 1 else math.ceil((w - crop_size_w) / (num_col - 1) - 1e-8)
        step_i = crop_size_h if num_row == 1 else math.ceil((h - crop_size_h) / (num_row - 1) - 1e-8)

        scale = self.scale
        step_i = step_i//scale*scale
        step_j = step_j//scale*scale

        parts = []
        idxes = []

        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + crop_size_h >= h:
                i = h - crop_size_h
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + crop_size_w >= w:
                    j = w - crop_size_w
                    last_j = True
                parts.append(self.lq[:, :, i // scale :(i + crop_size_h) // scale, j // scale:(j + crop_size_w) // scale])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        self.origin_lq = self.lq
        self.lq = torch.cat(parts, dim=0)
        self.idxes = idxes

    def grids_inverse(self):
        preds = torch.zeros(self.original_size)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w))
        if 'crop_size_h' in self.opt['val']:
            crop_size_h = self.opt['val']['crop_size_h']
        else:
            crop_size_h = int(self.opt['val'].get('crop_size_h_ratio') * h)

        if 'crop_size_w' in self.opt['val']:
            crop_size_w = self.opt['val'].get('crop_size_w')
        else:
            crop_size_w = int(self.opt['val'].get('crop_size_w_ratio') * w)

        crop_size_h, crop_size_w = crop_size_h // self.scale * self.scale, crop_size_w // self.scale * self.scale

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i: i + crop_size_h, j: j + crop_size_w] += self.outs[cnt]
            count_mt[0, 0, i: i + crop_size_h, j: j + crop_size_w] += 1.

        self.output = (preds / count_mt).to(self.device)
        self.lq = self.origin_lq

    def optimize_parameters(self, current_iter, tb_logger):
        self.optimizer_g.zero_grad()

        if self.opt['train'].get('mixup', False):
            self.mixup_aug()

        preds = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]

        self.output = preds[-1]

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = 0.
            for pred in preds:
                l_pix += self.cri_pix(pred, self.gt)
            # print(f"preds[-1]: {preds[-1].shape}")
            # print(f"self.gt: {self.gt.shape}")
            # print("-----------------------------------------------")
            # l_pix = self.cri_pix(preds[-1], self.gt)
            

            # print('l pix ... ', l_pix)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        # #
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style


        l_total = l_total + 0. * sum(p.sum() for p in self.net_g.parameters())

        l_total.backward()
        use_grad_clip = self.opt['train'].get('use_grad_clip', True)
        if use_grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()


        self.log_dict = self.reduce_loss_dict(loss_dict)

    def test(self, img_name):
        self.net_g.eval()
        with torch.no_grad():
            n = len(self.lq)
            outs = []
            # m = self.opt['val'].get('max_minibatch', n)
            m=1
            # print(f"m: {m}")
            # print(f"n: {n}")
            i = 0
            while i < n:
                j = i + m
                if j >= n:
                    j = n
                pred = self.net_g(self.lq[i:j])
                if isinstance(pred, list):
                    pred = pred[-1]
                outs.append(pred.detach().cpu())
                i = j

            self.output = torch.cat(outs, dim=0)
            # print(f"===================================")
            # print(f"self.output.shape: {self.output.shape}")
            # img_tensor = self.output.squeeze(0) 
            # transform_to_pil = transforms.ToPILImage()
            # image = transform_to_pil(img_tensor)
            # save_path = '/home/tahir/workspace/descan_extension/data_set/Descan_dataset/Test/nafnet_output_40/' + str(img_name) + '.png'
            # image.save(save_path)
            # print(f"===================================")
        self.net_g.train()

    def contrast_enhancement(img, save_path, img_name, mode): 
        
        k = 1.5
        high_boost_kernel = np.array([[-1, -1, -1],
                                      [-1,  k+8, -1],
                                      [-1, -1, -1]])
        if mode == 1:
            img_np_high_boost = cv2.filter2D(img, -1, high_boost_kernel)
            x = img_name[0:-4] + '_1_highbooast' + '.png'
            save_img_path = osp.join(save_path, x) 

            imwrite(img_np_high_boost, save_img_path)
            # cv2.imwrite(os.path.join(save_img_path, x), img_np_high_boost)

    
    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
    # def dist_validation(self, dataloader, current_iter, ce_mode, save_img, rgb2bgr, use_image):
        ce_mode = self.opt['ce_mode']
        dataset_name = dataloader.dataset.opt['name']
        # with_metrics = self.opt['val'].get('metrics') is not None
        # with_metrics = None
        # with_metrics = self.opt['val']['compute_metrics'] 
        with_metrics = False 
        
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }

        rank, world_size = get_dist_info()
        if rank == 0:
            pbar = tqdm(total=len(dataloader), unit='image')

        cnt = 0
        # time_for_each_img = []
        for idx, val_data in enumerate(dataloader):
            if idx % world_size != rank:
                continue

            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            # print("===========================") 
            # print(f"img_name: {img_name}") 

            self.feed_data(val_data, is_val=True)
            # if self.opt['val'].get('grids', False):
            #     print('HERE')
            #     self.grids()
            #############################################
            # start = time.time()
            self.test(img_name)
            # end = time.time()
            # time_in_seconds = end - start
            # print (f'idx: {idx} | time_in_seconds: {time_in_seconds}')
            # time_for_each_img.append(time_in_seconds)
            # if self.opt['val'].get('grids', False):
            #     self.grids_inverse()

            visuals = self.get_current_visuals()
            outa_img_yo = visuals['result'] 
            # img_save_path_tensor = '/home/tahir/workspace/NAFNet/NAFNet/results/tensor_imgs/img512_cc512/'+img_name+'.png' 
            # torchvision.utils.save_image(outa_img_yo, img_save_path_tensor)

            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            # lq_img = tensor2img([visuals['lq']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # print("============================")
            # print("CV2")
            # print(f"type(sr_img): {type(sr_img)}")
            # print(f"size(sr_img): {sr_img.shape}") 
            # print(f"max(sr_img): {sr_img.max()}")  
            # print("============================")
            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if sr_img.shape[2] == 6:
                    L_img = sr_img[:, :, :3]
                    R_img = sr_img[:, :, 3:]

                    # visual_dir = osp.join('visual_results', dataset_name, self.opt['name'])
                    visual_dir = osp.join(self.opt['path']['visualization'], dataset_name)

                    imwrite(L_img, osp.join(visual_dir, f'{img_name}_L.png'))
                    imwrite(R_img, osp.join(visual_dir, f'{img_name}_R.png'))
                else:
                    if self.opt['is_train']:

                        save_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}.png')

                        save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                                 img_name,
                                                 f'{img_name}_{current_iter}_gt.png')
                    # else:
                        # save_img_path = osp.join(
                        #     self.opt['path']['visualization'], dataset_name,
                        #     f'{img_name}.png')
                        # save_gt_img_path = osp.join(
                        #     self.opt['path']['visualization'], dataset_name,
                        #     f'{img_name}_gt.png')
                        # save_lq_img_path = osp.join(
                        #     self.opt['path']['visualization'], dataset_name,
                        #     f'{img_name}_LQ.png')

                    # sr_img = contrast_enhancement(sr_img, self.opt['save_path'], img_name, ce_mode) 
                    ############## CE #####################
                    k = 1.5
                    high_boost_kernel = np.array([[-1, -1, -1],
                                                  [-1,  k+8, -1],
                                                  [-1, -1, -1]])
                    mild_sharpening_kernel = np.array([[0, -1,  0],
                                                       [-1,  5, -1],
                                                       [0, -1,  0]])
                    sharpening_kernel = np.array([[-1, -1, -1], 
                                                  [-1,  9, -1], 
                                                  [-1, -1, -1]])
                    strong_sharpening_kernel = np.array([[-1, -2, -1],
                                                         [-2, 13, -2],
                                                         [-1, -2, -1]])
                    if ce_mode == 1:
                        sr_img = cv2.filter2D(sr_img, -1, high_boost_kernel)
                        x = img_name + '_1_highbooast' + '.png'
                        save_img_path = osp.join(self.opt['save_path'], x) 

                        # imwrite(img_np_high_boost, save_img_path)
                    if ce_mode == 2:
                        sr_img = cv2.filter2D(sr_img, -1, mild_sharpening_kernel)
                        x = img_name + '_2_mildSharp' + '.png'
                        save_img_path = osp.join(self.opt['save_path'], x) 

                        # imwrite(img_np_mild_sharp, save_img_path)
                    if ce_mode == 3:
                        sr_img = cv2.filter2D(sr_img, -1, sharpening_kernel)
                        x = img_name + '_3_Sharp' + '.png'
                        save_img_path = osp.join(self.opt['save_path'], x) 

                        # imwrite(img_np_sharp, save_img_path)
                    if ce_mode == 4:
                        sr_img = cv2.filter2D(sr_img, -1, strong_sharpening_kernel)
                        x = img_name + '_4_strongSharp' + '.png'
                        save_img_path = osp.join(self.opt['save_path'], x) 

                        # imwrite(img_np_strong_sharp, save_img_path)

                    if ce_mode == 5: 
                        img_float = img_as_float(sr_img) 
                        sharpened_img = unsharp_mask(img_float, radius=2.0, amount=1.5)  
                        sr_img = img_as_ubyte(sharpened_img) 
                        x = img_name + '_5_unsharpMask' + '.png'
                        save_img_path = osp.join(self.opt['save_path'], x) 

                        # imwrite(sharp_img_unshapmask, save_img_path)
                    if self.opt['name'] == 'EO': 
                        imwrite(sr_img, save_img_path)
                    elif self.opt['name'] == 'SAR': 
                        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2GRAY)
                        imwrite(sr_img, save_img_path)
                    ############## CE #####################
                    # print("=============MM===============")
                    # # print("CV2")
                    # print(f"type(sr_img): {type(sr_img)}")
                    # print(f"size(sr_img): {sr_img.shape}") 
                    # print(f"max(sr_img): {sr_img.max()}")  
                    # print("============================")
                    # imwrite(sr_img, save_img_path) 
                    # imwrite(lq_img, save_lq_img_path)
                    # imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1
            if rank == 0:
                for _ in range(world_size):
                    pbar.update(1)
                    pbar.set_description(f'Test {img_name}')
        if rank == 0:
            pbar.close()

        # current_metric = 0.
        collected_metrics = OrderedDict()
        if with_metrics:
            for metric in self.metric_results.keys():
                collected_metrics[metric] = torch.tensor(self.metric_results[metric]).float().to(self.device)
            collected_metrics['cnt'] = torch.tensor(cnt).float().to(self.device)

            self.collected_metrics = collected_metrics
        ## for test#############################################
        # time_for_each_img_avg = sum(time_for_each_img[1:])/len(time_for_each_img[1:])
        # log_strr = f'average time taken per image: {time_for_each_img_avg}\n'
        # logger = get_root_logger()
        # logger.info(log_strr)
        keys = []
        metrics = []
        for name, value in self.collected_metrics.items():
            keys.append(name)
            metrics.append(value)
        metrics = torch.stack(metrics, 0)
        torch.distributed.reduce(metrics, dst=0)
        if self.opt['rank'] == 0:
            metrics_dict = {}
            cnt = 0
            for key, metric in zip(keys, metrics):
                if key == 'cnt':
                    cnt = float(metric)
                    continue
                metrics_dict[key] = float(metric)

            for key in metrics_dict:
                metrics_dict[key] /= cnt

            self._log_validation_metric_values(current_iter, dataloader.dataset.opt['name'],
                                               tb_logger, metrics_dict)
        return 0.

    def nondist_validation(self, *args, **kwargs):
        logger = get_root_logger()
        logger.warning('nondist_validation is not implemented. Run dist_validation.')
        self.dist_validation(*args, **kwargs)


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger, metric_dict):
        log_str = f'Validation {dataset_name}, \t'
        for metric, value in metric_dict.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)

        log_dict = OrderedDict()
        # for name, value in loss_dict.items():
        for metric, value in metric_dict.items():
            log_dict[f'm_{metric}'] = value

        self.log_dict = log_dict

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)