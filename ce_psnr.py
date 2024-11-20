import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm 

def compute_psnr(image1, image2):
    mse = np.mean((image1 - image2) ** 2)
    if mse == 0:
        return float('inf')  # Identical images
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr 


# EO_GT = '/home/tahir/workspace/EO_SAR_val_100/EO/HR' 
EO_GT = '/home/tahir/workspace/EO_SAR_val_100/SAR/HR'  

# EO_SR = '/home/tahir/workspace/NAFNet/NAFNet/results/NAFSSR-B_EO_blur_test_2x/visualization/EO_val_100'
EO_SR = '/home/tahir/workspace/NAFNet/NAFNet/results/NAFSSR-B_SAR_blur_test_2x/visualization/SAR_val_100'

# EO_CE = '/home/tahir/workspace/EO_SAR_val_100/CE/EO'
EO_CE = '/home/tahir/workspace/EO_SAR_val_100/CE/SAR' 

eo_imgs = os.listdir(EO_GT) 
# sar_imgs = os.listdir(SAR_GT)  

df = pd.DataFrame() 

img_names = [] 
p_sr = [] 
p_p_sh = [] 
p_p_cn = [] 
p_cv_stsh = [] 
p_cv_sh = [] 
p_cv_mlsh = []
p_cv_hib = [] 

for i in tqdm(range(100)): 

    eo_img = eo_imgs[i]
    # eo_img = sar_imgs[i] 
    
    eo_gt = cv2.imread(os.path.join(EO_GT, eo_img))
    eo_sr = cv2.imread(os.path.join(EO_SR, eo_img))

    eo_ce_pil_sharp = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_pil_sharp.png'))
    eo_ce_pil_contrast = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_pil_contrast.png'))
    eo_ce_cv2_stsharp = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_cv2_strongsharp.png'))
    eo_ce_cv2_sharp = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_cv2_sharp.png'))
    eo_ce_cv2_mildsharp = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_cv2_mildsharp.png'))
    eo_ce_cv2_hiboost = cv2.imread(os.path.join(EO_CE, eo_img[0:-4]+'_cv2_highbooast.png')) 

    psnr_sr = compute_psnr(eo_gt, eo_sr)
    psnr_pil_sharp = compute_psnr(eo_gt, eo_ce_pil_sharp)
    psnr_pil_contrast = compute_psnr(eo_gt, eo_ce_pil_contrast)
    psnr_cv2_stsharp = compute_psnr(eo_gt, eo_ce_cv2_stsharp)
    psnr_cv2_sharp = compute_psnr(eo_gt, eo_ce_cv2_sharp)
    psnr_cv2_mildsharp = compute_psnr(eo_gt, eo_ce_cv2_mildsharp)
    psnr_cv2_highbooast = compute_psnr(eo_gt, eo_ce_cv2_hiboost) 

    img_names.append(eo_img) 
    p_sr.append(psnr_sr) 
    p_p_sh.append(psnr_pil_sharp) 
    p_p_cn.append(psnr_pil_contrast) 
    p_cv_stsh.append(psnr_cv2_stsharp) 
    p_cv_sh.append(psnr_cv2_sharp) 
    p_cv_mlsh.append(psnr_cv2_mildsharp) 
    p_cv_hib.append(psnr_cv2_highbooast) 

    

df['img_name'] = img_names 
df['p_sr'] = p_sr 
df['p_p_sh'] = p_p_sh 
df['p_p_cn'] = p_p_cn 
df['p_cv_stsh'] = p_cv_stsh 
df['p_cv_sh'] = p_cv_sh 
df['p_cv_mlsh'] = p_cv_mlsh 
df['p_cv_hib'] = p_cv_hib  


df.to_csv("PSNR_SAR.csv") 