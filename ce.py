
from PIL import Image, ImageEnhance
import numpy as np
import cv2
import os 
from skimage import io, img_as_float, img_as_ubyte
from skimage.filters import unsharp_mask
from tqdm import tqdm 


EO_path = '/NAS2/tahir/NAFNet/NAFNet/results/NAFSSR-B_EO_blur_test_2x/visualization/EO_val_100'
SAR_path = '/NAS2/tahir/NAFNet/NAFNet/results/NAFSSR-B_SAR_blur_test_2x_archived_20241112_132316/visualization/SAR_val_100' 

EO_save_path = '/home/tahir/workspace/EO_SAR_val_100/CE/EO' 
SAR_save_path = '/home/tahir/workspace/EO_SAR_val_100/CE/SAR'  

EO_imgs = os.listdir(EO_path) 
SAR_imgs = os.listdir(SAR_path) 


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


# for i in tqdm(range(100)): 
for i in range(1):
#############################################
# PIL
    eo_img = EO_imgs[i] 
    sar_img = SAR_imgs[i]  

    eo_img_pil = Image.open(os.path.join(EO_path,eo_img))
    sar_img_pil = Image.open(os.path.join(SAR_path,sar_img)) 

    # eo_img_arr = np.array(eo_img_pil)
    # sar_img_arr = np.array(sar_img_pil) 
    print("PIL")
    print(f"type(sar_img_pil): {type(sar_img_pil)}")
    print(f"size(sar_img_pil): {sar_img_pil.size}") 
    print(f"max(sar_img_pil): {max(list(sar_img_pil.getdata()))}")  

    eo_enhancer = ImageEnhance.Contrast(eo_img_pil)
    sar_enhancer = ImageEnhance.Contrast(sar_img_pil) 

    eo_enhanced_image = eo_enhancer.enhance(1.5)
    sar_enhanced_image = sar_enhancer.enhance(1.5) 

    x = eo_img[0:-4] + '_pil_contrast' + '.png'
    # eo_enhanced_image.save(os.path.join(EO_save_path, x)) 

    x = sar_img[0:-4] + '_pil_contrast' + '.png'
    # sar_enhanced_image.save(os.path.join(SAR_save_path, x))  
#===
    eo_enhancer = ImageEnhance.Sharpness(eo_img_pil)
    sar_enhancer = ImageEnhance.Sharpness(sar_img_pil) 

    eo_enhanced_image = eo_enhancer.enhance(3.0)
    sar_enhanced_image = sar_enhancer.enhance(3.0) 

    x = eo_img[0:-4] + '_pil_sharp' + '.png'
    # eo_enhanced_image.save(os.path.join(EO_save_path, x)) 

    x = sar_img[0:-4] + '_pil_sharp' + '.png'
    # sar_enhanced_image.save(os.path.join(SAR_save_path, x)) 
#############################################
# CV2 

    # eo_image = cv2.imread(os.path.join(EO_path,eo_img)) 
    # sar_image = cv2.imread(os.path.join(SAR_path,sar_img)) 
    eo_image = np.array(eo_img_pil)
    eo_image = cv2.cvtColor(eo_image, cv2.COLOR_RGB2BGR)
    sar_image = np.array(sar_img_pil)
    sar_image = cv2.cvtColor(sar_image, cv2.COLOR_RGB2BGR)
    print("CV2")
    print(f"type(sar_image): {type(sar_image)}")
    print(f"size(sar_image): {sar_image.shape}") 
    print(f"max(sar_image): {sar_image.max()}") 

    eo_sharp_imagecv = cv2.filter2D(eo_image, -1, sharpening_kernel)
    x = eo_img[0:-4] + '_cv2_sharp' + '.png'
    # cv2.imwrite(os.path.join(EO_save_path, x), eo_sharp_imagecv) 

    sar_sharp_imagecv = cv2.filter2D(sar_image, -1, sharpening_kernel)
    x = sar_img[0:-4] + '_cv2_sharp' + '.png'
    # cv2.imwrite(os.path.join(SAR_save_path, x), sar_sharp_imagecv)  
#---
    eo_sharp_imagecvm = cv2.filter2D(eo_image, -1, mild_sharpening_kernel)
    x = eo_img[0:-4] + '_cv2_mildsharp' + '.png'
    # cv2.imwrite(os.path.join(EO_save_path, x), eo_sharp_imagecvm) 

    sar_sharp_imagecvm = cv2.filter2D(sar_image, -1, mild_sharpening_kernel)
    x = sar_img[0:-4] + '_cv2_mildsharp' + '.png'
    # cv2.imwrite(os.path.join(SAR_save_path, x), sar_sharp_imagecvm) 
#---
    eo_sharp_imagecvs = cv2.filter2D(eo_image, -1, strong_sharpening_kernel)
    x = eo_img[0:-4] + '_cv2_strongsharp' + '.png'
    # cv2.imwrite(os.path.join(EO_save_path, x), eo_sharp_imagecvs) 

    sar_sharp_imagecvs = cv2.filter2D(sar_image, -1, strong_sharpening_kernel)
    x = sar_img[0:-4] + '_cv2_strongsharp' + '.png'
    # cv2.imwrite(os.path.join(SAR_save_path, x), sar_sharp_imagecvs) 
# #---    
    eo_sharp_imagecvb = cv2.filter2D(eo_image, -1, high_boost_kernel)
    x = eo_img[0:-4] + '_cv2_highbooast' + '.png'
    # cv2.imwrite(os.path.join(EO_save_path, x), eo_sharp_imagecvb) 

    sar_sharp_imagecvb = cv2.filter2D(sar_image, -1, high_boost_kernel)
    x = sar_img[0:-4] + '_cv2_highbooast' + '.png'
    # cv2.imwrite(os.path.join(SAR_save_path, x), sar_sharp_imagecvb) 
#---   
#############################################
# skimage 
    eo_image_sk = io.imread(os.path.join(EO_path,eo_img))
    print("SKI_before") 
    print(f"type(eo_image_sk): {type(eo_image_sk)}")
    print(f"size(eo_image_sk): {eo_image_sk.shape}") 
    print(f"max(eo_image_sk): {eo_image_sk.max()}")  

    eo_image_float = img_as_float(eo_image_sk)

    print("SKI") 
    print(f"type(eo_image_float): {type(eo_image_float)}")
    print(f"size(eo_image_float): {eo_image_float.shape}") 
    print(f"max(eo_image_float): {eo_image_float.max()}") 

    eo_sharpened_image = unsharp_mask(eo_image_float, radius=2.0, amount=1.5) 
    eo_sharp_image_unshapmask = img_as_ubyte(eo_sharpened_image) 
    print("after SKI") 
    print(f"type(eo_image_float): {type(eo_sharp_image_unshapmask)}")
    print(f"size(eo_image_float): {eo_sharp_image_unshapmask.shape}") 
    print(f"max(eo_image_float): {eo_sharp_image_unshapmask.max()}")
    x = eo_img[0:-4] + '_sk' + '.png'
    # io.imsave(os.path.join(EO_save_path, x), eo_sharp_image_unshapmask)
    
    sar_image_sk = io.imread(os.path.join(SAR_path,sar_img))
    sar_image_float = img_as_float(sar_image_sk)
    sar_sharpened_image = unsharp_mask(sar_image_float, radius=2.0, amount=1.5) 
    sar_sharp_image_unshapmask = img_as_ubyte(sar_sharpened_image) 
    x = sar_img[0:-4] + '_sk' + '.png'
    # io.imsave(os.path.join(SAR_save_path, x), sar_sharp_image_unshapmask)


    


# PIL
# Pillow provides the ImageEnhance module for enhancing images, including sharpness.
# imagepil = Image.open('IE_240729115026379_000003_20001_16.435142.png')
# image_array = np.array(imagepil)
# enhancer = ImageEnhance.Contrast(imagepil)
# enhanced_image = enhancer.enhance(1.5)  # Increase contrast by 50%
# enhanced_image.save('enhanced_imagepil.png')

# enhancer = ImageEnhance.Sharpness(imagepil)
# sharp_image = enhancer.enhance(3.0)  # Double the sharpness
# sharp_image.save('sharp_imagepil.png')

#############################################

# # openCV Kernel Sharpening: Create a sharpening filter using a convolution kernel.
# image = cv2.imread("A25_240402163532754_000034_20002_33.615427.png")
# # Define a sharpening kernel (This kernel emphasizes the central pixel while suppressing the surrounding pixels, making the image appear sharper.)
# sharpening_kernel = np.array([[-1, -1, -1], 
#                               [-1,  9, -1], 
#                               [-1, -1, -1]])
# mild_sharpening_kernel = np.array([[0, -1,  0],
#                                    [-1,  5, -1],
#                                    [0, -1,  0]])
# strong_sharpening_kernel = np.array([[-1, -2, -1],
#                                      [-2, 13, -2],
#                                      [-1, -2, -1]])
# k = 1.5  # Boost factor (high_boost_kernel=identity_matrix×k−low-pass_kernel)
# high_boost_kernel = np.array([[-1, -1, -1],
#                               [-1,  k+8, -1],
#                               [-1, -1, -1]])
# sharp_imagecv = cv2.filter2D(image, -1, sharpening_kernel)
# cv2.imwrite("sharp_imagecv.png", sharp_imagecv)

# sharp_imagecvm = cv2.filter2D(image, -1, mild_sharpening_kernel)
# cv2.imwrite("sharp_imagecvm.png", sharp_imagecvm)

# sharp_imagecvs = cv2.filter2D(image, -1, strong_sharpening_kernel)
# cv2.imwrite("sharp_imagecvs.png", sharp_imagecvs)

# sharp_imagecvb = cv2.filter2D(image, -1, high_boost_kernel)
# cv2.imwrite("sharp_imagecvb.png", sharp_imagecvb)

# ##########################################

# # Unsharp masking sharpens the image by enhancing the contrast between the edges and the surrounding areas.
# # It works by subtracting a blurred version of the image from the original image and then adding the result back to the original.
# # Read the image and convert it to a float in the range [0, 1]
# image = io.imread("IE_240729115026379_000003_20001_16.435142.png")
# image_float = img_as_float(image)

# # Apply unsharp mask with a suitable radius and amount
# sharpened_image = unsharp_mask(image_float, radius=2.0, amount=1.5)
# Convert the sharpened image back to uint8 and save it
# sharp_image_unshapmask = img_as_ubyte(sharpened_image)
# io.imsave("sharp_image_unshapmask.png", sharp_image_unshapmask)

##########################################

# # only edges, so leave it for now
# #Gaussian Laplace Filter
# from scipy import ndimage, misc
# import imageio.v2 as imageio

# image = imageio.imread("A25_240402163532754_000034_20002_33.615427.png")
# # Apply a Gaussian Laplace filter
# sharp_imagelaplace = ndimage.gaussian_laplace(image, sigma=1)
# imageio.imwrite("sharp_imagelaplace.jpg", sharp_imagelaplace)

# import cv2
# import numpy as np
# image = cv2.imread("A25_240402163532754_000034_20002_33.615427.png")
# # Apply Laplacian filter
# laplacian = cv2.Laplacian(image, cv2.CV_64F)
# sharpened_image = cv2.convertScaleAbs(laplacian)
# cv2.imwrite("sharpened_image.jpg", sharpened_image)