""" this script record useful functions """
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageOps


def img_crop_to_fit_block(img, block_size):
    
    # set image size to match block size and levels
    block_no_w, block_no_h = int(img.shape[1] / block_size), \
                             int(img.shape[0] / block_size)
    img_crop = img[0:(block_no_h-1) * block_size, \
                   0:(block_no_w-1) * block_size]

    return img_crop

def debug_imshow(img, window_name):

    """ cv2 imshow debug """
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, img)
    cv2.waitKey()


def img_crop_center(img, h, w):
    
    """ crop img from the center """ 
    """ the boarder is jpg 8x8 adapted """ 
    """ the corner condition is not correct """
    
    width, height = img.shape[1], img.shape[0]
    width_crop, height_crop = min(width, w), min(height, h)
    
    x_start, y_start, x_end, y_end = (width >> 1) - (width_crop >> 1),\
                                     (height >> 1) - (height_crop >> 1),\
                                     (width >> 1) + (width_crop >> 1),\
                                     (height >> 1) + (height_crop >> 1)
                                     
    x_start, y_start, x_end, y_end = int(np.floor(x_start / 8) * 8),\
                                    int(np.floor(y_start / 8) * 8),\
                                    int(np.floor(x_end / 8) * 8),\
                                    int(np.floor(y_end / 8) * 8)
    
    
    img_crop = img[ y_start : y_end,
                    x_start : x_end ]
    
    return img_crop


def img_crop(img, roi):
    """
    crop imgae bu roi, by x, y, w, h 0~1 order
    """
    height, width = img.shape[0], img.shape[1]
    height_start, height_step, width_start, width_step = int(height * roi[0]), int(height * roi[2]), int(
        width * roi[1]), int(width * roi[3])
    img_crop = img[height_start:height_start + height_step, width_start:width_start + width_step]
    return img_crop

def img_linearization(img, gamma = 2.4):
    
    img_norm = img.astype(float) /255
    img_norm_degamma = np.power(img_norm, gamma)
    img_degamma = (img_norm_degamma * 255).astype(np.uint8)

    return img_degamma

def img_apply_weighted_mask(img):
    """
    apply a weighted window on spatial domain to reduce the miss alignment
    """
    window1d = np.abs(np.blackman(img.shape[0]))
    window2d = np.sqrt(np.sqrt(np.outer(window1d,window1d)))
    
    img_masked = np.multiply(img, window2d).astype(np.uint8)
    
    return img_masked

def img_rotate(img, mode):
    if img.shape[0] > img.shape[1]:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img
    else:
        return img
        
def img_crop2square(img):
    #res = np.min([img.shape[0], img.shape[1]])
    if img.shape[0] < img.shape[1]:
        return img[0:img.shape[0] , 0:img.shape[0]]
    else:
        return img[0:img.shape[1] , 0:img.shape[1]]

def draw_grid_on_img(img, rows = 3, cols = 3):
    height, width, channels = img.shape
    r_step = int(height / rows)
    c_step = int(width / cols)
    for x in range(0, width -1):
        cv2.line(img, (c_step*(x+1) , 0), (c_step*(x+1), height), (255, 0, 0), 10, 1)
        cv2.line(img, (0, r_step*(x+1)), (width, r_step*(x+1)), (255, 0, 0), 10, 1)
    return img            

def draw_roi_on_img(img, points):
    """ draw lines from coordinate points """
    color = (0, 0, 255)
    thickness = 5
    for index in range(len(points)-1):
        print((points[index][0], points[index][1]), (points[index+1][0], points[index+1][1]))
        img = cv2.line(img, (points[index][0], points[index][1]), (points[index+1][0], points[index+1][1]), color, thickness)

    return img     

def color2gray(img):
    """ convert color image 2 gray, keep img unchanged if image is single channel """
    if len(img.shape) == 3:
        ## 
        img_out = np.zeros((img.shape[1], img.shape[0]), dtype = float)
        ## convert img 2 float
        img_float = img.astype(float)
        ## for color imge use equal weight for the channels first
        img_out = (img_float[:, :, 0] + img_float[:, :, 1] + img_float[:, :, 2]) * 0.33
        return img_out.astype(np.uint8)
    else:
        return img
        
def cal_ringing(img_color, roi_white, roi_black, roi_overlap):
    """ the black and white edge region as input """
    img = img_color[:,:,1]
    patch_white = img_crop(img, roi_white)
    patch_black = img_crop(img, roi_black)
    patch_overlap = img_crop(img, roi_overlap)
    
    ## use percentile to get black & white & undershot & overshot respectively 
    dn_whtie = np.median(patch_white)
    dn_black = np.median(patch_black)
    dn_black_under = np.percentile(patch_overlap, 1)
    dn_white_over = np.percentile(patch_overlap, 99)
    dn_whtie = np.percentile(patch_white, 75)
    dn_black = np.percentile(patch_black, 25)
    
    ringing_overshot = (dn_white_over - dn_whtie) / (dn_whtie - dn_black)
    ringing_undershot = (dn_black - dn_black_under) / (dn_whtie - dn_black)
    
    th = 0
    black_locs = np.where((img <= dn_black + th) &( img >= dn_black - th ))
    black_under_locs = np.where((img <= dn_black_under + th) &( img >= dn_black_under - th ))
    white_locs = np.where((img <= dn_whtie + th) &( img >= dn_whtie - th ))
    white_over_locs = np.where((img <= dn_white_over + th) &( img >= dn_white_over - th ))
    
    ## visualization foe debug purpose
    DEBUG = False
    if DEBUG:
        red = img_color[:, :, 2]
        green =  img_color[:, :, 1]
        blue = img_color[:, :, 0]
        red[black_locs] = 200
        green[black_locs] = 0
        blue[black_locs] = 0
        red[white_locs] = 0
        green[white_locs] = 200
        blue[white_locs] = 0
        red[white_over_locs] = 0
        green[white_over_locs] = 0
        blue[white_over_locs] = 200
        red[black_under_locs] = 200
        green[black_under_locs] = 200
        blue[black_under_locs] = 0
        cv2.imshow('   ', cv2.resize(cv2.merge([ blue, green, red]), (500, 500), cv2.INTER_NEAREST))
        cv2.waitKey()
    
    return (ringing_overshot, ringing_undershot)
    

if __name__ == "__main__":

    func = 1