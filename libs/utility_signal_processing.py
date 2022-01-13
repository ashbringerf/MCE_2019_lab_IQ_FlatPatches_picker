""" this script record useful functions """
import numpy as np
import matplotlib.pyplot as plt
import time
#from scipy import fftpack

def transform_dft2(img, mode):
    img = img.astype(float)
    img = img - np.mean(img)
    ## use rfft here to make correct result
    if 'forward' in mode:
        ## 2d dft
        #img_dft = fftpack.fft2(img)
        img_dft = np.fft.fft2(img)
        ## shift to center
        img_dft_shifted = np.fft.fftshift(img_dft)
        return img_dft_shifted
        
    if 'inverse' in mode:
        img_dft_shifted = img
        ## shift to corner
        img_dft= np.fft.ifftshift(img_dft_shifted)
        ## 2d idft
        img = np.fft.rfft2(img_dft)
        return img


def cal_power_spectral_density(img_fre):
   
    ## set the maximum spatial frequency
    res_max_spatial  = np.min([img_fre.shape[0],  img_fre.shape[1]])
    res_max_spatial_half = np.floor(res_max_spatial / 2).astype(int)
    ## normalize the DFT coeff to keep real part and to match res_max_spatial
    
    img_fre_norm = (np.abs(img_fre) / (res_max_spatial * res_max_spatial)) ** 2

    ## calculate radial average of frequency coefficent
    coordinates_1d = np.arange(-res_max_spatial_half-1, res_max_spatial_half, 1) 
    coordinates_2d_x, coordinates_2d_y = np.meshgrid(coordinates_1d, coordinates_1d)
    radius_map = np.hypot(coordinates_2d_x, coordinates_2d_y)
    radius_map_floor = np.floor(radius_map)
    
    ## add epsilon to avoid nan
    psd = np.zeros( (res_max_spatial_half+1, 1) ) 
    
    #plt.figure
    #plt.imshow(radius_map)
    #plt.show()
    
    for radius in range(0, res_max_spatial_half - 1, 4):
        radius_loc = np.where(radius_map_floor == radius)
        coeffs = []
        
        #coeffs1 = img_fre_norm.take( radius_loc)
        #print(img_fre_norm.shape[0], img_fre_norm.shape[1], radius_loc)
        coeffs = img_fre_norm[radius_loc]
        #print(radius, ' ',np.sum(coeffs1 - coeffs2))
        psd[radius] = np.mean(coeffs)
    
    
    maxX = 10 ** (np.ceil(np.log10(float(res_max_spatial_half))));
    frequency = np.linspace(1, maxX, len(psd)); 
    frequency = (frequency) / (np.max(frequency)) / 2;
    return psd, frequency        

def cal_mtf_of_psd(coeff_src, coeff_ref, mode):
    ### 
    if 'direct' in mode:
        coeff_src = coeff_src / np.max(coeff_src)
        coeff_ref = coeff_ref / np.max(coeff_ref)
        
        mtf = np.divide(coeff_src, coeff_ref)
        
    if 'cross' in mode:
       x=1 

    return mtf