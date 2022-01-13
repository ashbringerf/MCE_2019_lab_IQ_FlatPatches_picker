from flask import Flask, request, render_template
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
from PIL import Image
from scipy import signal, misc, ndimage
import os
import json

import libs.utility_img_processing as uti_ip
import libs.utility_signal_processing as uti_sp

class IQA_flatPatch():
    
    def __init__(self):
    
        # init parameters from json file
        with open('config.json') as f:
            para = json.load(f)
        
        """ default values are """
        # p_4x4_loop_step = 16
        # p_downscale_shift = 2
        # p_split_level = 2
        # p_split_th = 0.1
        # p_mask_th = 0.2
        # p_mask_block_size = 64
        # p_DEBUG = False
        
        self.p_4x4_loop_step   = para["p_4x4_loop_step"]
        self.p_downscale_shift = para["p_downscale_shift"]
        self.p_split_level     = para["p_split_level"]
        self.p_split_th        = para["p_split_th"]
        self.p_mask_th         = para["p_mask_th"]
        self.p_mask_block_size = para["p_mask_block_size"]
        self.p_DEBUG           = para["p_DEBUG"]
        
    
    def flat_patch_selection(self, img): 
    
        """ calculate non flat mask """
        """ input single channel img """
        """ output is mask """
        """ lowest level is 4x4 """
        """ highest level is 64x64  4-8-16-32-64"""
        
        """ parameters """

        
        img_ds_4x4 = cv2.resize(img, (img.shape[1] >> self.p_downscale_shift, img.shape[0] >> self.p_downscale_shift), interpolation = cv2.INTER_NEAREST)
        # interpolation = cv2.INTER_NEAREST
        # interpolation = cv2.INTER_NEAREST_EXACT
        
        # for level in range(levels):
        mask = np.zeros((img_ds_4x4.shape[0], img_ds_4x4.shape[1]))
        for row in range(0, img_ds_4x4.shape[0], self.p_4x4_loop_step):
            for col in range(0, img_ds_4x4.shape[1], self.p_4x4_loop_step):
                block = img_ds_4x4[row:row + self.p_4x4_loop_step, col:col + self.p_4x4_loop_step]
                mask_block = mask[row:row + self.p_4x4_loop_step, col:col + self.p_4x4_loop_step]
                self.split_block(block, mask_block, self.p_split_level, self.p_split_th)
        
        # dump result for debug or futher calculation
        if self.p_DEBUG:
            # mask_ = np.copy(mask)
            mask_ = np.zeros((mask.shape[0], mask.shape[1]))
            mask_loc = np.where(mask >= self.p_mask_th)
            mask_[mask_loc] = 1
            mask_us = cv2.resize(mask_, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
            
            patches_hf, counter_hf = [], 0
            patches_lf, counter_lf = [], 0
            for row in range(0, mask_us.shape[0], self.p_mask_block_size):
                for col in range(0, mask_us.shape[1], self.p_mask_block_size):
                    block_image = img[row:row + self.p_mask_block_size, col:col + self.p_mask_block_size]
                    #if mask_us[row, col] == 1:
                        # print(np.sum(mask_us[row:row + self.p_mask_block_size, col:col + self.p_mask_block_size]))
                    if np.mean(mask_us[row:row + self.p_mask_block_size, col:col + self.p_mask_block_size]) <= self.p_split_th:
                        if counter_lf == 0:
                            patches_lf = block_image
                        else:
                            patches_lf = np.concatenate((patches_lf, block_image), axis = 1)
                        counter_lf = counter_lf + 1
                    else:
                        if counter_hf == 0:
                            patches_hf = block_image
                        else:
                            patches_hf = np.concatenate((patches_hf, block_image), axis = 1)
                        counter_hf = counter_hf + 1
                
            return mask, img_ds_4x4, patches_hf, patches_lf, mask_us
        
        return mask, img_ds_4x4

    def split_block(self, block, mask, level, p_split_th):
    
        """ split block by max and min and level """
        """ input: block , current level, p_split_th """
        p_block_mean_min = 10
        steps = block.shape[0]
        steps_ = block.shape[0]>>1
        
        # tl, tr, bl, br
        # [row:row + steps_, col:col + steps_]
        # [row:row + steps_, col + steps_+1:col + steps]
        # [row + steps_ + 1:row + steps, col + steps_ + 1:col + steps]
        # [row + steps_ + 1:row + steps, col + steps_ + 1:col + steps]
        
        # pass one to avoid redundant calculations
        block_tl_sort = np.sort(block[0:0 + steps_, 0:0 + steps_].flatten())
        block_tr_sort = np.sort(block[0:0 + steps_, 0 + steps_:0 + steps].flatten())
        block_bl_sort = np.sort(block[0 + steps_ :0 + steps, 0:0 + steps_].flatten())
        block_br_sort = np.sort(block[0 + steps_ :0 + steps, 0 + steps_ :0 + steps].flatten())
        
        #print(block.shape, level)
        block_max = np.max([block_tl_sort[-1], block_tr_sort[-1], block_bl_sort[-1], block_br_sort[-1]])
        block_min = np.min([block_tl_sort[0], block_tr_sort[0], block_bl_sort[0], block_br_sort[0]])
        block_mean = block.mean()
        current_stat = (block_max - block_min) / block_mean
        
    
        #print('block_tl_sort.mean(): ',block_tl_sort.mean())
        mask[0:steps_, 0:steps_] = mask[0:steps_, 0:steps_] \
                                                + (block_tl_sort[-1]-block_tl_sort[0]) \
                                                / np.max([block_tl_sort.mean(), p_block_mean_min])
        mask[0:steps_, steps_:steps] = mask[0:steps_, steps_:steps] \
                                                + (block_tr_sort[-1]-block_tr_sort[0]) \
                                                / np.max([block_tr_sort.mean(), p_block_mean_min])
        mask[steps_:steps, 0:steps_] = mask[steps_:steps, 0:steps_] \
                                                    + (block_bl_sort[-1]-block_bl_sort[0]) \
                                                    / np.max([block_bl_sort.mean(), p_block_mean_min])
        mask[steps_:steps, steps_:steps] = mask[steps_:steps, steps_:steps] \
                                                            + (block_br_sort[-1] - block_br_sort[0]) \
                                                            / np.max([block_br_sort.mean(), p_block_mean_min])
        # end split logic when reach the end level
        if level > 0:
            self.split_block(block[0:steps_, 0:steps_], mask[0:steps_, 0:steps_], level - 1, p_split_th)
            self.split_block(block[0:steps_, steps_:steps], mask[0:steps_, steps_:steps],  level - 1, p_split_th)
            self.split_block(block[steps_:steps, 0:steps_], mask[steps_:steps, 0:steps_],  level - 1, p_split_th)
            self.split_block(block[steps_:steps, steps_:steps], mask[steps_:steps, steps_:steps], level - 1, p_split_th) 
            
        return block, mask

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

@app.route('/', methods=['GET', 'POST'])
def flask_main():
    if request.method == 'GET':
        return render_template('index.html', value='hi')
    if request.method == 'POST':
        
        # try to delete old result file
        try:
            files = os.listdir( os.path.join(os.path.dirname(__file__), \
                                             'static/' ) )
            for file in files:
                if 'results' in file and file.endswith('.png'):
                    os.remove( os.path.join(os.path.dirname(__file__), \
                                             'static/', \
                                             file ))
        except:
            pass
        
        # get posted image
        file_mea = request.files['file_mea']

        # convert to cv2 image
        file_mea = Image.open(file_mea)
        file_mea = np.array(file_mea)
        image_mea = cv2.cvtColor(np.array(file_mea), cv2.COLOR_BGR2RGB)

        fp = IQA_flatPatch()

        image_mea = uti_ip.img_crop_center(image_mea, h = int(image_mea.shape[0] * 0.75), w = int(image_mea.shape[1] * 0.75))
        
        image_mea = uti_ip.img_crop_to_fit_block(image_mea, fp.p_mask_block_size)
        image_mea_gray = cv2.cvtColor(image_mea, cv2.COLOR_BGR2GRAY)
        mask, img_ds_4x4, patches_hf, patches_lf, mask_us = fp.flat_patch_selection(image_mea_gray)
        
        
        mask_low_noise_loc = np.where(mask[:, :] <= fp.p_split_th)
        
        #img_center_crop[mask_low_noise_loc] = 0    
        #img_center_crop = cv2.cvtColor(img_center_crop, cv2.COLOR_BGR2RGB)
        
        plt.imshow(image_mea, cmap = 'gray')
        plt.imshow(mask_us, cmap = 'Reds', vmin = fp.p_mask_th, vmax = fp.p_mask_th*10, alpha = 0.7)
        plt.axis('off')
        #plt.colorbar()
        #plt.show()
        
        new_graph_name = "results" + str(time.time()) + ".png"
        plt.savefig('static/' + new_graph_name, bbox_inches='tight', pad_inches = 0, dpi = 300)
        plt.close()
        

        try:
            new_graph_name_02 = "results" + str(time.time()) + ".png"
            cv2.imwrite('static/' + new_graph_name_02, patches_lf)
        except:
            print('no valid hf patch')
        try:
            new_graph_name_03 = "results" + str(time.time()) + ".png"
            cv2.imwrite('static/' + new_graph_name_03, patches_hf)
        except:
            print('no valid lf patch')
        
        return render_template('result.html', r_cal = '1', results = new_graph_name, result_02 = new_graph_name_02, result_03 = new_graph_name_03)
        



if __name__ == '__main__':
    app.run(debug = True)
