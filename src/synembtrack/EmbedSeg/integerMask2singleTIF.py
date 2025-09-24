# -*- coding: utf-8 -*-
"""
Created on Thu Jan 27 16:02:24 2022

@author: trisara
"""

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os, glob
from tqdm import tqdm

def split_integer_label(label_name):
    # read TIFF as numpy-array by using plt
    img_arr = plt.imread(label_name)
    #print(img_arr.max())
    bins = np.arange(0,256,1)
    
    if img_arr.max() == 0 :
        print('no instance')

    
    mask_list = []    
    #tmp = img_arr.copy()
    for i in range(1,img_arr.max()+1):
        #mask = np.where(img_arr == i, img_arr, 0).astype(np.uint8)
        mask = np.where(img_arr == i, 1, 0).astype(np.uint8)
        
        if np.histogram(mask, bins = bins)[0][1] !=Image.fromarray(mask).histogram()[1]:
            print('histgram does not match')
            print(np.histogram(mask, bins = bins)[0][0:2], Image.fromarray(mask).histogram()[1])
        
        #print(i, tmp.max(), imarray.max())
        img = Image.fromarray(mask)
        #print(np.max(mask*4))
        #print(Image.fromarray(mask).histogram()[i])
        #print(mask.dtype)
        mask_list.append(img)
    
    return mask_list



def get_single_TIFs(tif_path, take_last_one = False):
    
    
    savepath = tif_path + 'single_mask/'
    os.makedirs(savepath, exist_ok=True)
    
    file_list = glob.glob(tif_path +'*.tif')
    file_list.sort()
    
    if take_last_one:
        file_list = file_list[-1:]
    else:
        print('# of predictions:', len(file_list))
    
    
    
    #tif_path = tif_path.replace(os.sep, '/')
    #savepath = savepath.replace(os.sep, '/')
    
    #print(file_list)
    for idx, defl_tif_path in enumerate(file_list):
        
        # split into singles
        mask_list = split_integer_label(defl_tif_path)
        
        # infos
        img_name = os.path.basename(defl_tif_path)
       # print(img_name)
        frm = int(img_name[-8:-4])
        #frm = int(img_name[-12:-8])
        #print('frm:', frm)
        
        # save
        exist_flag = 0
        num_masks = len(mask_list)
        
        for i, mask in enumerate(mask_list):
            single_path = savepath + f'mask_embd_{frm:04d}_{i:04d}.tif'
            if os.path.exists(single_path):
                exist_flag += 1
        
        exist_flag = 0 #깨진 파일 상태로 남아있는 경우가 있더라. 우선 항상 덮어쓰도록 하자.
        if exist_flag == 0: # 전부 있거나, 하나도 없는 경우에만
            for i, mask in enumerate(mask_list):        
                single_path = savepath + f'mask_embd_{frm:04d}_{i:04d}.tif'
                mask.save(single_path, compression = 'tiff_adobe_deflate')
        elif exist_flag == num_masks:
            pass
        else:
            print('warning: some single-masks are present already!')
            break

if __name__ == "__main__":

    'JM_220210_LB', '08-5',    ['01', '02', '03', '04', '05', '06', '07', '08', '010', '011', '012']
    'JM_220322_BALO'
    
    day = 'JM_220414'
    seg_job = '11'
    
    for vid in ['PC20']:#, '05', '06', '07', '08', '010', '011', '012']:
        vid_name = f'{day}_{vid}'
        tif_path = f'/home/trisara/mMPM/04_jobs_1_Emb/{seg_job}/inference_train_01_{vid_name}/predictions/'
        print(vid_name)
        get_single_TIFs(tif_path)
