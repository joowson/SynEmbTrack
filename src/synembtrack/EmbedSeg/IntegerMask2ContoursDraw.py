# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:51:58 2022

@author: trisara
"""


import os
import cv2
import numpy as np

from glob import glob
from tqdm import tqdm

import matplotlib.pyplot as plt

def get_contours(mask_single, calculation=False):
    
    ## get mask
    # 프로그램 실행 중에 mask를 읽어들일 경우 바로 쓸 수 있도록 처리.
    # if mask_single == None:
    #     mask_single = cv2.imread(mask_path, 0)
    
    frame_width, frame_height = mask_single.shape[0], mask_single.shape[1]
    
    ## get imgs
    #pre_img_debug, cur_img, detected  = images_debug[0], images_debug[1], images_debug[2]
    
    ## Threshold to reverse the B/W
    ret, thr = cv2.threshold(mask_single*10, 5, 1, cv2.THRESH_BINARY_INV) # (thresh, True, method) thresh 초과는 true값
    contours, _ = cv2.findContours(thr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #cv2.imwrite('tmp.png', thr*255)
	
    ## get contour
    cnt_merged = np.array(contours[0])
    for i in range(1,len(contours)):
        cnt = contours[i]
        cnt_merged = np.concatenate((cnt_merged,np.array(cnt)), axis = 0)
    cnt_merged = cnt_merged.tolist()
    
    ## 4귀퉁이 점 제거 
    vertices = [[[0,0]],[[0,frame_width-1]],[[frame_height-1,0]],[[frame_height-1,frame_width-1]]] #좌표의 x,y 순서?
    for vertex in vertices:
        if vertex in cnt_merged:
            cnt_merged.remove(vertex)
    cnt_merged = np.array(cnt_merged)


    #print(cnt_merged)
    
    ### MASK APPEARNACE INFORMATION
    
    ## get rectacgular
    if calculation:
        rect = cv2.minAreaRect(cnt_merged)   # [(coord1, coord2),(w,h),angle]  
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        
        ## center of mass
        #spots = np.where(thr == 0) #(ind0, ind1)
        #com_x = np.mean(spots[1])
        #com_y = np.mean(spots[0])
        
        ## Area & Angle
        area_norm_factor = 1 # 150,  각도와 달리 경계값이 없어 적당히 신변잡기로 평균이 1이 되도록 나눈.
        area_norm_factor_mask = 1 # 70
        if rect[1][1] > rect[1][0]:   
            width  = rect[1][0] / np.sqrt(area_norm_factor)
            height = rect[1][1] / np.sqrt(area_norm_factor)
            angle  = rect[2]/180
        else:
            width  = rect[1][1] / np.sqrt(area_norm_factor)
            height = rect[1][0] / np.sqrt(area_norm_factor)
            angle  = (rect[2]+90)/180
        area_box  =  width * height  
        area_mask = np.sum(thr == 0) / area_norm_factor_mask
        
        return cnt_merged, [width, height, angle, area_box, area_mask]
    return cnt_merged

def draw_mask_contour(img_dir, IntegerMask_folder, save_path, frame_num = None):
    os.makedirs(save_path, exist_ok=True)
    
    
    
    IntegerMask_folder = str(IntegerMask_folder)
    if frame_num == None:
        mask_list = sorted(glob(os.path.join(IntegerMask_folder, '*.tif')))
        # frame_list = sorted(glob(os.path.join(img_dir, 'images/*.tif')))   
        iterable = enumerate(tqdm(mask_list,unit='frames'))
    else:
        mask_list = sorted(glob(os.path.join(IntegerMask_folder, f'*{frame_num}.tif')))
        # frame_list = sorted(glob(os.path.join(img_dir, 'images/*{frame_num}.tif')))   
        iterable = enumerate(mask_list)
        
    for idx, intMask_path in iterable:

        total_mask = plt.imread(intMask_path)
        
        frm_count = os.path.basename(intMask_path)[-8:-4]
        
        try:
            frame_img = cv2.imread(glob(os.path.join(img_dir, f'images/*{frm_count}.tif'))[0])
        except:
            print('no frame image for FRAME =', frm_count)
            raise RuntimeError
        px_dtype = total_mask.dtype 
        if px_dtype == 'uint16':
            bins = np.arange(0,65536,1)
        elif px_dtype == 'uint8':
            bins = np.arange(0,256,1)
        else:
            print('check dtype:', px_dtype)
            exit()
        
        top_val = total_mask.max()
        
        if top_val == 0 :
            print('DrawingContour Task: No instance found in mask tiff. Frame with no contour is saved.', f'(Frame num: {frame_num})')
            
        
        #for ind, i in enumerate(tqdm(range(1,top_val+1), unit=' masks')):
        for ind, i in enumerate(range(1,top_val+1)):
           
            ## get mask image
            single_mask = np.where(total_mask == i, 1, 0).astype(np.uint8)
            
            
            # ## get pixels of mask
            # spots = np.where(total_mask == i) #(ind0, ind1)
            # mask_coords = tuple(map(tuple, np.stack([spots[0],spots[1]], axis= -1)))   ## (y,x)
            # com_y, com_x = np.mean(spots[0:2], axis=-1)     
            # area_mask = len(mask_coords)
            # if area_mask ==0:
            #     # instance segmentation을 수행한 이미지와 그 마스크의 일부를 잘라서 불러올 경우, 비어있는 integer가 있을 수 있다.
            #     continue
            # # 영역제한 
            # #if not ((com_x < 60) & (com_y >200)):
            # if not ((com_x < 200) & (com_y >100)):
            #     pass
            #     #continue
            
            ## debugging
            # if np.histogram(single_mask, bins = bins)[0][1] !=Image.fromarray(single_mask).histogram()[1]:
            #    print('histgram does not match')
            #    print(np.histogram(single_mask, bins = bins)[0][0:2], Image.fromarray(single_mask).histogram()[1])
    
            
            cnt = get_contours(single_mask)
            frame_img = cv2.drawContours(frame_img,[cnt],0,(0,0,100),1) #(200,100,50)
                
        print()
        cv2.imwrite(os.path.join(save_path, f'contours_{frm_count}.png'), frame_img)



if __name__ =='__main__':

    #vdname = 'JM_220519_pc_02 SPL-SPL 2D'
    #day = 'SU_220821'
    #day = 'MK_221103'
    day_seg = 'JM_221027'
    day = 'JM_221103'
    #vd  = '23_B200'
    #tr_name = 'train_01'    
    #tr_name = 'train_04then06'
    tr_name = 'train_06'
    vdlist = ['01']
    seg_suffix = ''
    #for vd in ['01_60ul_hr'][:]:
    for vd in vdlist:

        img_dir                 = f'../00_src_real_img/frames_{day}/{vd}/'
        IntegerMask_folder      = f'../04_jobs_1_Emb/job_{day_seg}/{seg_suffix}inference_{tr_name}_{day}_{vd}/predictions/'
        save_path               = f'../04_jobs_1_Emb/job_{day_seg}/{seg_suffix}inference_{tr_name}_{day}_{vd}/frms_with_cnts/'
    
    
        print(save_path)
        draw_mask_contour(img_dir, IntegerMask_folder, save_path)
        # draw_mask_contour(img_dir, single_TIFs_dir, save_path, frame_num = '0936') # pinpoint
        
