# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 16:51:58 2022

@author: trisara
"""


import os
import cv2
import numpy as np
from PIL import Image

from glob import glob
from tqdm import tqdm

def get_contours(mask_path, mask_single=None, calculation=False):
    
    ## get mask
    # 프로그램 실행 중에 mask를 읽어들일 경우 바로 쓸 수 있도록 처리.
    if mask_single == None:
        mask_single = cv2.imread(mask_path, 0)
    
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

def draw_mask_contour(img_dir, single_TIFs_dir, save_path, frame_num = None):
    os.makedirs(save_path, exist_ok=True)
    

    if frame_num == None:
        frame_list = glob(img_dir + '/test/images/*.tif')
    else:
        frame_list = glob(img_dir + f'/test/images/*{frame_num}.tif')
    

    
    frame_list.sort()

    #print(frame_list[0])
    for idx, frame_path in enumerate(frame_list):
        
        frm_count = os.path.basename(frame_path)[-8:-4]
        frame = cv2.imread(frame_path,1)
        
        mask_list = glob(single_TIFs_dir + f'mask_embd_{frm_count}_*.tif')
        mask_list.sort()
        for single_mask_path in mask_list:
            #print(single_mask_path)
            cnt = get_contours(single_mask_path)
            frame = cv2.drawContours(frame,[cnt],0,(0,0,100),1) #(200,100,50)
            
    

        #cv2.imwrite(save_path + f'contours_{frm_count}.png', frame)
        
        # 압축을 포함 (2023.05.08.)
        img = Image.fromarray(frame)
        img.save(save_path + f'contours_{frm_count}.png', compression = 'tiff_adobe_deflate')
        img.close()

if __name__ =='__main__':

    vdname = 'JM_220519_pc_02 SPL-SPL 2D'
    
    
    img_dir = f'/home/trisara/mMPM/00_src_real_img/frames_{vdname}/'
    single_TIFs_dir = f'/home/trisara/mMPM/04_jobs_1_Emb/11/inference_train_01_{vdname}/predictions/single_mask/'
    save_path = f'/home/trisara/mMPM/04_jobs_1_Emb/11/inference_train_01_{vdname}/frms_with_cnts/'


    
    #draw_mask_contour(img_dir, single_TIFs_dir, save_path)
    
    # pinpoint
    draw_mask_contour(img_dir, single_TIFs_dir, save_path, frame_num = '0936')
    
