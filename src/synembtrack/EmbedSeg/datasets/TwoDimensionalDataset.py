import glob
import os
import random
import numpy as np
import tifffile
from skimage.segmentation import relabel_sequential
from torch.utils.data import Dataset
import cv2

import albumentations as A


class TwoDimensionalDataset(Dataset):
    """
        TwoDimensionalDataset class
    """

    def __init__(self, data_dir='./', center='center-medoid', type="train", bg_id=0, size=None, transform=None,
                 one_hot=False):

        print('2-D `{}` dataloader created! Accessing data from {}/{}/'.format(type, data_dir, type))

        # get image and instance list
        #image_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images_inverted/*.tif')))
        if type == 'test': image_list = sorted(glob.glob(os.path.join(data_dir, 'images/*.tif')))
        else:
            image_list = sorted(glob.glob(os.path.join(data_dir, '{}/'.format(type), 'images/*.tif')))
        
        
        start_ind_img =0
        if start_ind_img > 0: print('IMG started from the index:', start_ind_img)
        image_list = image_list[start_ind_img:]
        print('Starting frame:', image_list[0])
        #image_list = glob.glob(os.path.join(data_dir, 'images/*.tif'))
        #image_list.sort()

        print('Number of images in `{}` directory is {}'.format(type, len(image_list)))
        self.image_list = image_list

        instance_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), 'masks/*.tif'))
        instance_list.sort()
        print('Number of instances in `{}` directory is {}'.format(type, len(instance_list)))
        self.instance_list = instance_list

        center_image_list = glob.glob(os.path.join(data_dir, '{}/'.format(type), center + '/*.tif'))
        center_image_list.sort()
        print('Number of center images in `{}` directory is {}'.format(type, len(center_image_list)))
        print('*************************')
        self.center_image_list = center_image_list

        self.bg_id = bg_id
        self.size = size
        self.real_size = len(self.image_list)
        self.transform = transform
        self.one_hot = one_hot


        self.transfrom_2 = A.Compose([

            A.RandomBrightnessContrast  (brightness_limit=0.2, contrast_limit=0.2, always_apply=False, p=1)
        ])



    def __len__(self):

        return self.real_size if self.size is None else self.size

    def convert_yx_to_cyx(self, im, key):
        if im.ndim == 2 and key == 'image':  # gray-scale image
            im = im[np.newaxis, ...]  # CYX
        elif im.ndim == 3 and key == 'image':  # multi-channel image image
            pass
        else:
            im = im[np.newaxis, ...]
        return im

    def __getitem__(self, index):

        index = index if self.size is None else random.randint(0, self.real_size - 1)
        sample = {}

        # load image
        #print(index, self.image_list[index])
        image = tifffile.imread(self.image_list[index])  # YX or CYX

        #cv2.imwrite(f'{index}_after1.png', image)
        #print(index, 'before',  image.mean(), np.median(image), image.max(),image.shape)

        mean_img = image.mean()
        medi_img = np.median(image)
        image = image/medi_img*128

        #image = cv2.GaussianBlur(image, (5, 5), 0)

        #image = cv2.fastNlMeansDenoising(image.astype(np.float32),None,3,7,21)


        #print(index, image.mean(), np.median(image), image.max())
        #cv2.imwrite(f'{index}_after2.png', image)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

        image = self.convert_yx_to_cyx(image, key='image')

        if image.ndim ==3:
            pass
            #print(image.shape)
            #print('---------------------------')
            # 왜 그런지 모르겠는데 우리 데이터는 h,w,c로 불러와짐. 임시방편으로 여기서 c,h,w로 수정하여 사용하자. 22-04-13
            # 1장일 땐 (1,480,640) 이렇게 불러와짐..
            #image = np.transpose(image, (2, 0, 1))

        #print(image.mean())

        # 평균치 보정 - JW


        #print(image.(), image.min())

        sample['image'] = image  # CYX
        sample['im_name'] = self.image_list[index]
        if (len(self.instance_list) != 0):
            instance = tifffile.imread(self.instance_list[index])  # YX or DYX (one-hot!)
            instance, label = self.decode_instance(instance, self.one_hot, self.bg_id)
            instance = self.convert_yx_to_cyx(instance, key='instance')  # CYX or CDYX
            label = self.convert_yx_to_cyx(label, key='label')  # CYX
            sample['instance'] = instance
            sample['label'] = label
        if (len(self.center_image_list) != 0):
            center_image = tifffile.imread(self.center_image_list[index])
            center_image = self.convert_yx_to_cyx(center_image, key='center_image')  # CYX
            sample['center_image'] = center_image

        #before = sample['image'][0,:,:]


       # sample['image'] = self.transfrom_2(image=sample['image'])['image']


        #after = sample['image'][0,:,:]

        #dst = np.hstack([before, after])

        # transform

        if (self.transform is not None):
            #print('transformed!')
            #sample['image'] = self.transfrom_2(image=sample['image'])['image']

            return self.transform(sample)
        else:
            return sample

    @classmethod
    def decode_instance(cls, pic, one_hot, bg_id=None):
        pic = np.array(pic, copy=False, dtype=np.uint16)
        if (one_hot):
            instance_map = np.zeros((pic.shape[0], pic.shape[1], pic.shape[2]), dtype=np.uint8)
            class_map = np.zeros((pic.shape[1], pic.shape[2]), dtype=np.uint8)
        else:
            instance_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.int16)
            class_map = np.zeros((pic.shape[0], pic.shape[1]), dtype=np.uint8)

        if bg_id is not None:
            mask = pic > bg_id
            if mask.sum() > 0:
                ids, _, _ = relabel_sequential(pic[mask])
                instance_map[mask] = ids
                if (one_hot):
                    class_map[np.max(mask, axis=0)] = 1
                else:
                    class_map[mask] = 1

        return instance_map, class_map
