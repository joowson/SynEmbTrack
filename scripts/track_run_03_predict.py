#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from synembtrack.EmbedSeg.utils.create_dicts import create_test_configs_dict
from synembtrack.EmbedSeg.test import begin_evaluating
from synembtrack.EmbedSeg.integerMask2singleTIF import get_single_TIFs
from synembtrack.EmbedSeg.draw_contours import draw_mask_contour
from glob import glob
import tifffile
import matplotlib.pyplot as plt
from synembtrack.EmbedSeg.utils.visualize import visualize
from synembtrack.EmbedSeg.train import invert_one_hot
import os
from matplotlib.colors import ListedColormap
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import json
import torch

from tqdm import tqdm
import sys



from synembtrack._paths import get_results_dir, get_project_root, get_raw_data_dir

# ### Specify the path to the evaluation images

# In[2]:

raw_data_code = "demo_2Dsuspension_25C" 


train_code = 'train_01'


checkpoint_path = get_results_dir() / raw_data_code / f'training/experiment/{train_code}/' / 'best_iou_model.pth'



#%% 


project_name = raw_data_code


data_dir  = get_raw_data_dir() 
train_dir = get_results_dir() / raw_data_code / 'training'

save_dir  = get_results_dir() / raw_data_code / 'segmentation'/ f'inference_{train_code}_{project_name}/'
save_dir.mkdir(parents=True, exist_ok=True)

#%%


torch.set_num_threads(1)

print("Evaluation images shall be read from: {}".format(os.path.join(data_dir, project_name)))

# ### Specify evaluation parameters 


# Some hints:
# * `tta`: Setting this to True (default) would enable **test-time augmentation**
# * `ap_val`: This parameter ("average precision value") comes into action if ground truth segmentations exist for evaluation images, and allows to compare how good our predictions are versus the available ground truth segmentations.
# * `seed_thresh`: This parameter ("seediness threshold") allows considering only those pixels as potential instance-centres which have a seediness score greater than `seed_thresh`
# * `checkpoint_path`: This parameter provides the path to the trained model weights which you would like to use for evaluation. One could test the pretrained model (available at `'../../../pretrained_models/bbbc010-2012/best_iou_model.pth'`) to get a quick glimpse on the results.
# * `save_dir`: This parameter specifies the path to the prediction instances. Equal to `inference` by default.
# * `save_images`: If True, this saves predictions at `./inference/predictions/` 
# * `save_results`: If True, this saves results at `./inference/results/`
# 
# In the cell after this one, a `test_configs` dictionary is generated from the parameters specified here!
# <a id='checkpoint'></a>


# uncomment for the model trained by you
# checkpoint_path = os.path.join('experiment', project_name+'-'+'demo', 'best_iou_model.pth')
# if os.path.isfile('data_properties.json'): 
#     with open('data_properties.json') as json_file:
#         data = json.load(json_file)
#         one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(data['min_object_size']), int(data['n_y']), int(data['n_x']), float(data['avg_background_intensity'])

# use the following for the pretrained model weights
json_path = train_dir / 'data_properties.json'
#json_path = os.path.join('../../../pretrained_models', project_name,'data_properties.json')
if os.path.isfile(json_path): 
    with open(json_path) as json_file:
        data = json.load(json_file)
        one_hot, data_type, min_object_size, n_y, n_x, avg_bg = data['one_hot'], data['data_type'], int(data['min_object_size']), int(data['n_y']), int(data['n_x']), float(data['avg_background_intensity'])

else: raise RuntimeError(f"Failed to find data_properties.json from: {os.path.abspath(json_path)}")

if os.path.exists(checkpoint_path):
    print("Trained model weights found at : {}".format(checkpoint_path))
else:
    print("Trained model weights were not found at the specified location!:", checkpoint_path)

# <div class="alert alert-block alert-info"> 
# For better extensibility to different datasets, try playing with 
#     <b>normalization_factor</b> and <b>seed_thresh</b>!
#     
# For example, adding an extra line at the end of the next code cell
# 
# `normalization_factor /=2`
#     
# would have the effect of increasing brightness of the evaluation images by a factor of 2 and may help in some cases.
#     
#     
# Similarly, lowering the <b> seed_thresh</b> has the effect of increasing the number of pixels which can be used to generate clusters.    
# 
# `seed_thresh = 0.80`
# 
# or so may at times lead to better predictions than using the default value of 0.90.   
# </div>



normalization_factor = 65535 if data_type=='16-bit' else 255
#normalization_factor = 255
seed_thresh = 0.90

tta = True
ap_val = 0.5
save_images = True
save_results = True




### Create `test_configs` dictionary from the above-specified parameters
test_configs = create_test_configs_dict(data_dir = os.path.join(data_dir, project_name), #'./crops/101/val/', #
                                        checkpoint_path = checkpoint_path,
                                        tta = tta, 
                                        ap_val = ap_val,
                                        seed_thresh = seed_thresh, 
                                        min_object_size = min_object_size, 
                                        save_images = save_images,
                                        save_results = save_results,
                                        save_dir = save_dir,
                                        normalization_factor = normalization_factor,
                                        one_hot = one_hot,
                                        n_y = n_y,
                                        n_x = n_x,)
# </div>  
# ### Load a glasbey-style color map   


cmap_path = get_project_root() / 'src' / 'synembtrack' / 'cmaps/cmap_60.npy' ## TODO 
new_cmp = np.load(cmap_path)
new_cmp = ListedColormap(new_cmp)



#print(test_configs)
# ### Begin Evaluating

# Setting `verbose` to True shows you Average Precision at IOU threshold specified by `ap_val` above for each individual image. The higher this score is, the better the network has learnt to perform instance segmentation on these unseen images.

# In[8]:
# <div class="alert alert-block alert-warning"> 
#   Common causes for a low score/error is: <br>
#     1. Accessing the model weights at the wrong location: simply editing the <b> checkpoint_path</b> would fix the issue.  <br>
#     2. At times, you would notice an improved performance by lowering <b><a href="#checkpoint"> seed_thresh</a></b> from 0.90 (default) to say 0.80. <br>
#     3. GPU is out of memory - ensure that you shutdown <i>02-train.ipynb</i> notebook
# </div>

# <div class="alert alert-block alert-info"> 
# The complete set of runs for different partitions of the data is available <b><a href = "https://github.com/juglab/EmbedSeg/wiki/BBBC010_2012"> here </a></b>!
 


begin_evaluating(test_configs, verbose = False, avg_bg= avg_bg/normalization_factor)





# %%
#get_single_TIFs(save_dir + '/predictions/')


#draw_mask_contour(os.path.join(data_dir, project_name), save_dir + '/predictions/single_mask/', save_dir + '/frms_with_cnts/')

# In[13]:


# ### Investigate some qualitative results

# Here you can investigate some quantitative predictions. GT segmentations and predictions, if they exist, are loaded from sub-directories under `save_dir`.
# Simply change `index` in the next two cells, to show the prediction for a random index.
# Going clockwise from top-left is 
# 
#     * the raw-image which needs to be segmented, 
#     * the corresponding ground truth instance mask, 
#     * the network predicted instance mask, and 
#     * (if display_embedding = True) from each object instance, 5 pixels are randomly selected (indicated with `+`), their embeddings are plotted (indicated with `.`) and the predicted margin for that object is visualized as an axis-aligned ellipse centred on the predicted - center (indicated with `x`)  for that object

# In[14]:

save_montage = False

#get_ipython().run_line_magic('matplotlib', 'inline')
if(save_images) and save_montage:
    prediction_file_names = sorted(glob(os.path.join(save_dir,'predictions','*.tif')))
    ground_truth_file_names = sorted(glob(os.path.join(save_dir,'ground-truth','*.tif')))
    embedding_file_names = sorted(glob(os.path.join(save_dir,'embedding','*.tif')))
    image_file_names = sorted(glob(os.path.join(data_dir, project_name, 'test', 'images','*.tif')))
    
    for index in tqdm(range(len(prediction_file_names))):
        #print("Image filename is {} and index is {}".format(os.path.basename(image_file_names[index]), index))
        prediction = tifffile.imread(prediction_file_names[index])
        image = tifffile.imread(image_file_names[index])
        embedding = tifffile.imread(embedding_file_names[index])
        if len(ground_truth_file_names) > 0:
            ground_truth = tifffile.imread(ground_truth_file_names[index])
            visualize(index, image = image, prediction = prediction, ground_truth = invert_one_hot(ground_truth), embedding = embedding, new_cmp = new_cmp, save_dir = save_dir)
        else:
            visualize(index, image = image, prediction = prediction, ground_truth = None, embedding = embedding, new_cmp = new_cmp, save_dir = save_dir)

    

