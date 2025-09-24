#!/usr/bin/env python
# coding: utf-8



import numpy as np
import os
from synembtrack.EmbedSeg.train import begin_training
from synembtrack.EmbedSeg.utils.create_dicts import create_dataset_dict, create_model_dict, create_loss_dict, create_configs
import torch
from matplotlib.colors import ListedColormap
import json


from synembtrack._paths import get_results_dir, get_project_root


# In[20]:


    
raw_data_code = "demo_2Dsuspension_25C" 

# ### Specify the path to `train`, `val` crops and the type of `center` embedding which we would like to train the network for:

#dataset = 'seg_dset_JM_221027_1'
dataset = 'dataset_demo'
experiment_name = 'train_01'






# In[21]:

working_dir = get_results_dir() / raw_data_code / "training"
os.chdir(working_dir)


project_name = dataset
data_dir = working_dir / 'crops'
center = 'medoid'
print("Project Name chosen as : {}. \nTrain-Val images-masks-center-images will be accessed from : {}".format(project_name, data_dir))

try:
    assert center in {'medoid', 'approximate-medoid', 'centroid'}
    print("Spatial Embedding Location chosen as : {}".format(center))
except AssertionError as e:
    e.args += ('Please specify center as one of : {"medoid", "approximate-medoid", "centroid"}', 42)
    raise



# In[22]:
# ### Obtain properties of the dataset 
json_name = 'data_properties.json'

if os.path.isfile(json_name): 
    with open(json_name) as json_file:
        data = json.load(json_file)
        one_hot, data_type, foreground_weight, n_y, n_x = data['one_hot'], data['data_type'], int(data['foreground_weight']), int(data['n_y']), int(data['n_x'])

normalization_factor = 65535 if data_type=='16-bit' else 255

# ### Create the `train_dataset_dict` dictionary
train_size = 2*len(os.listdir(os.path.join(data_dir, project_name, 'train', 'images')))
train_batch_size = 1
virtual_train_batch_multiplier = 10

train_dataset_dict = create_dataset_dict(data_dir = data_dir, 
                                         project_name = project_name,  
                                         center = center, 
                                         size = train_size, 
                                         batch_size = train_batch_size, 
                                         virtual_batch_multiplier = virtual_train_batch_multiplier, 
                                         normalization_factor= normalization_factor,
                                         one_hot = one_hot,
                                         type = 'train')

# ### Create the `val_dataset_dict` dictionary
#val_size = 8*len(os.listdir(os.path.join(data_dir, project_name, 'val', 'images')))
val_size = len(os.listdir(os.path.join(data_dir, project_name, 'val', 'images')))
val_batch_size = 1
virtual_val_batch_multiplier = 1

val_dataset_dict = create_dataset_dict(data_dir = data_dir, 
                                       project_name = project_name, 
                                       center = center, 
                                       size = val_size,
                                       batch_size = val_batch_size, 
                                       virtual_batch_multiplier = virtual_val_batch_multiplier,
                                       normalization_factor= normalization_factor,
                                       one_hot = one_hot,
                                       type ='val',)



# ### Specify model-related parameters
input_channels = 1

model_dict = create_model_dict(input_channels = input_channels)
loss_dict = create_loss_dict(foreground_weight=foreground_weight)

# ### Specify additional parameters 
# * The `n_epochs` attribute determines how long the training should proceed. In general for good results on `bbbbc_010` dataset with the configurations above, you should train for atleast 50 epochs.
# * The `display` attribute, if set to True, allows you to see the network predictions as the training proceeds. 
# * The `display_embedding` attribute, if set to True, allows you to see some sample embedding as the training proceeds. Setting this to False leads to faster training times.
# * The `save_dir` attribute identifies the location where the checkpoints and loss curve details are saved. 
# * If one wishes to **resume training** from a previous checkpoint, they could point `resume_path` attribute appropriately. For example, one could set `resume_path = './experiment/bbbc010-2012-demo/checkpoint.pth'` to resume training from the last checkpoint. 
# * The `one_hot` attribute should be set to True if the instance image is present in an one-hot encoded style (i.e. object instance is encoded as 1 in its own individual image slice) and False if the instance image is the same dimensions as the raw-image. 
# 
n_epochs = 100
display = True #True
display_embedding = True #False
save_dir = os.path.join('experiment', experiment_name)
resume_path = None
# resume_path = './experiment/train_01/checkpoint_epoch200.pth'

# ### Create the  `configs` dictionary 
configs = create_configs(n_epochs = n_epochs,
                         one_hot = one_hot,
                         display = display, 
                         display_embedding = display_embedding,
                         resume_path = resume_path, 
                         save_dir = save_dir, 
                         n_y = n_y, 
                         n_x = n_x,
                         #train_lr = 5e-4,
                         )


# ### Choose a `color map`
# Here, we load a `glasbey`-style color map. But other color maps such as `viridis`, `magma` etc would work equally well.
#new_cmap = np.load('../../02_EmbedSeg/cmaps/cmap_60.npy')

cmap_path = get_project_root() / 'src' / 'synembtrack' / 'cmaps/cmap_60.npy' ## TODO 
new_cmap = np.load(cmap_path)
new_cmap = ListedColormap(new_cmap) # new_cmap = 'magma' would also work! 


# ### Begin training!
begin_training(train_dataset_dict, val_dataset_dict, model_dict, loss_dict, configs, color_map=new_cmap)



# Executing the next cell would begin the training. 
# 
# If `display` attribute was set to `True` above, then you would see the network predictions at every $n^{th}$ step (equals 5, by default) on training and validation images. 
# 
# Going clockwise from top-left is 
# 
#     * the raw-image which needs to be segmented, 
#     * the corresponding ground truth instance mask, 
#     * the network predicted instance mask, and 
#     * (if display_embedding = True) from each object instance, 5 pixels are randomly selected (indicated with `+`), their embeddings are plotted (indicated with `.`) and the predicted margin for that object is visualized as an axis-aligned ellipse centred on the ground-truth - center (indicated with `x`)  for that object


# <div class="alert alert-block alert-warning"> 
#   Common causes for errors during training, may include : <br>
#     1. Not having <b>center images</b> for  <b>both</b> train and val directories  <br>
#     2. <b>Mismatch</b> between type of center-images saved in <b>01-data.ipynb</b> and the type of center chosen in this notebook (see the <b><a href="#center"> center</a></b> parameter in the third code cell in this notebook)   <br>
#     3. In case of resuming training from a previous checkpoint, please ensure that the model weights are read from the correct directory, using the <b><a href="#resume"> resume_path</a></b> parameter. Additionally, please ensure that the <b>save_dir</b> parameter for saving the model weights points to a relevant directory. 
# </div>
