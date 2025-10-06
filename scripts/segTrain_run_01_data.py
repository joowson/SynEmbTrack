#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from glob import glob
import tifffile
import numpy as np
import os
from synembtrack.EmbedSeg.utils.preprocess_data import extract_data, split_train_val, split_train_test, split_train_crops, get_data_properties
from synembtrack.EmbedSeg.utils.generate_crops import *
from synembtrack.EmbedSeg.utils.visualize import visualize_many_crops
import json
from matplotlib.colors import ListedColormap


from synembtrack._paths import get_results_dir, get_project_root #,get_raw_data_dir
from synembtrack.utils.dataset_io import copy_dataset_to_training

# In[2]:

### use synthecized data
# train_dataset_dir =  '../demo_2Dsuspension_25C/imgGen/generated_synthSet_VSreal_cv1_density200/'


### use synthecized data
raw_data_code       = 'demo_2Dsuspension_25C'
import_train_code  = 'synthSet_demoTrain'
import_cv_code     = 'synthSet_demoVal'

trainingSet_code   = 'demo'


train_data_dir, data_folder = copy_dataset_to_training(
    raw_data_code      = raw_data_code,
    result_archive_dir = get_results_dir(),

    import_train_code = import_train_code,
    import_cv_code    = import_cv_code,

    trainingSet_code  = trainingSet_code,

    # skip_if_exists     = True,
)




# In[3]:

os.chdir(train_data_dir)


data_dir     = train_data_dir
project_name = data_folder


# In[4]:
# ### Split Data into `train`, `val` \& `test`
# Since the `train`-`test` data partition doesn't exist by itself in the original data, we can execute the following cell to reserve some data as evaluation or test data. Here, we reserve 50 % of the available data for evaluation, as is usually done in literature, with regards to the `bbbc010-2012` dataset.


split_train_test(
    data_dir = data_dir,
    project_name = project_name,
    train_test_name = 'train',
    subset = 0.2)


# In[5]:

# For this dataset, instead of reserving a small fraction of the train dataset for validation at this stage, we first crop the images and masks in the subsequent code cells, and
# <b><a href= "split_val">later</a></b> reserve some of the generated crops for the purposes of validation. We notice that such a strategy allows better results for `bbbc010-2012` during prediction
# (because of a small dataset size). Running the next cell simply copies the train and test images and masks to the `$data_dir/$project_name/train/.` and `$data_dir/$project_name/test/.` respectively.

split_train_val(
    data_dir = data_dir,
    project_name = project_name,
    train_val_name = 'train',
    subset = 0.0)


# ### Specify desired centre location for spatial embedding of pixels
# <a id='center'></a>

# Interior pixels of an object instance can either be embedded at the `medoid`, the `approximate-medoid` or the `centroid`.

# In[6]:


center = 'medoid'  # 'medoid', 'approximate-medoid', 'centroid'
try:
    assert center in {'medoid', 'approximate-medoid', 'centroid'}
    print("Spatial Embedding Location chosen as : {}".format(center))
except AssertionError as e:
    e.args += ('Please specify center as one of : {"medoid", "approximate-medoid", "centroid"}', 42)
    raise


# ### Calculate some dataset specific properties

# In the next cell, we will calculate properties of the data such as `min_object_size`, `foreground_weight` etc. <br>
# We will also specify some properties, for example,
# * set `data_properties_dir['one_hot'] = True` in case the instances are encoded in a one-hot style.
# * set `data_properties_dir['data_type']='16-bit'` if the images are of datatype `unsigned 16 bit` and
#     `data_properties_dir['data_type']='8-bit'` if the images are of datatype `unsigned 8 bit`.
#
# Lastly, we will save the dictionary `data_properties_dir` in a json file, which we will access in the `02-train` and `03-predict` notebooks.

# In[7]:


one_hot = True
data_properties_dir = get_data_properties(data_dir, project_name, train_val_name=['train'],
                                          test_name=['test'], mode='2d', one_hot=one_hot)

data_properties_dir['data_type']='8-bit'

with open('data_properties.json', 'w') as outfile:
    json.dump(data_properties_dir, outfile)
    print("Dataset properies of the `{}` dataset is saved to `data_properties.json`".format(project_name))


# ### Specify cropping configuration parameters

# Images and the corresponding masks are cropped into patches centred around an object instance, which are pre-saved prior to initiating the training. Note that the cropped images, masks and center-images would be saved at the path specified by `crops_dir` (The parameter `crops_dir` is set to ```./crops``` by default, which creates a directory at the same location as this notebook). Here, `data_subset` defines the directory which is processed. Since we only have `train` images and masks at `$data_dir/$project_name/train`, hence we set `data_subset=train`.

# In[8]:


def round_up_8(x):
    return (x.astype(int)+7) & (-8)


# In[9]:


crops_dir = './crops'
data_subset = 'train'
crop_size = np.maximum(round_up_8(3*data_properties_dir['avg_object_size_y'] + 5*data_properties_dir['stdev_object_size_y']),
round_up_8(data_properties_dir['avg_object_size_x'] + 5*data_properties_dir['stdev_object_size_x']))
print("Crop size in x and y will be set equal to {}".format(crop_size))
## 8의 배수가 되어야 함.

# ### Generate Crops
#
#

# <div class="alert alert-block alert-warning">
#     The cropped images and masks are saved at the same-location as the example notebooks. <br>
#     Generating the crops might take a little while!
# </div>

# In[10]:


image_dir = os.path.join(data_dir, project_name, data_subset, 'images')
instance_dir = os.path.join(data_dir, project_name, data_subset, 'masks')
image_names = sorted(glob(os.path.join(image_dir, '*.tif')))
instance_names = sorted(glob(os.path.join(instance_dir, '*.tif')))
for i in tqdm(np.arange(len(image_names))):
    if one_hot:
        process_one_hot(image_names[i], instance_names[i], os.path.join(crops_dir, project_name), data_subset, crop_size, center, one_hot = one_hot)
    else:
        process(image_names[i], instance_names[i], os.path.join(crops_dir, project_name), data_subset, crop_size, center, one_hot=one_hot)
print("Cropping of images, instances and centre_images for data_subset = `{}` done!".format(data_subset))


# ### Move a fraction of the generated crops for validation purposes

# Here we reserve a small fraction (15 \% by default) of the images, masks and center-images crops for the purpose of validation.
# <a id="later_val">

# In[11]:


split_train_crops(project_name = project_name, center = center, crops_dir = crops_dir, subset = 0.15)


# ### Visualize cropped images, corresponding ground truth masks and object center images

# In[12]:


new_cmap = np.load( get_project_root() / 'src' / 'synembtrack' / 'cmaps/cmap_60.npy')
new_cmap = ListedColormap(new_cmap) # new_cmap = 'magma' would also work!
visualize_many_crops(data_dir=crops_dir, project_name=project_name, train_val_dir='val', center=center, n_images=5, new_cmp=new_cmap, one_hot=one_hot)


# In[ ]:
