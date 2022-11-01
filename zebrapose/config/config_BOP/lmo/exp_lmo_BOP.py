# Author: PM van der Burg (pmvanderburg@tudelft.nl)
# Cognitive Robotics, Delft University of Technology

from .. import base_config

#### training dataset
##for lmo, bop_challange = False during the training
bop_challenge = False
bop_path = base_config.datasets_path
file_name = 'exp_lmo_bop_py'
dataset_name = 'lmo'
training_data_folder = 'train'
training_data_folder_2 = 'none'
val_folder = 'test'
test_folder = 'test'
second_dataset_ratio = 0.75
num_workers = base_config.num_workers
train_obj_visible_threshold = base_config.visib_thresh
####


#### network settings
BoundingBox_CropSize_image = base_config.BBox_CropSize_image
BoundingBox_CropSize_GT = base_config.BBox_CropSize_GT

BinaryCode_Loss_Type = 'BCE'
mask_binary_code_loss = True
predict_entire_mask = False

use_histgramm_weighted_binary_loss = True

output_kernel_size = 1

resnet_layer = 34
concat_encoder_decoder = True
use_progressive_x = False
####


#### check points
load_checkpoint=False
check_point_path=base_config.ckpt_path
tensorboard_path=base_config.tensorboard_path
####


#### optimizer
optimizer_type = 'Adam'
learning_rate = 0.0002
batch_size = base_config.batch_size
total_iteration = 380000

binary_loss_weight = 3
#### 


#### augmentations
Detection_results = r'detection_results/lmo/fcos_V57eSE_MSx1333_ColorAugAAEWeaker_8e_lmo_pbr.json'

padding_ratio = 1.5
resize_method = 'crop_square_resize'

use_peper_salt= True
use_motion_blur= True

#binary coding settings
divide_number_each_iteration = 2
number_of_iterations = 16
