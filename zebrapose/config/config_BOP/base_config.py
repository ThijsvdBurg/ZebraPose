# Author: PM van der Burg (pmvanderburg@tudelft.nl)
# Cognitive Robotics, Delft University of Technology

"""Configuration of the BOP paths."""

import os


######## Basic ########

# Folder with the BOP datasets.
if 'BOP_PATH' in os.environ:
  datasets_path = os.environ['BOP_PATH']
else:
  datasets_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets'

# Folder with pose results to be evaluated.
results_path =  datasets_path # r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/pose_results'

# Folder for the calculated pose errors and performance scores.
eval_path =  datasets_path # r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/performance'

# Custom
#num_workers = 8
#batch_size = 4
visib_thresh = 0.2
BBox_CropSize_image = 256
BBox_CropSize_GT = 128
ckpt_path=os.path.join(datasets_path,'6dof_pose_experiments','experiments','checkpoints')
tensorboard_path=os.path.join(datasets_path,'6dof_pose_experiments','experiments','tensorboard_logs','runs') # r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/6dof_pose_experiments/experiments/tensorboard_logs/runs/'
efficientnet_key=None
