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
results_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/pose_results'

# Folder for the calculated pose errors and performance scores.
eval_path = r'/path/to/eval/folder'

######## Extended ########

# Folder for outputs (e.g. visualizations).
output_path = r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets'

# For offscreen C++ rendering: Path to the build folder of bop_renderer (github.com/thodan/bop_renderer).
bop_renderer_path = r'/path/to/bop_renderer/build'

# Executable of the MeshLab server.
#meshlab_server_path = r'/path/to/meshlabserver.exe'
meshlab_server_path = r'/usr/bin/meshlabserver'

# Custom
#dataset_name = 'tudl'
dataset_name = 'husky'
dataset_name = 'lmo'
#dataset_name = 'husky_devel'
#dataset_split = 'test'
dataset_split = 'train_pbr'
num_workers = 6
visib_thresh = 0.2
BBox_CropSize_image = 256
BBox_CropSize_GT = 128
ckpt_path=r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/6dof_pose_experiments/experiments/checkpoints/'
tensorboard_path=r'/home/pmvanderburg/noetic-husky/datasets/bop_datasets/6dof_pose_experiments/experiments/tensorboard_logs/runs/'
efficientnet_key='none'