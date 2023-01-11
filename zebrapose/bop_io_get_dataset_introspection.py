import os
import sys

sys.path.insert(0, os.getcwd())

#from config_parser import parse_cfg

#from config.config_BOP.lmo import exp_lmo_BOP as cfg_file
#from config.config_BOP.lmo import exp_lmo_BOP as cfg_file
from config.config_BOP.tudl import exp_tudl_BOP as cfg_file

import argparse

from tools_for_BOP import bop_io_debug
from tools_for_BOP.common_dataset_info import get_obj_info
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
from torch import optim
import numpy as np

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose

sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from model.BinaryCodeNet import BinaryCodeNet_Deeplab
from model.BinaryCodeNet import MaskLoss, BinaryCodeLoss

from torch.utils.tensorboard import SummaryWriter

from utils import save_checkpoint, get_checkpoint, save_best_checkpoint
from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP

from get_detection_results import get_detection_results, ycbv_select_keyframe

from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, get_batch_size

from test_network_with_test_data import test_network_with_single_obj

def main(configs):
    config_file_name = configs['config_file_name']
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
    val_folder=configs['val_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']                                # for data loader
    train_obj_visible_threshold = configs['train_obj_visible_threshold']  # for test is always 0.1, for training we can set different values
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    mask_binary_code_loss=configs['mask_binary_code_loss']          # if binary code loss only applied for object mask region
    use_histgramm_weighted_binary_loss = configs['use_histgramm_weighted_binary_loss']
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']  
    predict_entire_mask=configs['predict_entire_mask']                  # if predict the entire object part rather than the visible one
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']
    ProgX = configs['use_progressive_x']

    #### check points
    load_checkpoint = configs['load_checkpoint']
    tensorboard_path = configs['tensorboard_path']
    check_point_path = configs['check_point_path']
    total_iteration = configs['total_iteration']                         # train how many steps
    #### optimizer
    optimizer_type = configs['optimizer_type']                           # Adam is the best sofar
    batch_size=configs['batch_size']                                     # 32 is the best so far, set to 16 for debug in local machine
    learning_rate = configs['learning_rate']                             # 0.002 or 0.003 is the best so far
    binary_loss_weight = configs['binary_loss_weight']                     # 3 is the best so far

    #### augmentations
    Detection_results=configs['Detection_results']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # vertex code settings
    divide_number_each_iteration = configs['divide_number_each_iteration']
    number_of_iterations = configs['number_of_iterations']

    sym_aware_training=configs['sym_aware_training']

    # get dataset informations
    gotbop_dataset_dir, gotmodel_plys,gotmodel_info =  bop_io_debug.get_dataset_basic_info(bop_path, dataset_name, train=True)
    print('get dataset basic info', gotbop_dataset_dir, gotmodel_plys,gotmodel_info)
    #dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io_debug.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_threshold=train_obj_visible_threshold)

    #print('dataset_dir',dataset_dir)
    #print('rgb_files',rgb_files)
    #print('gts',gts)
    '''
    print('',)
    print('',)
    print('',)
    print('',)
    print('',)
    print('',)
    print('',)
    print('',)
    '''
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1)    # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error = Calculate_ADI_Error_BOP
    else:
        Calculate_Pose_Error = Calculate_ADD_Error_BOP
    mesh_path = model_plys[obj_id+1]         # mesh_path is a dict, the obj_id should start from 1
    print(mesh_path, flush=True)
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter, flush=True)
    ########################## define data loader
    batch_size_1_dataset, batch_size_2_dataset = get_batch_size(second_dataset_ratio, batch_size)

    print("training_data_folder image example:", rgb_files[obj_id][0], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str)      # config file
    parser.add_argument('--obj_name', type=str) # obj_name
    parser.add_argument('--sym_aware_training', type=str, choices=('True','False'), default='False') # config file
    args = parser.parse_args()
    #config_file = 
    #configs = parse_cfg(config_file)
    #configs['obj_name'] = args.obj_name
    #configs['sym_aware_training'] = (args.sym_aware_training == 'True')
    configs = {
      ### args
      'obj_name': args.obj_name,
      'sym_aware_training': (args.sym_aware_training == 'True'),
      ### training dataset
      'bop_challange': cfg_file.bop_challenge,
      'dataset_name':  cfg_file.dataset_name,
      'training_data_folder': cfg_file.training_data_folder,
      'training_data_folder_2': cfg_file.training_data_folder_2,
      'val_folder':  cfg_file.val_folder,
      'test_folder':  cfg_file.test_folder,
      'second_dataset_ratio':  cfg_file.second_dataset_ratio,
      'num_workers':  cfg_file.num_workers,
      'train_obj_visible_threshold':  cfg_file.train_obj_visible_threshold,

      ### Paths

      'bop_path': cfg_file.bop_path,

      'check_point_path': cfg_file.check_point_path,
      'tensorboard_path': cfg_file.tensorboard_path,

      ### Network settings
      'BoundingBox_CropSize_image': cfg_file.BoundingBox_CropSize_image,
      'BoundingBox_CropSize_GT': cfg_file.BoundingBox_CropSize_GT,

      'BinaryCode_Loss_Type': cfg_file.BinaryCode_Loss_Type,
      'mask_binary_code_loss': cfg_file.mask_binary_code_loss,
      'predict_entire_mask': cfg_file.predict_entire_mask,

      'use_histgramm_weighted_binary_loss': cfg_file.use_histgramm_weighted_binary_loss,

      'output_kernel_size': cfg_file.output_kernel_size,

      'resnet_layer': cfg_file.resnet_layer,
      'concat_encoder_decoder': cfg_file.concat_encoder_decoder,
      'load_checkpoint': cfg_file.load_checkpoint,
      'use_progressive_x': cfg_file.use_progressive_x,
      ### Optimizer
      'optimizer_type': cfg_file.optimizer_type,
      'learning_rate': cfg_file.learning_rate,
      'batch_size': cfg_file.batch_size,
      'total_iteration': cfg_file.total_iteration,

      'binary_loss_weight': cfg_file.binary_loss_weight,
      ####


      #### augmentations
      'Detection_results': cfg_file.Detection_results,

      'padding_ratio': cfg_file.padding_ratio,
      'resize_method': cfg_file.resize_method,

      'use_peper_salt': cfg_file.use_peper_salt,
      'use_motion_blur': cfg_file.use_motion_blur,

      #binary coding settings
      'divide_number_each_iteration': cfg_file.divide_number_each_iteration,
      'number_of_iterations': cfg_file.number_of_iterations,
}

    check_point_path = configs['check_point_path']
    tensorboard_path= configs['tensorboard_path']

    config_file_name = os.path.basename(cfg_file.file_name)
    print('config_file_name',config_file_name)
    config_file_name = os.path.splitext(cfg_file.file_name)[0]
    print('config_file_name',config_file_name)
    check_point_path = configs['check_point_path'] + config_file_name
    tensorboard_path = configs['tensorboard_path'] + config_file_name
    configs['check_point_path'] = check_point_path + args.obj_name + '/'
    configs['tensorboard_path'] = tensorboard_path + args.obj_name + '/'

    configs['config_file_name'] = config_file_name

    if configs['Detection_results'] != 'none':
        Detection_results = configs['Detection_results']
        dirname = os.path.dirname(__file__)
        Detection_results = os.path.join(dirname, Detection_results)
        configs['Detection_results'] = Detection_results

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)
    print('configs:', configs)

    main(configs)
