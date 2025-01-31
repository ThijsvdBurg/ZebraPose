from distutils.command.config import config
import os
import sys
import time

sys.path.insert(0, os.getcwd())

from config_parser import parse_cfg
import argparse
import shutil
from tqdm import tqdm


from tools_for_BOP import bop_io
from bop_dataset_pytorch import bop_dataset_single_obj_pytorch

import torch
import numpy as np
import cv2

from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose
#sys.path.append("../bop_toolkit")
from bop_toolkit_lib import inout
from bop_toolkit_lib import visualization

from model.BinaryCodeNet import BinaryCodeNet_Deeplab

from metric import Calculate_ADD_Error_BOP, Calculate_ADI_Error_BOP
# calculations can be found in https://github.com/ThijsvdBurg/bop_toolkit/blob/784932032d448ba31bb8493f67e9a68d1c658832/bop_toolkit_lib/pose_error.py#L147

from get_detection_results import ycbv_select_keyframe
from common_ops import from_output_to_class_mask, from_output_to_class_binary_code, compute_original_mask
from tools_for_BOP.common_dataset_info import get_obj_info

from binary_code_helper.generate_new_dict import generate_new_corres_dict

from tools_for_BOP import write_to_csv 

def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    # training_data_folder=configs['training_data_folder']
    # training_data_folder_2=configs['training_data_folder_2']
    test_folder=configs['test_folder']                                  # usually is 'test'
    second_dataset_ratio = configs['second_dataset_ratio']              # the percentage of second dataset in the batch
    num_workers = configs['num_workers']
    train_obj_visible_threshold = configs['train_obj_visible_threshold']  # for test is always 0.1, for training we can set different values, usually 0.2
    #### network settings
    BoundingBox_CropSize_image = configs['BoundingBox_CropSize_image']  # input image size
    BoundingBox_CropSize_GT = configs['BoundingBox_CropSize_GT']        # network output size
    BinaryCode_Loss_Type = configs['BinaryCode_Loss_Type']              # now only support "L1" or "BCE"
    output_kernel_size = configs['output_kernel_size']                  # last layer kernel size
    resnet_layer = configs['resnet_layer']                              # usually resnet 34
    concat=configs['concat_encoder_decoder']
    if 'efficientnet_key' in configs.keys():
        efficientnet_key = configs['efficientnet_key']
    else:
        efficientnet_key = None
    ProgX = 0 #configs['use_progressive_x']

    #### metric params
    # diameter_threshold = 0.1 # float(configs['diameter_threshold'])
    # auc_resolution = 10
    # num_th = 3
    # calc_add_and_adi=True

    #### augmentations
    Detection_results=configs['Detection_results']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # pixel code settings
    divide_number_each_iteration = configs['divide_number_each_iteration']
    number_of_iterations = configs['number_of_iterations']
    ignore_bit = configs['ignore_bit']


    # torch.manual_seed(0)      # the both are only good for ablation study
    # np.random.seed(0)         # if can be removed in the final experiments



    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_per_obj=True, incl_param=True, train_obj_visible_threshold=train_obj_visible_threshold)
    obj_name_obj_id, symmetry_obj = get_obj_info(dataset_name)
    obj_id = int(obj_name_obj_id[obj_name] - 1) # now the obj_id started from 0
    if obj_name in symmetry_obj:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name = 'ADI'
        supp_metric_name = 'ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'
    
    mesh_path = model_plys[obj_id+1] # mesh_path is a dict, the obj_id should start from 1
    obj_diameter = model_info[str(obj_id+1)]['diameter']
    print("obj_diameter", obj_diameter)
    path_dict = os.path.join(dataset_dir, "models_GT_color", "Class_CorresPoint{:06d}.txt".format(obj_id+1))
    total_number_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    divide_number_each_iteration = int(divide_number_each_iteration)
    total_number_class = int(total_number_class)
    print("total_number_class", total_number_class)
    number_of_iterations = int(number_of_iterations)
    if divide_number_each_iteration ** number_of_iterations != total_number_class:
        raise AssertionError("the combination is not valid")
    GT_code_infos = [divide_number_each_iteration, number_of_iterations, total_number_class]

    if divide_number_each_iteration != 2 and (BinaryCode_Loss_Type=='BCE' or BinaryCode_Loss_Type=='L1'):
        raise AssertionError("for non-binary case, use CE as loss function")
    if divide_number_each_iteration == 2 and BinaryCode_Loss_Type=='CE':
        raise AssertionError("not support for now")

    vertices = inout.load_ply(mesh_path)["pts"]

    # define test data loader
    if not bop_challange:
        dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_,camera_params_test = bop_io.get_dataset(
            bop_path, dataset_name,train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, 
            train_obj_visible_threshold=train_obj_visible_threshold
            )
    else:
        print("use BOP test images")
        dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_,camera_params_test = bop_io.get_bop_challange_test_data(
            bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder
            )

    has_gt = True
    if test_gts[obj_id][0] == None:
        has_gt = False
    # currently no detections are available for husky dataset
    Det_Bbox = None

    test_dataset = bop_dataset_single_obj_pytorch(
        dataset_dir_test, test_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, 
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
    )
    print("test image example:", test_rgb_files[obj_id][0], flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    # binary_code_length = number_of_iterations
    # print("predicted binary_code_length", binary_code_length)
    # configs['binary_code_length'] = binary_code_length
 
    resnet_layer = 34
    binary_code_length = 16
    concat = True
    divide_number_each_iteration = 2
    output_kernel_size = 1
    efficientnet_key = None
    checkpoint_file='/media/pmvanderburg/T7/bop_datasets/6dof_pose_experiments/experiments/checkpoints/exp_husky_BOP_4xWorkers_8xBatch__1xGPUx32G_1xTasks_8xBatchTest_EntireMaskobj07/best_score/0_9850step4000'
        
    net = BinaryCodeNet_Deeplab(
                num_resnet_layers=resnet_layer, 
                concat=concat, 
                binary_code_length=binary_code_length, 
                divided_number_each_iteration = divide_number_each_iteration, 
                output_kernel_size = output_kernel_size,
                efficientnet_key = efficientnet_key
            )

    if torch.cuda.is_available():
        net=net.cuda()

    # checkpoint = torch.load( configs['checkpoint_file'] )
    checkpoint = torch.load( checkpoint_file )
    
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()
    
    ### wrong auc threshold linspace ###
    # th = np.linspace(10, 100, num=10)
    
    #test with test data
    # ADX_passed=np.zeros(len(test_loader.dataset))
    # ADX_error=np.zeros(len(test_loader.dataset))
    # #AUC_ADX_error1=np.zeros(len(test_loader.dataset))
    # #AUC_ADX_error2=np.zeros(len(test_loader.dataset))
    # AUC_ADX_passed_orig=np.zeros((auc_resolution,len(test_loader.dataset)))
    # AUC_ADX_passed=np.zeros((num_th,auc_resolution,len(test_loader.dataset)))
    # if calc_add_and_adi:
    #     ADY_passed=np.zeros(len(test_loader.dataset))
    #     ADY_error=np.zeros(len(test_loader.dataset))
    #     AUC_ADY_error=np.zeros(len(test_loader.dataset))

    # print("test dataset")
    # print(len(test_loader.dataset))
    total_images = len(test_loader.dataset)
    
    if ignore_bit!=0:
        new_dict_class_id_3D_points = generate_new_corres_dict(dict_class_id_3D_points, 16, 16-ignore_bit)
    
    img_ids = []
    scene_ids = []
    estimated_Rs = []
    estimated_Ts = []
    for rgb_fn in test_rgb_files[obj_id]:
        rgb_fn = rgb_fn.split("/")
        scene_id = rgb_fn[-3]
        img_id = rgb_fn[-1].split(".")[0]
        img_ids.append(img_id)
        scene_ids.append(scene_id)

    # if configs['use_icp']:
    #     #init the ICP Refiner
    #     from icp_module.ICP_Cosypose import ICPRefiner, read_depth
    #     test_img = cv2.imread(test_rgb_files[obj_id][0])
    #     icp_refiner = ICPRefiner(mesh_path, test_img.shape[1], test_img.shape[0], num_iters=100)

    ### GPU warm up as described by https://towardsdatascience.com/the-correct-way-to-measure-inference-time-of-deep-neural-networks-304a54e5187f
    # device = torch.device(“cuda”)
    dummy_input = torch.randn(1, 3,256,256,dtype=torch.float).cuda()
    # GPU-WARM-UP
    for _ in range(10):
        _ = net(dummy_input)
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 30
    timings=np.zeros((repetitions,1))
    
    # with torch.no_grad():    
    #     for rep in range(repetitions):
    #         # starter.record()
    #         pred_mask_prob, pred_code_prob = net(dummy_input)
    #         starter.record()
    #         pred_masks = from_output_to_class_mask(pred_mask_prob)
    #         # print('pred_masks shape',pred_masks.shape) # (1, 1, 128, 128)
    #         pred_code_images = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_iteration, binary_code_length=binary_code_length)
        
    #         # print('pred_code_images shape',pred_code_images.shape) # (1, 16, 128, 128)
    #         # from binary code to pose
    #         pred_code_images = pred_code_images.transpose(0, 2, 3, 1) # (1, 128, 128, 16)
        

    #         pred_masks = pred_masks.transpose(0, 2, 3, 1)
    #         # print('pred_masks shape after transpose(0, 2, 3, 1)',pred_masks.shape) # (1, 128, 128, 1)
    #         pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')
    #         # print('pred_masks shape after squeeze(axis=-1).astype(uint8)',pred_masks.shape) # (1, 128, 128) the last one was removed
        
    #         ender.record()
    #         # WAIT FOR GPU SYNC
    #         torch.cuda.synchronize()
    #         curr_time = starter.elapsed_time(ender)
    #         timings[rep] = curr_time
    # mean_syn = np.sum(timings) / repetitions
    # std_syn = np.std(timings)
    # print('mean_syn prob to pred',mean_syn)
    # print('std',std_syn)
    
    timings=np.zeros((repetitions,1))
    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(test_loader)):
        starter.record()
        
        if torch.cuda.is_available():
            data=data.cuda()
            masks = masks.cuda()
            class_code_images = class_code_images.cuda()

        # print('data shape',data.shape)
        if batch_idx==repetitions:
            break
        # Get CNN output
        pred_mask_prob, pred_code_prob = net(data)
        
        # print('pred_mask_prob shape',pred_mask_prob.shape) # (1, 16, 128, 128)
        # print('pred_code_prob shape',pred_code_prob.shape) # (1, 1, 128, 128)
        # print('data shape',data.shape) # (1, 3, 256,256)
        
        # starter.record()
        
        pred_masks = from_output_to_class_mask(pred_mask_prob)
        # print('pred_masks shape',pred_masks.shape) # (1, 1, 128, 128)
        pred_code_images = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_iteration, binary_code_length=binary_code_length)
        
        # print('pred_code_images shape',pred_code_images.shape) # (1, 16, 128, 128)
        # from binary code to pose
        pred_code_images = pred_code_images.transpose(0, 2, 3, 1) # (1, 128, 128, 16)
        

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        # print('pred_masks shape after transpose(0, 2, 3, 1)',pred_masks.shape) # (1, 128, 128, 1)
        pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')
        # print('pred_masks shape after squeeze(axis=-1).astype(uint8)',pred_masks.shape) # (1, 128, 128) the last one was removed
        
        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        Bboxes = Bboxes.detach().cpu().numpy()
        class_code_images = class_code_images.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(axis=0).astype('uint8')
        # print('class_code_images shape after transpose(0, 2, 3, 1).squeeze(axis=0)',class_code_images.shape) # (128, 128, 16)
        
        # Visualisations
        visualization.visualise_tensor(data, 'data', num_ch=1,batch_id=batch_idx,eval_output_path=eval_output_path)
        visualization.visualise_tensor(pred_code_prob, 'pred_code_prob from net(data)', num_ch=16,batch_id=batch_idx,eval_output_path=eval_output_path)
        visualization.visualise_tensor(pred_mask_prob, 'pred_mask_prob from net(data)', num_ch=1, batch_id=batch_idx,eval_output_path=eval_output_path)
        # visualization.visualise_tensor(pred_masks, 'pred_masks=from_output_to_class_mask(pred_mask_prob)', 1, batch_idx, eval_output_path)
        # visualization.visualise_tensor(class_code_images, 'class_code_images from test_loader', 16,batch_idx,eval_output_path)
        # visualization.visualise_tensor(pred_code_images.transpose(1,2,3,0), 'pred_code_images from_output_to_class_binary_code(pred_code_prob', 16, batch_idx, eval_output_path)
        
        # time.sleep(1)
        # ender.record()
        # starter.record()
        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            # print('pred_code_images[',counter,'].shape',pred_code_images[counter].shape) # (128, 128, 16) counter is 1 since batch_size=1
            if ignore_bit!=0:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter][:,:,:-ignore_bit],
                                                                            Bbox, BoundingBox_CropSize_GT, ProgX, divide_number_each_iteration, new_dict_class_id_3D_points,
                                                                            intrinsic_matrix=cam_K)
            else:   
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter],
                                                                            Bbox, BoundingBox_CropSize_GT, ProgX, divide_number_each_iteration, dict_class_id_3D_points,
                                                                            intrinsic_matrix=cam_K)
                # visualization.visualise_tensor(class_id_image, 'class_id_image from class_corresp', 0, batch_idx, eval_output_path)
            
            if success:
                # if configs['use_icp']:
                    # add icp refinement and replace R_predict, t_predict
                    # depth_image = read_depth(test_depth_files[obj_id][batch_idx])
                    # if dataset_name == 'ycbv' or dataset_name == 'tless':
                    #     depth_image = depth_image * 0.1
                    # full_mask = compute_original_mask(Bbox, test_img.shape[0], test_img.shape[1], pred_masks[counter])
                    # R_refined, t_refined = icp_refiner.refine_poses(t_predict*0.1, R_predict, full_mask, depth_image, cam_K.cpu().detach().numpy())
                    # R_predict = R_refined
                    # t_predict = t_refined*10.
                    # t_predict = t_predict.reshape((3,1))

                estimated_Rs.append(R_predict)
                estimated_Ts.append(t_predict)
            else:
                print('no success :O \nzeroing the estimated Rs and Ts')
                R_ = np.zeros((3,3))
                R_[0,0] = 1
                R_[1,1] = 1
                R_[2,2] = 1
                estimated_Rs.append(R_)
                estimated_Ts.append(np.zeros((3,1)))
        ender.record()
        # WAIT FOR GPU SYNC
        torch.cuda.synchronize()
        curr_time = starter.elapsed_time(ender)
        timings[batch_idx] = curr_time
        
            # adx_error = 10000
            # if success and has_gt:
                # adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                # if np.isnan(adx_error):
                    # adx_error = 10000
    print('mean of timing',np.mean(timings))
             
    if Det_Bbox == None:         
        scores = [1 for x in range(len(estimated_Rs))]
    cvs_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(cvs_path):
        os.makedirs(cvs_path)

    write_to_csv.write_csv(cvs_path, "{}_{}".format(dataset_name, obj_name), obj_id+1, scene_ids, img_ids, estimated_Rs, estimated_Ts, scores)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    # parser.add_argument('--cfg', type=str) # config file
    # parser.add_argument('--obj_name', type=str)
    # parser.add_argument('--ckpt_file', type=str)
    # parser.add_argument('--ignore_bit', type=str)
    # parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_icp', type=str, choices=('True','False'), default='False') # config file
    args = parser.parse_args()
    # config_file = args.cfg
    # checkpoint_file = args.ckpt_file
    # eval_output_path = args.eval_output_path
    # #eval_output_path = /media/pmvanderburg/T7/bop_datasets # args.eval_output_path
    # obj_name = args.obj_name
    args.obj_name = 'obj07'
    args.ignore_bit = 0 
    obj_name = args.obj_name
    
    config_file = '/home/pmvanderburg/ZebraPose/zebrapose/config/config_BOP/husky/exp_husky_BOP.txt'
    
    configs = parse_cfg(config_file)

    # checkpoint_file = '/media/pmvanderburg/T7/bop_datasets/6dof_pose_experiments/experiments/checkpoints/exp_husky_BOP_4xWorkers_8xBatch__1xGPUx32G_1xTasks_8xBatchTest_EntireMaskobj07/best_score/0_9850step4000'
    checkpoint_file = '/media/pmvanderburg/T7/bop_datasets/6dof_pose_experiments/experiments/checkpoints/exp_husky_BOP_4xWorkers_8xBatch__1xGPUx32G_1xTasks_8xBatchTest_EntireMaskobj07/best_score/0_9850step4000'
    eval_output_path = '/home/pmvanderburg/6dof_pose_experiments/20230220_test_debug' # args.eval_output_path
    
    configs['use_icp'] = False # (args.use_icp == 'True')
    configs['obj_name'] = obj_name

    if configs['Detection_results'] != 'none':
        Detection_results = configs['Detection_results']
        dirname = os.path.dirname(__file__)
        Detection_results = os.path.join(dirname, Detection_results)
        configs['Detection_results'] = Detection_results

    configs['checkpoint_file'] = checkpoint_file
    configs['eval_output_path'] = eval_output_path

    configs['ignore_bit'] = int(args.ignore_bit)

    #print the configurations
    for key in configs:
        print(key, " : ", configs[key], flush=True)

    main(configs)
