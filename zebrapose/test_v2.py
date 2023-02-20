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

import matplotlib.pyplot as plt

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

def visualise_tensor(tensor_gpu, ch=0, allkernels=False, nrows=8, ncols=8):
    """
    # 1. The function visualise_tensor() takes the following arguments:

    # tensor_gpu: the tensor to be visualised
    # ch: the channel to be visualised, if ch=0, all channels are visualised
    # allkernels: if True, all kernels are visualised
    # nrows: number of rows of subplots to be used
    # ncols: number of columns of subplots to be used

    """
    #tensorcopy=tensor.copy()
    tensor_cpu = tensor_gpu.detach().cpu()
    # tensor=tensor_cpu.permute(1,0,2,3)
    tensor=tensor_cpu.numpy()
    # print('tensor t type and shape', type(tensor),tensor.shape)
    if allkernels: tensor = tensor.view((1,-1) + tensor.shape[-2:])
    elif ch >= 0: tensor = tensor[:,ch,:,:]
    elif ch < 0:
        c = int(tensor.shape[1] / abs(ch))
        row = int(c / nrows) if (c % nrows == 0) else int(c / nrows) + 1
        col = min(ncols, c) if (c < ncols) else c % ncols
        print('row={}, col={}'.format(row, col))     # debug only !!!

    # print('tensor t type and shape', type(tensor),tensor.shape)
    # for kernel in tensor:
        # print(kernel)
    kernels = np.array([kernel for kernel in tensor])
    print('kernels.shape',kernels.shape)
    fig = plt.figure()
    row = col = 4
    for i in range((row * col)):   # number of subplots to show all kernels/filters of a layer !!!

        ax1 = fig.add_subplot(row, col , i+1 )   # add subplot to figure with index i+1 and size row x column !!!

        ax1.imshow((kernels[i]).reshape((32 , 32)), cmap='gray')   # show image on subplot with index i+1 and size row x column !!!

        ax1.axis('off')   # remove axis from the plot/subplot with index i+1 and size row x column !!!

        ax1.set_xticklabels([])   # remove tick labels from the plot/subplot with index i+! and size row x column !!!

        ax1.set_yticklabels([])   # remove tick labels from the plot/subplot with index i+! and size row x column !!!

    plt.tight_layout()      ## adjust spacing between subplots to minimize the overlaps !!

def visualise(pred_mask_prob, pred_code_prob):
    #cv2.namedWindow('rgb', #cv2.WINDOW_NORMAL)
    #cv2.namedWindow('mask', #cv2.WINDOW_NORMAL)
    #cv2.namedWindow('entire_mask', #cv2.WINDOW_NORMAL)
    #cv2.namedWindow('GT_img_visible', #cv2.WINDOW_NORMAL)
    #cv2.namedWindow('GT_img_invisible', #cv2.WINDOW_NORMAL)
    ### PMB DEBUG ###
    print('pred_mask_prob',type(pred_mask_prob), pred_mask_prob.shape)
    print('pred_code_prob',type(pred_code_prob), pred_code_prob.shape)
    #################
    #x_ = x.copy()
    #if Bbox is not None:
        #cv2.rectangle(x_,(Bbox[0],Bbox[1]),(Bbox[0]+Bbox[2] ,Bbox[1]+Bbox[3] ),(0,255,0),3)
    #cv2.imshow('rgb',x_)
    #cv2.imshow('mask',mask)
    #cv2.imshow('entire_mask',entire_mask)

    #cv2.imshow('GT_img_visible',GT_img_visible)
    #cv2.imshow('GT_img_invisible',GT_img_invisible)

    #cv2.waitKey(0)

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap


def compute_auc_posecnn(errors):
    # NOTE: Adapted from https://github.com/yuxng/YCB_Video_toolbox/blob/master/evaluate_poses_keyframe.m
    errors = errors.copy()
    d = np.sort(errors)
    d[d > 0.1] = np.inf
    accuracy = np.cumsum(np.ones(d.shape[0])) / d.shape[0]
    ids = np.isfinite(d)
    d = d[ids]
    accuracy = accuracy[ids]
    if len(ids) == 0 or ids.sum() == 0:
        return np.nan
    rec = d
    prec = accuracy
    mrec = np.concatenate(([0], rec, [0.1]))
    mpre = np.concatenate(([0], prec, [prec[-1]]))
    for i in np.arange(1, len(mpre)):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.arange(1, len(mpre))
    ids = np.where(mrec[1:] != mrec[:-1])[0] + 1
    ap = ((mrec[ids] - mrec[ids-1]) * mpre[ids]).sum() * 10
    return ap

def calc_AUC(adx_error, diameter_threshold, obj_diameter, res, batch_idx, AUC_ADX_passed, num_th): #, AUC_ADX_error):    # error thresholds and AUC definitions
    
    auc_resolution=res
    abs_error_threshold = np.zeros(num_th)
    #abs_error_threshold_base = obj_diameter * diameter_threshold
    #for i in range(0, num_th):
    #    abs_error_threshold[i] =
    abs_error_threshold[0] = obj_diameter * diameter_threshold
    abs_error_threshold[1] = obj_diameter * diameter_threshold * 0.5
    abs_error_threshold[2] = obj_diameter * diameter_threshold * 0.25
    
    AUC_lower = 0.1 * abs_error_threshold
    AUC_upper = abs_error_threshold
    th = np.zeros((num_th,auc_resolution))
    for i in range(0,num_th):
        th[i] = np.linspace(0.1*(abs_error_threshold[i]), abs_error_threshold[i], num=auc_resolution)
    #print(th)
    # th2 = np.linspace(0.1*(abs_error_threshold2), abs_error_threshold2, num=auc_resolution)
    # th3 = np.linspace(0.1*(abs_error_threshold3), abs_error_threshold3, num=auc_resolution)

    ### correct AUC calculation ###

    # for i in range(0,len(th)):
    th_idx = 0
    for i in th:
        #sum_correct = 0
        AUC_idx = 0        
        for j in i:
            if adx_error < j:
                AUC_ADX_passed[th_idx][AUC_idx][batch_idx] = 1
                #sum_correct = sum_correct + 1
            AUC_idx = AUC_idx + 1
        th_idx = th_idx + 1
    #print(AUC_ADX_passed[:,:,batch_idx])
    
    return AUC_ADX_passed # , AUC_ADX_error


def main(configs):
    #### training dataset
    bop_challange = configs['bop_challange']
    bop_path = configs['bop_path']
    obj_name = configs['obj_name']
    dataset_name = configs['dataset_name']
    training_data_folder=configs['training_data_folder']
    training_data_folder_2=configs['training_data_folder_2']
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
    diameter_threshold = 0.1 # float(configs['diameter_threshold'])
    auc_resolution = 10
    num_th = 3
    calc_add_and_adi=True

    #### augmentations
    Detection_results=configs['Detection_results']                       # for the test, the detected bounding box provided by GDR Net
    padding_ratio=configs['padding_ratio']                               # pad the bounding box for training and test
    resize_method = configs['resize_method']                             # how to resize the roi images to 256*256
    use_peper_salt= configs['use_peper_salt']                            # if add additional peper_salt in the augmentation
    use_motion_blur= configs['use_motion_blur']                          # if add additional motion_blur in the augmentation
    # pixel code settings
    divide_number_each_iteration = configs['divide_number_each_iteration']
    number_of_iterations = configs['number_of_iterations']



    torch.manual_seed(0)      # the both are only good for ablation study
    np.random.seed(0)         # if can be removed in the final experiments



    # get dataset informations
    dataset_dir,source_dir,model_plys,model_info,model_ids,rgb_files,depth_files,mask_files,mask_visib_files,gts,gt_infos,cam_param_global, cam_params = bop_io.get_dataset(bop_path,dataset_name, train=True, data_folder=training_data_folder, data_per_obj=True, incl_param=True, train_obj_visible_threshold=train_obj_visible_threshold)
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
        dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_dataset(bop_path, dataset_name,train=False, data_folder=test_folder, data_per_obj=True, incl_param=True, train_obj_visible_threshold=train_obj_visible_threshold)
        if dataset_name == 'ycbv':
            print("select key frames from ycbv test images")
            key_frame_index = ycbv_select_keyframe(Detection_results, test_rgb_files[obj_id])
            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
            test_depth_files_keyframe = [test_depth_files[obj_id][i] for i in key_frame_index]
            test_rgb_files[obj_id] = test_rgb_files_keyframe
            test_mask_files[obj_id] = test_mask_files_keyframe
            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
            test_gts[obj_id] = test_gts_keyframe
            test_gt_infos[obj_id] = test_gt_infos_keyframe
            camera_params_test[obj_id] = camera_params_test_keyframe
            test_depth_files[obj_id] = test_depth_files_keyframe
    else:
        print("use BOP test images")
        dataset_dir_test,_,_,_,_,test_rgb_files,test_depth_files,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_, camera_params_test = bop_io.get_bop_challange_test_data(bop_path, dataset_name, target_obj_id=obj_id+1, data_folder=test_folder)

    has_gt = True
    if test_gts[obj_id][0] == None:
        has_gt = False

    ### PMB DEBUG ###
    print('type of test_rgb_files[obj_id] to see what is rgb_fn in get_detection_results', type(test_rgb_files[obj_id]))
    print('test_rgb_files[obj_id][0:3] to see what is rgb_fn in get_detection_results', test_rgb_files[obj_id])
    #################
    if Detection_results != 'none':
        if configs['detector']=='FCOS':
            from get_detection_results_pmb_debug import get_detection_results, get_detection_scores
        elif configs['detector']=='MASKRCNN':
            from get_mask_rcnn_results import get_detection_results, get_detection_scores
        Det_Bbox = get_detection_results(Detection_results, test_rgb_files[obj_id], obj_id+1, 0)
        scores = get_detection_scores(Detection_results, test_rgb_files[obj_id], obj_id+1, 0)
    else:
        Det_Bbox = None
    ### PMB DEBUG ###
    print('Det_Bbox', Det_Bbox)
    #################
    test_dataset = bop_dataset_single_obj_pytorch(
        dataset_dir_test, test_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id], 
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, 
        BoundingBox_CropSize_image, BoundingBox_CropSize_GT, GT_code_infos, 
        padding_ratio=padding_ratio, resize_method=resize_method, Detect_Bbox=Det_Bbox,
        use_peper_salt=use_peper_salt, use_motion_blur=use_motion_blur
                                        )
    print("test image example:", test_rgb_files[obj_id][0], flush=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    binary_code_length = number_of_iterations
    print("predicted binary_code_length", binary_code_length)
    configs['binary_code_length'] = binary_code_length
 
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

    checkpoint = torch.load( configs['checkpoint_file'] )
    net.load_state_dict(checkpoint['model_state_dict'])

    net.eval()

    
    ### wrong auc threshold linspace ###
    th = np.linspace(10, 100, num=10)
    
    #test with test data
    ADX_passed=np.zeros(len(test_loader.dataset))
    ADX_error=np.zeros(len(test_loader.dataset))
    #AUC_ADX_error1=np.zeros(len(test_loader.dataset))
    #AUC_ADX_error2=np.zeros(len(test_loader.dataset))
    AUC_ADX_passed_orig=np.zeros((auc_resolution,len(test_loader.dataset)))
    AUC_ADX_passed=np.zeros((num_th,auc_resolution,len(test_loader.dataset)))
    if calc_add_and_adi:
        ADY_passed=np.zeros(len(test_loader.dataset))
        ADY_error=np.zeros(len(test_loader.dataset))
        AUC_ADY_error=np.zeros(len(test_loader.dataset))

    print("test dataset")
    print(len(test_loader.dataset))
    total_images = len(test_loader.dataset)
    ignore_bit = configs['ignore_bit']
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

    if configs['use_icp']:
        #init the ICP Refiner
        from icp_module.ICP_Cosypose import ICPRefiner, read_depth
        test_img = cv2.imread(test_rgb_files[obj_id][0])
        icp_refiner = ICPRefiner(mesh_path, test_img.shape[1], test_img.shape[0], num_iters=100)

    for batch_idx, (data, entire_masks, masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(test_loader)):
        if torch.cuda.is_available():
            data=data.cuda()
            masks = masks.cuda()
            class_code_images = class_code_images.cuda()

        pred_mask_prob, pred_code_prob = net(data)
        # visualise(pred_mask_prob, pred_code_prob)
        visualise_tensor(pred_code_prob,1, False)
        time.sleep(10)
        pred_masks = from_output_to_class_mask(pred_mask_prob)
        pred_code_images = from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, divided_num_each_interation=divide_number_each_iteration, binary_code_length=binary_code_length)
       
        # from binary code to pose
        pred_code_images = pred_code_images.transpose(0, 2, 3, 1)

        pred_masks = pred_masks.transpose(0, 2, 3, 1)
        pred_masks = pred_masks.squeeze(axis=-1).astype('uint8')

        Rs = Rs.detach().cpu().numpy()
        ts = ts.detach().cpu().numpy()
        Bboxes = Bboxes.detach().cpu().numpy()

        class_code_images = class_code_images.detach().cpu().numpy().transpose(0, 2, 3, 1).squeeze(axis=0).astype('uint8')
        
        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            if ignore_bit!=0:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter][:,:,:-ignore_bit],
                                                                            Bbox, BoundingBox_CropSize_GT, ProgX, divide_number_each_iteration, new_dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
            else:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_code_images[counter], 
                                                                            Bbox, BoundingBox_CropSize_GT, ProgX, divide_number_each_iteration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
        
            if success:
                if configs['use_icp']:
                    # add icp refinement and replace R_predict, t_predict
                    depth_image = read_depth(test_depth_files[obj_id][batch_idx])
                    if dataset_name == 'ycbv' or dataset_name == 'tless':
                        depth_image = depth_image * 0.1
                    full_mask = compute_original_mask(Bbox, test_img.shape[0], test_img.shape[1], pred_masks[counter])
                    R_refined, t_refined = icp_refiner.refine_poses(t_predict*0.1, R_predict, full_mask, depth_image, cam_K.cpu().detach().numpy())
                    R_predict = R_refined
                    t_predict = t_refined*10.
                    t_predict = t_predict.reshape((3,1))

                estimated_Rs.append(R_predict)
                estimated_Ts.append(t_predict)
            else:
                R_ = np.zeros((3,3))
                R_[0,0] = 1
                R_[1,1] = 1
                R_[2,2] = 1
                estimated_Rs.append(R_)
                estimated_Ts.append(np.zeros((3,1)))

            adx_error = 10000
            if success and has_gt:
                adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(adx_error):
                    adx_error = 10000
            
            if adx_error < obj_diameter * diameter_threshold:
                ADX_passed[batch_idx] = 1

            ADX_error[batch_idx] = adx_error
                        
            AUC_ADX_passed = calc_AUC(adx_error, diameter_threshold, obj_diameter, auc_resolution, batch_idx, AUC_ADX_passed,num_th)
            
            if calc_add_and_adi:
                ady_error = 10000
                if success and has_gt:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*diameter_threshold:
                    ADY_passed[batch_idx] = 1
                ADY_error[batch_idx] = ady_error
               
                th = np.linspace(10, 100, num=10)
                sum_correct = 0
                for t in th:
                    if ady_error < t:
                        sum_correct = sum_correct + 1
                AUC_ADY_error[batch_idx] = sum_correct/10
             
    if Det_Bbox == None:         
        scores = [1 for x in range(len(estimated_Rs))]
    csv_path = os.path.join(eval_output_path, 'pose_result_bop/')
    if not os.path.exists(csv_path):
        os.makedirs(csv_path)

    write_to_csv.write_csv(csv_path, "{}_{}".format(dataset_name, obj_name), obj_id+1, scene_ids, img_ids, estimated_Rs, estimated_Ts, scores)
    
    ADX_passed = np.mean(ADX_passed)
    ADX_error_mean= np.mean(ADX_error)
    #print('{}/{}_mean'.format(main_metric_name,main_metric_name), ADX_error_mean,'mm')
    #AUC_ADX_error1 = np.mean(AUC_ADX_error1)
    print('{}/{}'.format(main_metric_name,main_metric_name), ADX_passed)
    #print('erroneous AUC_1_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error1)
    #AUC_ADX_error2 = np.mean(AUC_ADX_error2)
    # print('{}/{}'.format(main_metric_name,main_metric_name), ADX_passed2)
    #print('Correct AUC_2_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error2)
    
    ### correct AUC calculation ###
    # AUC_ADX_passed[batch_idx][AUC_idx]
    AUC_cumulative = np.zeros((num_th,auc_resolution))
    AUC_ADX = np.zeros(num_th)
    
    for i,th_grid in enumerate(AUC_ADX_passed):
        for j, passed in enumerate(th_grid):
            AUC_cumulative[i,j] = np.mean(th_grid[j])

    ### visualize AUC graph ###
    # visualization.AUC_graph(AUC_cumulative, auc_resolution, diameter_threshold)

    for i in range(0,num_th):
        AUC_ADX[i] = np.mean(AUC_ADX_passed[i])
    print('{}/{}_10%'.format(main_metric_name,main_metric_name), AUC_cumulative[0,9])
    print('{}/{}_5%'.format(main_metric_name,main_metric_name), AUC_cumulative[1,9])
    print('{}/{}_2.5%'.format(main_metric_name,main_metric_name), AUC_cumulative[2,9])
    print('AUC_10%_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX[0])
    print('AUC_5%_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX[1])
    print('AUC_2.5%_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX[2])
    print('Cumulative AUC_10%{}/{}'.format(main_metric_name,main_metric_name), AUC_cumulative[0])
    print('Cumulative AUC_5%{}/{}'.format(main_metric_name,main_metric_name), AUC_cumulative[1])
    print('Cumulative AUC_2.5%{}/{}'.format(main_metric_name,main_metric_name), AUC_cumulative[2])
    #AUC_ADX_error_posecnn = compute_auc_posecnn(ADX_error/1000.)
    #print('AUC_posecnn_{}/{}'.format(main_metric_name,main_metric_name), AUC_ADX_error_posecnn)

    if calc_add_and_adi:
        ADY_passed = np.mean(ADY_passed)
        ADY_error_mean= np.mean(ADY_error)
        AUC_ADY_error = np.mean(AUC_ADY_error)
        print('{}/{}'.format(supp_metric_name,supp_metric_name), ADY_passed)
        print('AUC_{}/{}'.format(supp_metric_name,supp_metric_name), AUC_ADY_error)
        AUC_ADY_error_posecnn = compute_auc_posecnn(ADY_error/1000.)
        print('AUC_posecnn_{}/{}'.format(supp_metric_name,supp_metric_name), AUC_ADY_error_posecnn)

    ####save results to file
    if has_gt:
        path = os.path.join(eval_output_path, "ADD_result/")
        if not os.path.exists(path):
            os.makedirs(path)
        path1 = path + "{}_{}".format(dataset_name, obj_name) + ".txt" 
        path_AUC = path + "{}_{}_{}".format(dataset_name, obj_name,diameter_threshold) + "_AUC.npy" 
        #path = path + dataset_name + obj_name  + "ignorebit_" + str(configs['ignore_bit']) + ".txt"
        #path = path + dataset_name + obj_name + "radix" + "_" + str(divide_number_each_iteration)+"_"+str(number_of_iterations) + ".txt"
        print('save ADD results to', path1)
        print(path1)
        f = open(path1, "w")
        #auc = open(path_AUC, "w")
        ### ADD ###
        f.write('{}/{}_10% '.format(main_metric_name,main_metric_name))
        f.write(str(ADX_passed.item()))
        f.write('\n')
        ### ADD ###
        f.write('{}/{}_10% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[0,9].item()))
        f.write('\n')
        ### ADD ###
        f.write('{}/{}_5% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[1,9].item()))
        f.write('\n')
        ### ADD ###
        f.write('{}/{}_2.5% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[2,9].item()))
        f.write('\n')
        ### correct AUC ADD ###
        #f.write('AUC_10%_{}/{} '.format(main_metric_name,main_metric_name))
        #f.write(str(AUC_ADX_error2.item()))
        #f.write('\n')
        ### Cumulative AUC ADD ###
        f.write('Cumulative AUC_{}/{}_10% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[0])) #.item()))
        f.write('\n')
        f.write('Cumulative AUC_{}/{}_5% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[1])) #.item()))
        f.write('\n')
        f.write('Cumulative AUC_{}/{}_2.5% '.format(main_metric_name,main_metric_name))
        f.write(str(AUC_cumulative[2])) #.item()))
        f.write('\n')
        np.save(path_AUC,AUC_cumulative)
        ### AUC ADD ###
        #f.write('Erroneous AUC_{}/{} '.format(main_metric_name,main_metric_name))
        #f.write(str(AUC_ADX_error1.item()))
        #f.write('\n')
        ### posecnn AUC ###
        #f.write('Erroneous AUC_posecnn_{}/{} '.format(main_metric_name,main_metric_name))
        #f.write(str(AUC_ADX_error_posecnn.item()))
        #f.write('\n')
        ### AUC ADI ###
        #f.write('AUC_{}/{} '.format(supp_metric_name,supp_metric_name))
        #f.write(str(AUC_ADY_error.item()))
        #f.write('\n')
        ### AUC posecnn ADI
        #f.write('AUC_posecnn_{}/{} '.format(main_metric_name,main_metric_name))
        #f.write(str(AUC_ADY_error_posecnn.item()))
        #f.write('\n')
        ####
        f.close()
        ####

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='BinaryCodeNet')
    parser.add_argument('--cfg', type=str) # config file
    parser.add_argument('--obj_name', type=str)
    parser.add_argument('--ckpt_file', type=str)
    parser.add_argument('--ignore_bit', type=str)
    parser.add_argument('--eval_output_path', type=str)
    parser.add_argument('--use_icp', type=str, choices=('True','False'), default='False') # config file
    args = parser.parse_args()
    config_file = args.cfg
    checkpoint_file = args.ckpt_file
    eval_output_path = args.eval_output_path
    #eval_output_path = /media/pmvanderburg/T7/bop_datasets # args.eval_output_path
    obj_name = args.obj_name
    configs = parse_cfg(config_file)

    configs['use_icp'] = (args.use_icp == 'True')
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
