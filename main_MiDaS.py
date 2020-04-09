import numpy as np
from mesh_utils import create_mesh, write_ply, read_ply, output_3d_photo, output_3d_video, output_pytorch3d_photo
import argparse
import networkx as nx
import glob
import os
import time
from functools import partial
import vispy
# vispy.use(app='egl')
from vispy import scene, io
from vispy.scene import visuals
from moviepy.editor import ImageSequenceClip
import scipy.misc as misc
from vispy.visuals.filters import Alpha
from tqdm import tqdm
from argparse import Namespace
import yaml
import sys
from utils import get_MiDaS_samples, read_MiDaS_depth, read_DPS_depth, get_samples, get_multiple_samples, get_rgb_samples
from utils import get_depth_samples, image_preprocessing, vis_depth_discontinuity, sparse_bilateral_filtering, depth_resize, canny, load_ckpt, read_iphone_depth, vis_depth_edge_connectivity, follow_image_aspect_ratio
from utils import diffuse_depth
from matplotlib import pyplot as plt
import torch
import cv2
from skimage.transform import resize
import time
import imageio
import copy
import shutil
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='argument.yml',help='Configure of post processing')
args = parser.parse_args()
config = yaml.load(open(args.config, 'r'))
sys.path.append(config['rgb_method_dir'])
sys.path.append(config['depth_method_dir'])
sys.path.append(config['depth_edge_method_dir'])
from EG_Config import EG_Config_template
from src.networks import Inpaint_Color_Net, Inpaint_Depth_Net, Inpaint_Edge_Net

os.makedirs(config['mesh_folder'], exist_ok=True)
os.makedirs(config['3P_folder'], exist_ok=True)
if config['refresh_dir'] is True:
    if os.path.exists(config['dst_folder']):
        shutil.rmtree(config['dst_folder'])
os.makedirs(config['dst_folder'], exist_ok=True)
os.makedirs(config['pickle_folder'], exist_ok=True)
if 'midas' in config['depth_source'].lower():
    sample_list = get_MiDaS_samples(config['src_folder'], config['depth_folder'], config['data_list_fi'], config, config['specific'], config['after_certain'])
elif 'dps' in config['depth_source'].lower():
    sample_list = get_multiple_samples(config['src_folder'], config['tgt_folder'], config['depth_folder'], config['data_list_fi'], config['img_format'], config['specific'], config['after_certain'], config=config)
normal_canvas, all_canvas = None, None
for idx in tqdm(range(len(sample_list))):
    depth = None
    sample = sample_list[idx]
    print("Current Source ==> ", sample['src_pair_name'])
    pickle_fi = os.path.join(config['pickle_folder'], sample['src_pair_name'] + '.pkl')
    mesh_fi = os.path.join(config['mesh_folder'], sample['src_pair_name'] +'.ply')
    image = imageio.imread(sample['ref_img_fi'])
    config['output_h'], config['output_w'] = np.load(sample['depth_fi']).shape[:2]
    frac = config['longer_side_len'] / max(config['output_h'], config['output_w'])
    config['output_h'], config['output_w'] = int(config['output_h'] * frac), int(config['output_w'] * frac)
    config['original_h'], config['original_w'] = config['output_h'], config['output_w']
    if image.ndim == 2:
        image = image[..., None].repeat(3, -1)
    if image.ndim == 3:
        image = image[..., :3]
    if np.sum(np.abs(image[..., 0] - image[..., 1])) == 0 and np.sum(np.abs(image[..., 1] - image[..., 2])) == 0:
        config['gray_image'] = True
    else:
        config['gray_image'] = False
    if not(config['load_ply'] is True and os.path.exists(mesh_fi)):
        if not(config['load_pickle'] is True and os.path.exists(pickle_fi)):
            ## Start Depth and Depth Edge Processing ##
            print("Start Depth and Depth Edge Processing ...")
            image = cv2.resize(image, (config['output_w'], config['output_h']), interpolation=cv2.INTER_AREA)
            if 'DPS' in config['depth_source'].upper():
                depth = read_DPS_depth(sample['depth_fi'], config['output_h'], config['output_w'], bord=0)
            elif 'MIDAS' in config['depth_source'].upper():
                depth = read_MiDaS_depth(sample['depth_fi'], 3.0, config['output_h'], config['output_w'])
            ## Depth Map and Depth Discontinuity Pre-processing ##
            start_time = time.time()
            vis_photos, vis_depths = sparse_bilateral_filtering(depth.copy(), image.copy(), config, num_iter=config['sparse_iter'], spdb=False)
            depth = vis_depths[-1]
            dname = os.path.join(copy.deepcopy(config['dst_folder']), os.path.splitext(os.path.basename(copy.deepcopy(sample['tgt_name'])[0]))[0] + "_disp.png")
            cname = os.path.join(copy.deepcopy(config['dst_folder']), os.path.splitext(os.path.basename(copy.deepcopy(sample['tgt_name'])[0]))[0] + "_image.png")
            cmap = plt.get_cmap('plasma')
            cdisp = cmap((1. / depth) / (1. / depth).max())
            cmap = plt.get_cmap('plasma'); plt.imsave('depth_aft.png', cmap((1. / depth) / (1. / depth).max()))
            # plt.imshow(cdisp); plt.show()
            # plt.imsave(dname, cdisp)
            # plt.imsave(cname, image)
            # continue
            print("Sparse time = ", time.time() - start_time)
            model = None
            torch.cuda.empty_cache()
        else:
            depth = None
        ## Start Grouping Discontinuity and Iterative inpainting ##
        print("Start Running 3D_Photo ...")
        depth_edge_config = EG_Config_template({'USE_INSTANCE_NORM': True,
                                                'USE_SPECTRAL_NORM': True,
                                                'LRELU': 0.2})
        depth_edge_model = Inpaint_Edge_Net(init_weights=True, config=depth_edge_config)
        depth_edge_weight = torch.load(config['depth_edge_model_ckpt'])['generator']
        depth_edge_model.load_state_dict(depth_edge_weight)
        depth_edge_model = depth_edge_model.cuda(config['gpu_ids']['inpaint_model'])
        depth_edge_model.eval()
        depth_edge_model = depth_edge_model.cuda(config['gpu_ids']['inpaint_model'])
        
        depth_feat_model = Inpaint_Depth_Net()
        depth_feat_weight = torch.load(config['depth_feat_model_ckpt'])['generator']
        depth_feat_model.load_state_dict(depth_feat_weight, strict=True)
        depth_feat_model = depth_feat_model.cuda(config['gpu_ids']['inpaint_model'])
        depth_feat_model.eval()
        depth_feat_model = depth_feat_model.cuda(config['gpu_ids']['inpaint_model'])
        rgb_model = Inpaint_Color_Net()
        try:
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'])['model']
        except:
            rgb_feat_weight = torch.load(config['rgb_feat_model_ckpt'])['generator']
        rgb_model.load_state_dict(rgb_feat_weight)
        rgb_model.eval()
        rgb_model = rgb_model.cuda(config['gpu_ids']['inpaint_model'])
        graph = None
        start_time = time.time()
        rt_info = write_ply(image, 
                          depth,
                          sample['int_mtx'],
                          mesh_fi,
                          pickle_fi,
                          config,
                          rgb_model,
                          depth_edge_model,
                          depth_edge_model,
                          depth_feat_model)
        if rt_info is False:
            continue
        print("Write ply time = ", time.time() - start_time)
        rgb_model = None
        color_feat_model = None
        depth_edge_model = None
        depth_feat_model = None
        torch.cuda.empty_cache()
        start_time = time.time()
    if config['save_ply'] is True or config['load_ply'] is True:
        verts, colors, faces, Height, Width, hFov, vFov = read_ply(mesh_fi)
    else:
        verts, colors, faces, Height, Width, hFov, vFov = rt_info
    if config['inference_img'] is True:
        if config['inference_video'] is True:
            video_pose, video_basename = copy.deepcopy(sample['tgts_pose'][0]), sample['tgt_name']
        else:
            video_pose, video_basename = None, None
        if (config.get('original_h') is not None):
            top = (config.get('original_h') // 2 - sample['int_mtx'][1, 2] * config['output_h'])
            left = (config.get('original_w') // 2 - sample['int_mtx'][0, 2] * config['output_w'])
            down, right = top + config['output_h'], left + config['output_w']
            border = [int(xx) for xx in [top, down, left, right]]
        else:
            border = None
        normal_canvas, all_canvas = output_3d_photo(verts.copy(), colors.copy(), faces.copy(), copy.deepcopy(Height), copy.deepcopy(Width), copy.deepcopy(hFov), copy.deepcopy(vFov), 
                         copy.deepcopy(sample['tgt_name']), copy.deepcopy(sample['tgt_pose']), copy.deepcopy(sample['ref_pose']), copy.deepcopy(config['dst_folder']), 
                         image.copy(), copy.deepcopy(sample['int_mtx']), config, image, 
                         video_pose, video_basename, config.get('original_h'), config.get('original_w'), border=border, depth=depth, normal_canvas=normal_canvas, all_canvas=all_canvas)
