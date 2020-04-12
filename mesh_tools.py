import os
import numpy as np
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx

import json
import scipy.misc as misc
#import OpenEXR
import scipy.signal as signal
import matplotlib.pyplot as plt
import cv2
import scipy.misc as misc
from skimage import io
from functools import partial
from vispy import scene, io
from vispy.scene import visuals
from functools import reduce
# from moviepy.editor import ImageSequenceClip
import scipy.misc as misc
from vispy.visuals.filters import Alpha
import cv2
from skimage.transform import resize
import copy
import torch
import os
from utils import refine_depth_around_edge, smooth_cntsyn_gap
from utils import require_depth_edge, filter_irrelevant_edge_new, open_small_mask
from skimage.feature import canny
from scipy import ndimage
import time
import transforms3d

def relabel_node(mesh, nodes, cur_node, new_node):
    if cur_node == new_node:
        return mesh
    mesh.add_node(new_node)
    for key, value in nodes[cur_node].items():
        nodes[new_node][key] = value
    for ne in mesh.neighbors(cur_node):
        mesh.add_edge(new_node, ne)
    mesh.remove_node(cur_node)

    return mesh

def filter_edge(mesh, edge_ccs, config, invalid=False):
    context_ccs = [set() for _ in edge_ccs]
    mesh_nodes = mesh.nodes
    for edge_id, edge_cc in enumerate(edge_ccs):
        if config['context_thickness'] == 0:
            continue
        edge_group = {}
        for edge_node in edge_cc:
            far_nodes = mesh_nodes[edge_node].get('far')
            if far_nodes is None:
                continue
            for far_node in far_nodes:
                context_ccs[edge_id].add(far_node)
                if mesh_nodes[far_node].get('edge_id') is not None:
                    if edge_group.get(mesh_nodes[far_node]['edge_id']) is None:
                        edge_group[mesh_nodes[far_node]['edge_id']] = set()
                    edge_group[mesh_nodes[far_node]['edge_id']].add(far_node)
        if len(edge_cc) > 2:
            for edge_key in [*edge_group.keys()]:
                if len(edge_group[edge_key]) == 1:
                    context_ccs[edge_id].remove([*edge_group[edge_key]][0])
    valid_edge_ccs = []
    for xidx, yy in enumerate(edge_ccs):
        if invalid is not True and len(context_ccs[xidx]) > 0:
            # if len(context_ccs[xidx]) > 0:
            valid_edge_ccs.append(yy)
        elif invalid is True and len(context_ccs[xidx]) == 0:
            valid_edge_ccs.append(yy)
        else:
            valid_edge_ccs.append(set())
    # valid_edge_ccs = [yy for xidx, yy in enumerate(edge_ccs) if len(context_ccs[xidx]) > 0]

    return valid_edge_ccs

def extrapolate(global_mesh,
                info_on_pix,
                image,
                depth,
                other_edge_with_id,
                edge_map,
                edge_ccs,
                depth_edge_model,
                depth_feat_model,
                rgb_feat_model,
                config,
                direc='right-up'):
    h_off, w_off = global_mesh.graph['hoffset'], global_mesh.graph['woffset']
    noext_H, noext_W = global_mesh.graph['noext_H'], global_mesh.graph['noext_W']

    if "up" in direc.lower() and "-" not in direc.lower():
        all_anchor = [0, h_off + config['context_thickness'], w_off, w_off + noext_W]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [0, h_off, w_off, w_off + noext_W]
        context_anchor = [h_off, h_off + config['context_thickness'], w_off, w_off + noext_W]
        valid_line_anchor = [h_off, h_off + 1, w_off, w_off + noext_W]
        valid_anchor = [min(mask_anchor[0], context_anchor[0]), max(mask_anchor[1], context_anchor[1]),
                        min(mask_anchor[2], context_anchor[2]), max(mask_anchor[3], context_anchor[3])]
    elif "down" in direc.lower() and "-" not in direc.lower():
        all_anchor = [h_off + noext_H - config['context_thickness'], 2 * h_off + noext_H, w_off, w_off + noext_W]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [h_off + noext_H, 2 * h_off + noext_H, w_off, w_off + noext_W]
        context_anchor = [h_off + noext_H - config['context_thickness'], h_off + noext_H, w_off, w_off + noext_W]
        valid_line_anchor = [h_off + noext_H - 1, h_off + noext_H, w_off, w_off + noext_W]
        valid_anchor = [min(mask_anchor[0], context_anchor[0]), max(mask_anchor[1], context_anchor[1]),
                        min(mask_anchor[2], context_anchor[2]), max(mask_anchor[3], context_anchor[3])]
    elif "left" in direc.lower() and "-" not in direc.lower():
        all_anchor = [h_off, h_off + noext_H, 0, w_off + config['context_thickness']]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [h_off, h_off + noext_H, 0, w_off]
        context_anchor = [h_off, h_off + noext_H, w_off, w_off + config['context_thickness']]
        valid_line_anchor = [h_off, h_off + noext_H, w_off, w_off + 1]
        valid_anchor = [min(mask_anchor[0], context_anchor[0]), max(mask_anchor[1], context_anchor[1]),
                        min(mask_anchor[2], context_anchor[2]), max(mask_anchor[3], context_anchor[3])]
    elif "right" in direc.lower() and "-" not in direc.lower():
        all_anchor = [h_off, h_off + noext_H, w_off + noext_W - config['context_thickness'], 2 * w_off + noext_W]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [h_off, h_off + noext_H, w_off + noext_W, 2 * w_off + noext_W]
        context_anchor = [h_off, h_off + noext_H, w_off + noext_W - config['context_thickness'], w_off + noext_W]
        valid_line_anchor = [h_off, h_off + noext_H, w_off + noext_W - 1, w_off + noext_W]
        valid_anchor = [min(mask_anchor[0], context_anchor[0]), max(mask_anchor[1], context_anchor[1]),
                        min(mask_anchor[2], context_anchor[2]), max(mask_anchor[3], context_anchor[3])]
    elif "left" in direc.lower() and "up" in direc.lower() and "-" in direc.lower():
        all_anchor = [0, h_off + config['context_thickness'], 0, w_off + config['context_thickness']]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [0, h_off, 0, w_off]
        context_anchor = "inv-mask"
        valid_line_anchor = None
        valid_anchor = all_anchor
    elif "left" in direc.lower() and "down" in direc.lower() and "-" in direc.lower():
        all_anchor = [h_off + noext_H - config['context_thickness'], 2 * h_off + noext_H, 0, w_off + config['context_thickness']]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [h_off + noext_H, 2 * h_off + noext_H, 0, w_off]
        context_anchor = "inv-mask"
        valid_line_anchor = None
        valid_anchor = all_anchor
    elif "right" in direc.lower() and "up" in direc.lower() and "-" in direc.lower():
        all_anchor = [0, h_off + config['context_thickness'], w_off + noext_W - config['context_thickness'], 2 * w_off + noext_W]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [0, h_off, w_off + noext_W, 2 * w_off + noext_W]
        context_anchor = "inv-mask"
        valid_line_anchor = None
        valid_anchor = all_anchor
    elif "right" in direc.lower() and "down" in direc.lower() and "-" in direc.lower():
        all_anchor = [h_off + noext_H - config['context_thickness'], 2 * h_off + noext_H, w_off + noext_W - config['context_thickness'], 2 * w_off + noext_W]
        global_shift = [all_anchor[0], all_anchor[2]]
        mask_anchor = [h_off + noext_H, 2 * h_off + noext_H, w_off + noext_W, 2 * w_off + noext_W]
        context_anchor = "inv-mask"
        valid_line_anchor = None
        valid_anchor = all_anchor

    global_mask = np.zeros_like(depth)
    global_mask[mask_anchor[0]:mask_anchor[1],mask_anchor[2]:mask_anchor[3]] = 1
    mask = global_mask[valid_anchor[0]:valid_anchor[1], valid_anchor[2]:valid_anchor[3]] * 1
    context = 1 - mask
    global_context = np.zeros_like(depth)
    global_context[all_anchor[0]:all_anchor[1],all_anchor[2]:all_anchor[3]] = context
    # context = global_context[valid_anchor[0]:valid_anchor[1], valid_anchor[2]:valid_anchor[3]] * 1



    valid_area = mask + context
    input_rgb = image[valid_anchor[0]:valid_anchor[1], valid_anchor[2]:valid_anchor[3]] / 255. * context[..., None]
    input_depth = depth[valid_anchor[0]:valid_anchor[1], valid_anchor[2]:valid_anchor[3]] * context
    log_depth = np.log(input_depth + 1e-8)
    log_depth[mask > 0] = 0
    input_mean_depth = np.mean(log_depth[context > 0])
    input_zero_mean_depth = (log_depth - input_mean_depth) * context
    input_disp = 1./np.abs(input_depth)
    input_disp[mask > 0] = 0
    input_disp = input_disp / input_disp.max()
    valid_line = np.zeros_like(depth)
    if valid_line_anchor is not None:
        valid_line[valid_line_anchor[0]:valid_line_anchor[1], valid_line_anchor[2]:valid_line_anchor[3]] = 1
    valid_line = valid_line[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]]
    # f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True); ax1.imshow(global_context * 1 + global_mask * 2); ax2.imshow(image); plt.show()
    # f, ((ax1, ax2, ax3)) = plt.subplots(1, 3, sharex=True, sharey=True); ax1.imshow(context * 1 + mask * 2); ax2.imshow(input_rgb); ax3.imshow(valid_line); plt.show()
    # import pdb; pdb.set_trace()
    # return
    input_edge_map = edge_map[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]] * context
    input_other_edge_with_id = other_edge_with_id[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]]
    end_depth_maps = ((valid_line * input_edge_map) > 0) * input_depth


    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"

    valid_edge_ids = sorted(list(input_other_edge_with_id[(valid_line * input_edge_map) > 0]))
    valid_edge_ids = valid_edge_ids[1:] if (len(valid_edge_ids) > 0 and valid_edge_ids[0] == -1) else valid_edge_ids
    edge = reduce(lambda x, y: (x + (input_other_edge_with_id == y).astype(np.uint8)).clip(0, 1), [np.zeros_like(mask)] + list(valid_edge_ids))
    t_edge = torch.FloatTensor(edge).to(device)[None, None, ...]
    t_rgb = torch.FloatTensor(input_rgb).to(device).permute(2,0,1).unsqueeze(0)
    t_mask = torch.FloatTensor(mask).to(device)[None, None, ...]
    t_context = torch.FloatTensor(context).to(device)[None, None, ...]
    t_disp = torch.FloatTensor(input_disp).to(device)[None, None, ...]
    t_depth_zero_mean_depth = torch.FloatTensor(input_zero_mean_depth).to(device)[None, None, ...]

    depth_edge_output = depth_edge_model.forward_3P(t_mask, t_context, t_rgb, t_disp, t_edge, unit_length=128,
                                                    cuda=device)
    t_output_edge = (depth_edge_output> config['ext_edge_threshold']).float() * t_mask + t_edge
    output_raw_edge = t_output_edge.data.cpu().numpy().squeeze()
    # import pdb; pdb.set_trace()
    mesh = netx.Graph()
    hxs, hys = np.where(output_raw_edge * mask > 0)
    valid_map = mask + context
    for hx, hy in zip(hxs, hys):
        node = (hx, hy)
        mesh.add_node((hx, hy))
        eight_nes = [ne for ne in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1), \
                                   (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)]\
                        if 0 <= ne[0] < output_raw_edge.shape[0] and 0 <= ne[1] < output_raw_edge.shape[1] and 0 < output_raw_edge[ne[0], ne[1]]]
        for ne in eight_nes:
            mesh.add_edge(node, ne, length=np.hypot(ne[0] - hx, ne[1] - hy))
            if end_depth_maps[ne[0], ne[1]] != 0:
                mesh.nodes[ne[0], ne[1]]['cnt'] = True
                mesh.nodes[ne[0], ne[1]]['depth'] = end_depth_maps[ne[0], ne[1]]
    ccs = [*netx.connected_components(mesh)]
    end_pts = []
    for cc in ccs:
        end_pts.append(set())
        for node in cc:
            if mesh.nodes[node].get('cnt') is not None:
                end_pts[-1].add((node[0], node[1], mesh.nodes[node]['depth']))
    fpath_map = np.zeros_like(output_raw_edge) - 1
    npath_map = np.zeros_like(output_raw_edge) - 1
    for end_pt, cc in zip(end_pts, ccs):
        sorted_end_pt = []
        if len(end_pt) >= 2:
            continue
        if len(end_pt) == 0:
            continue
        if len(end_pt) == 1:
            sub_mesh = mesh.subgraph(list(cc)).copy()
            pnodes = netx.periphery(sub_mesh)
            ends = [*end_pt]
            edge_id = global_mesh.nodes[(ends[0][0] + all_anchor[0], ends[0][1] + all_anchor[2], -ends[0][2])]['edge_id']
            pnodes = sorted(pnodes,
                            key=lambda x: np.hypot((x[0] - ends[0][0]), (x[1] - ends[0][1])),
                            reverse=True)[0]
            npath = [*netx.shortest_path(sub_mesh, (ends[0][0], ends[0][1]), pnodes, weight='length')]
            for np_node in npath:
                npath_map[np_node[0], np_node[1]] = edge_id
            fpath = []
            if global_mesh.nodes[(ends[0][0] + all_anchor[0], ends[0][1] + all_anchor[2], -ends[0][2])].get('far') is None:
                print("None far")
                import pdb; pdb.set_trace()
            else:
                fnodes = global_mesh.nodes[(ends[0][0] + all_anchor[0], ends[0][1] + all_anchor[2], -ends[0][2])].get('far')
                fnodes = [(xx[0] - all_anchor[0], xx[1] - all_anchor[2], xx[2]) for xx in fnodes]
                dmask = mask + 0
                did = 0
                while True:
                    did += 1
                    dmask = cv2.dilate(dmask, np.ones((3, 3)), iterations=1)
                    if did > 3:
                        break
                    # ffnode = [fnode for fnode in fnodes if (dmask[fnode[0], fnode[1]] > 0)]
                    ffnode = [fnode for fnode in fnodes if (dmask[fnode[0], fnode[1]] > 0 and mask[fnode[0], fnode[1]] == 0)]
                    if len(ffnode) > 0:
                        fnode = ffnode[0]
                        break
                if len(ffnode) == 0:
                    continue
                fpath.append((fnode[0], fnode[1]))
                for step in range(0, len(npath) - 1):
                    parr = (npath[step + 1][0] - npath[step][0], npath[step + 1][1] - npath[step][1])
                    new_loc = (fpath[-1][0] + parr[0], fpath[-1][1] + parr[1])
                    new_loc_nes = [xx for xx in [(new_loc[0] + 1, new_loc[1]), (new_loc[0] - 1, new_loc[1]),
                                                (new_loc[0], new_loc[1] + 1), (new_loc[0], new_loc[1] - 1)]\
                                        if xx[0] >= 0 and xx[0] < fpath_map.shape[0] and xx[1] >= 0 and xx[1] < fpath_map.shape[1]]
                    if np.sum([fpath_map[nlne[0], nlne[1]] for nlne in new_loc_nes]) != -4:
                        break
                    if npath_map[new_loc[0], new_loc[1]] != -1:
                        if npath_map[new_loc[0], new_loc[1]] != edge_id:
                            break
                        else:
                            continue
                    if valid_area[new_loc[0], new_loc[1]] == 0:
                        break
                    new_loc_nes_eight = [xx for xx in [(new_loc[0] + 1, new_loc[1]), (new_loc[0] - 1, new_loc[1]),
                                                        (new_loc[0], new_loc[1] + 1), (new_loc[0], new_loc[1] - 1),
                                                        (new_loc[0] + 1, new_loc[1] + 1), (new_loc[0] + 1, new_loc[1] - 1),
                                                        (new_loc[0] - 1, new_loc[1] - 1), (new_loc[0] - 1, new_loc[1] + 1)]\
                                        if xx[0] >= 0 and xx[0] < fpath_map.shape[0] and xx[1] >= 0 and xx[1] < fpath_map.shape[1]]
                    if np.sum([int(npath_map[nlne[0], nlne[1]] == edge_id) for nlne in new_loc_nes_eight]) == 0:
                        break
                    fpath.append((fpath[-1][0] + parr[0], fpath[-1][1] + parr[1]))
                if step != len(npath) - 2:
                    for xx in npath[step+1:]:
                        if npath_map[xx[0], xx[1]] == edge_id:
                            npath_map[xx[0], xx[1]] = -1
            if len(fpath) > 0:
                for fp_node in fpath:
                    fpath_map[fp_node[0], fp_node[1]] = edge_id
    # import pdb; pdb.set_trace()
    far_edge = (fpath_map > -1).astype(np.uint8)
    update_edge = (npath_map > -1) * mask + edge
    t_update_edge = torch.FloatTensor(update_edge).to(device)[None, None, ...]
    depth_output = depth_feat_model.forward_3P(t_mask, t_context, t_depth_zero_mean_depth, t_update_edge, unit_length=128,
                                               cuda=device)
    depth_output = depth_output.cpu().data.numpy().squeeze()
    depth_output = np.exp(depth_output + input_mean_depth) * mask # + input_depth * context
    # if "right" in direc.lower() and "-" not in direc.lower():
    #     plt.imshow(depth_output); plt.show()
    #     import pdb; pdb.set_trace()
    #     f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True); ax1.imshow(depth_output); ax2.imshow(npath_map + fpath_map); plt.show()
    for near_id in np.unique(npath_map[npath_map > -1]):
        depth_output = refine_depth_around_edge(depth_output.copy(),
                                                (fpath_map == near_id).astype(np.uint8) * mask, # far_edge_map_in_mask,
                                                (fpath_map == near_id).astype(np.uint8), # far_edge_map,
                                                (npath_map == near_id).astype(np.uint8) * mask,
                                                mask.copy(),
                                                np.zeros_like(mask),
                                                config)
    # if "right" in direc.lower() and "-" not in direc.lower():
    #     plt.imshow(depth_output); plt.show()
    #     import pdb; pdb.set_trace()
    #     f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True); ax1.imshow(depth_output); ax2.imshow(npath_map + fpath_map); plt.show()
    rgb_output = rgb_feat_model.forward_3P(t_mask, t_context, t_rgb, t_update_edge, unit_length=128,
                                           cuda=device)

    # rgb_output = rgb_feat_model.forward_3P(t_mask, t_context, t_rgb, t_update_edge, unit_length=128, cuda=config['gpu_ids'])
    if config.get('gray_image') is True:
        rgb_output = rgb_output.mean(1, keepdim=True).repeat((1,3,1,1))
    rgb_output = ((rgb_output.squeeze().data.cpu().permute(1,2,0).numpy() * mask[..., None] + input_rgb) * 255).astype(np.uint8)
    image[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]][mask > 0] = rgb_output[mask > 0] # np.array([255,0,0]) # rgb_output[mask > 0]
    depth[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]][mask > 0] = depth_output[mask > 0]
    # nxs, nys = np.where(mask > -1)
    # for nx, ny in zip(nxs, nys):
    #     info_on_pix[(nx, ny)][0]['color'] = rgb_output[]


    nxs, nys = np.where((npath_map > -1))
    for nx, ny in zip(nxs, nys):
        n_id = npath_map[nx, ny]
        four_nes = [xx for xx in [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)]\
                        if 0 <= xx[0] < fpath_map.shape[0] and 0 <= xx[1] < fpath_map.shape[1]]
        for nex, ney in four_nes:
            if fpath_map[nex, ney] == n_id:
                na, nb = (nx + all_anchor[0], ny + all_anchor[2], info_on_pix[(nx + all_anchor[0], ny + all_anchor[2])][0]['depth']), \
                        (nex + all_anchor[0], ney + all_anchor[2], info_on_pix[(nex + all_anchor[0], ney + all_anchor[2])][0]['depth'])
                if global_mesh.has_edge(na, nb):
                    global_mesh.remove_edge(na, nb)
    nxs, nys = np.where((fpath_map > -1))
    for nx, ny in zip(nxs, nys):
        n_id = fpath_map[nx, ny]
        four_nes = [xx for xx in [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)]\
                        if 0 <= xx[0] < npath_map.shape[0] and 0 <= xx[1] < npath_map.shape[1]]
        for nex, ney in four_nes:
            if npath_map[nex, ney] == n_id:
                na, nb = (nx + all_anchor[0], ny + all_anchor[2], info_on_pix[(nx + all_anchor[0], ny + all_anchor[2])][0]['depth']), \
                        (nex + all_anchor[0], ney + all_anchor[2], info_on_pix[(nex + all_anchor[0], ney + all_anchor[2])][0]['depth'])
                if global_mesh.has_edge(na, nb):
                    global_mesh.remove_edge(na, nb)
    nxs, nys = np.where(mask > 0)
    for x, y in zip(nxs, nys):
        x = x + all_anchor[0]
        y = y + all_anchor[2]
        cur_node = (x, y, 0)
        new_node = (x, y, -abs(depth[x, y]))
        disp = 1. / -abs(depth[x, y])
        mapping_dict = {cur_node: new_node}
        info_on_pix, global_mesh = update_info(mapping_dict, info_on_pix, global_mesh)
        global_mesh.nodes[new_node]['color'] = image[x, y]
        global_mesh.nodes[new_node]['old_color'] = image[x, y]
        global_mesh.nodes[new_node]['disp'] = disp
        info_on_pix[(x, y)][0]['depth'] = -abs(depth[x, y])
        info_on_pix[(x, y)][0]['disp'] = disp
        info_on_pix[(x, y)][0]['color'] = image[x, y]


    nxs, nys = np.where((npath_map > -1))
    for nx, ny in zip(nxs, nys):
        self_node = (nx + all_anchor[0], ny + all_anchor[2], info_on_pix[(nx + all_anchor[0], ny + all_anchor[2])][0]['depth'])
        if global_mesh.has_node(self_node) is False:
            break
        n_id = int(round(npath_map[nx, ny]))
        four_nes = [xx for xx in [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)]\
                        if 0 <= xx[0] < fpath_map.shape[0] and 0 <= xx[1] < fpath_map.shape[1]]
        for nex, ney in four_nes:
            ne_node = (nex + all_anchor[0], ney + all_anchor[2], info_on_pix[(nex + all_anchor[0], ney + all_anchor[2])][0]['depth'])
            if global_mesh.has_node(ne_node) is False:
                continue
            if fpath_map[nex, ney] == n_id:
                if global_mesh.nodes[self_node].get('edge_id') is None:
                    global_mesh.nodes[self_node]['edge_id'] = n_id
                    edge_ccs[n_id].add(self_node)
                    info_on_pix[(self_node[0], self_node[1])][0]['edge_id'] = n_id
                if global_mesh.has_edge(self_node, ne_node) is True:
                    global_mesh.remove_edge(self_node, ne_node)
                if global_mesh.nodes[self_node].get('far') is None:
                    global_mesh.nodes[self_node]['far'] = []
                global_mesh.nodes[self_node]['far'].append(ne_node)

    global_fpath_map = np.zeros_like(other_edge_with_id) - 1
    global_fpath_map[all_anchor[0]:all_anchor[1], all_anchor[2]:all_anchor[3]] = fpath_map
    fpath_ids = np.unique(global_fpath_map)
    fpath_ids = fpath_ids[1:] if fpath_ids.shape[0] > 0 and fpath_ids[0] == -1 else []
    fpath_real_id_map = np.zeros_like(global_fpath_map) - 1
    for fpath_id in fpath_ids:
        fpath_real_id = np.unique(((global_fpath_map == fpath_id).astype(np.int) * (other_edge_with_id + 1)) - 1)
        fpath_real_id = fpath_real_id[1:] if fpath_real_id.shape[0] > 0 and fpath_real_id[0] == -1 else []
        fpath_real_id = fpath_real_id.astype(np.int)
        fpath_real_id = np.bincount(fpath_real_id).argmax()
        fpath_real_id_map[global_fpath_map == fpath_id] = fpath_real_id
    nxs, nys = np.where((fpath_map > -1))
    for nx, ny in zip(nxs, nys):
        self_node = (nx + all_anchor[0], ny + all_anchor[2], info_on_pix[(nx + all_anchor[0], ny + all_anchor[2])][0]['depth'])
        n_id = fpath_map[nx, ny]
        four_nes = [xx for xx in [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)]\
                        if 0 <= xx[0] < npath_map.shape[0] and 0 <= xx[1] < npath_map.shape[1]]
        for nex, ney in four_nes:
            ne_node = (nex + all_anchor[0], ney + all_anchor[2], info_on_pix[(nex + all_anchor[0], ney + all_anchor[2])][0]['depth'])
            if global_mesh.has_node(ne_node) is False:
                continue
            if npath_map[nex, ney] == n_id or global_mesh.nodes[ne_node].get('edge_id') == n_id:
                if global_mesh.has_edge(self_node, ne_node) is True:
                    global_mesh.remove_edge(self_node, ne_node)
                if global_mesh.nodes[self_node].get('near') is None:
                    global_mesh.nodes[self_node]['near'] = []
                if global_mesh.nodes[self_node].get('edge_id') is None:
                    f_id = int(round(fpath_real_id_map[self_node[0], self_node[1]]))
                    global_mesh.nodes[self_node]['edge_id'] = f_id
                    info_on_pix[(self_node[0], self_node[1])][0]['edge_id'] = f_id
                    edge_ccs[f_id].add(self_node)
                global_mesh.nodes[self_node]['near'].append(ne_node)

    return info_on_pix, global_mesh, image, depth, edge_ccs
    # for edge_cc in edge_ccs:
    #     for edge_node in edge_cc:
    #         edge_ccs
    # context_ccs, mask_ccs, broken_mask_ccs, edge_ccs, erode_context_ccs, init_mask_connect, edge_maps, extend_context_ccs, extend_edge_ccs

def get_valid_size(imap):
    x_max = np.where(imap.sum(1).squeeze() > 0)[0].max() + 1
    x_min = np.where(imap.sum(1).squeeze() > 0)[0].min()
    y_max = np.where(imap.sum(0).squeeze() > 0)[0].max() + 1
    y_min = np.where(imap.sum(0).squeeze() > 0)[0].min()
    size_dict = {'x_max':x_max, 'y_max':y_max, 'x_min':x_min, 'y_min':y_min}

    return size_dict

def dilate_valid_size(isize_dict, imap, dilate=[0, 0]):
    osize_dict = copy.deepcopy(isize_dict)
    osize_dict['x_min'] = max(0, osize_dict['x_min'] - dilate[0])
    osize_dict['x_max'] = min(imap.shape[0], osize_dict['x_max'] + dilate[0])
    osize_dict['y_min'] = max(0, osize_dict['y_min'] - dilate[0])
    osize_dict['y_max'] = min(imap.shape[1], osize_dict['y_max'] + dilate[1])

    return osize_dict

def size_operation(size_a, size_b, operation):
    assert operation == '+' or operation == '-', "Operation must be '+' (union) or '-' (exclude)"
    osize = {}
    if operation == '+':
        osize['x_min'] = min(size_a['x_min'], size_b['x_min'])
        osize['y_min'] = min(size_a['y_min'], size_b['y_min'])
        osize['x_max'] = max(size_a['x_max'], size_b['x_max'])
        osize['y_max'] = max(size_a['y_max'], size_b['y_max'])
    assert operation != '-', "Operation '-' is undefined !"

    return osize

def fill_dummy_bord(mesh, info_on_pix, image, depth, config):
    context = np.zeros_like(depth).astype(np.uint8)
    context[mesh.graph['hoffset']:mesh.graph['hoffset'] + mesh.graph['noext_H'],
            mesh.graph['woffset']:mesh.graph['woffset'] + mesh.graph['noext_W']] = 1
    mask = 1 - context
    xs, ys = np.where(mask > 0)
    depth = depth * context
    image = image * context[..., None]
    cur_depth = 0
    cur_disp = 0
    color = [0, 0, 0]
    for x, y in zip(xs, ys):
        cur_node = (x, y, cur_depth)
        mesh.add_node(cur_node, color=color,
                        synthesis=False,
                        disp=cur_disp,
                        cc_id=set(),
                        ext_pixel=True)
        info_on_pix[(x, y)] = [{'depth':cur_depth,
                    'color':mesh.nodes[(x, y, cur_depth)]['color'],
                    'synthesis':False,
                    'disp':mesh.nodes[cur_node]['disp'],
                    'ext_pixel':True}]
        # for x, y in zip(xs, ys):
        four_nes = [(xx, yy) for xx, yy in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)] if\
                    0 <= x < mesh.graph['H'] and 0 <= y < mesh.graph['W'] and info_on_pix.get((xx, yy)) is not None]
        for ne in four_nes:
            # if (ne[0] - x) + (ne[1] - y) == 1 and info_on_pix.get((ne[0], ne[1])) is not None:
            mesh.add_edge(cur_node, (ne[0], ne[1], info_on_pix[(ne[0], ne[1])][0]['depth']))

    return mesh, info_on_pix


def enlarge_border(mesh, info_on_pix, depth, image, config):
    mesh.graph['hoffset'], mesh.graph['woffset'] = config['extrapolation_thickness'], config['extrapolation_thickness']
    mesh.graph['bord_up'], mesh.graph['bord_left'], mesh.graph['bord_down'], mesh.graph['bord_right'] = \
        0, 0, mesh.graph['H'], mesh.graph['W']
    # new_image = np.pad(image,
    #                    pad_width=((config['extrapolation_thickness'], config['extrapolation_thickness']),
    #                               (config['extrapolation_thickness'], config['extrapolation_thickness']), (0, 0)),
    #                    mode='constant')
    # new_depth = np.pad(depth,
    #                    pad_width=((config['extrapolation_thickness'], config['extrapolation_thickness']),
    #                               (config['extrapolation_thickness'], config['extrapolation_thickness'])),
    #                    mode='constant')

    return mesh, info_on_pix, depth, image

def fill_missing_node(mesh, info_on_pix, image, depth):
    for x in range(mesh.graph['bord_up'], mesh.graph['bord_down']):
        for y in range(mesh.graph['bord_left'], mesh.graph['bord_right']):
            if info_on_pix.get((x, y)) is None:
                print("fill missing node = ", x, y)
                import pdb; pdb.set_trace()
                re_depth, re_count = 0, 0
                for ne in [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]:
                    if info_on_pix.get(ne) is not None:
                        re_depth += info_on_pix[ne][0]['depth']
                        re_count += 1
                if re_count == 0:
                    re_depth = -abs(depth[x, y])
                else:
                    re_depth = re_depth / re_count
                depth[x, y] = abs(re_depth)
                info_on_pix[(x, y)] = [{'depth':re_depth,
                                            'color':image[x, y],
                                            'synthesis':False,
                                            'disp':1./re_depth}]
                mesh.add_node((x, y, re_depth), color=image[x, y],
                                                synthesis=False,
                                                disp=1./re_depth,
                                                cc_id=set())
    return mesh, info_on_pix, depth



def refresh_bord_depth(mesh, info_on_pix, image, depth):
    H, W = mesh.graph['H'], mesh.graph['W']
    corner_nodes = [(mesh.graph['bord_up'], mesh.graph['bord_left']),
                    (mesh.graph['bord_up'], mesh.graph['bord_right'] - 1),
                    (mesh.graph['bord_down'] - 1, mesh.graph['bord_left']),
                    (mesh.graph['bord_down'] - 1, mesh.graph['bord_right'] - 1)]
                    # (0, W - 1), (H - 1, 0), (H - 1, W - 1)]
    bord_nodes = []
    bord_nodes += [(mesh.graph['bord_up'], xx) for xx in range(mesh.graph['bord_left'] + 1, mesh.graph['bord_right'] - 1)]
    bord_nodes += [(mesh.graph['bord_down'] - 1, xx) for xx in range(mesh.graph['bord_left'] + 1, mesh.graph['bord_right'] - 1)]
    bord_nodes += [(xx, mesh.graph['bord_left']) for xx in range(mesh.graph['bord_up'] + 1, mesh.graph['bord_down'] - 1)]
    bord_nodes += [(xx, mesh.graph['bord_right'] - 1) for xx in range(mesh.graph['bord_up'] + 1, mesh.graph['bord_down'] - 1)]
    for xy in bord_nodes:
        tgt_loc = None
        if xy[0] == mesh.graph['bord_up']:
            tgt_loc = (xy[0] + 1, xy[1])# (1, xy[1])
        elif xy[0] == mesh.graph['bord_down'] - 1:
            tgt_loc = (xy[0] - 1, xy[1]) # (H - 2, xy[1])
        elif xy[1] == mesh.graph['bord_left']:
            tgt_loc = (xy[0], xy[1] + 1)
        elif xy[1] == mesh.graph['bord_right'] - 1:
            tgt_loc = (xy[0], xy[1] - 1)
        if tgt_loc is not None:
            ne_infos = info_on_pix.get(tgt_loc)
            if ne_infos is None:
                import pdb; pdb.set_trace()
            # if ne_infos is not None and len(ne_infos) == 1:
            tgt_depth = ne_infos[0]['depth']
            tgt_disp = ne_infos[0]['disp']
            new_node = (xy[0], xy[1], tgt_depth)
            src_node = (tgt_loc[0], tgt_loc[1], tgt_depth)
            tgt_nes_loc = [(xx[0], xx[1]) \
                            for xx in mesh.neighbors(src_node)]
            tgt_nes_loc = [(xx[0] - tgt_loc[0] + xy[0], xx[1] - tgt_loc[1] + xy[1]) for xx in tgt_nes_loc \
                            if abs(xx[0] - xy[0]) == 1 and abs(xx[1] - xy[1]) == 1]
            tgt_nes_loc = [xx for xx in tgt_nes_loc if info_on_pix.get(xx) is not None]
            tgt_nes_loc.append(tgt_loc)
            # if (xy[0], xy[1]) == (559, 60):
            #     import pdb; pdb.set_trace()
            if info_on_pix.get(xy) is not None and len(info_on_pix.get(xy)) > 0:
                old_depth = info_on_pix[xy][0].get('depth')
                old_node = (xy[0], xy[1], old_depth)
                mesh.remove_edges_from([(old_ne, old_node) for old_ne in mesh.neighbors(old_node)])
                mesh.add_edges_from([((zz[0], zz[1], info_on_pix[zz][0]['depth']), old_node) for zz in tgt_nes_loc])
                mapping_dict = {old_node: new_node}
                # if old_node[2] == new_node[2]:
                #     print("mapping_dict = ", mapping_dict)
                info_on_pix, mesh = update_info(mapping_dict, info_on_pix, mesh)
            else:
                info_on_pix[xy] = []
                info_on_pix[xy][0] = info_on_pix[tgt_loc][0]
                info_on_pix['color'] = image[xy[0], xy[1]]
                info_on_pix['old_color'] = image[xy[0], xy[1]]
                mesh.add_node(new_node)
                mesh.add_edges_from([((zz[0], zz[1], info_on_pix[zz][0]['depth']), new_node) for zz in tgt_nes_loc])
            mesh.nodes[new_node]['far'] = None
            mesh.nodes[new_node]['near'] = None
            if mesh.nodes[src_node].get('far') is not None:
                redundant_nodes = [ne for ne in mesh.nodes[src_node]['far'] if (ne[0], ne[1]) == xy]
                [mesh.nodes[src_node]['far'].remove(aa) for aa in redundant_nodes]
            if mesh.nodes[src_node].get('near') is not None:
                redundant_nodes = [ne for ne in mesh.nodes[src_node]['near'] if (ne[0], ne[1]) == xy]
                [mesh.nodes[src_node]['near'].remove(aa) for aa in redundant_nodes]
    for xy in corner_nodes:
        hx, hy = xy
        four_nes = [xx for xx in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] if \
                        mesh.graph['bord_up'] <= xx[0] < mesh.graph['bord_down'] and \
                            mesh.graph['bord_left'] <= xx[1] < mesh.graph['bord_right']]
        ne_nodes = []
        ne_depths = []
        for ne_loc in four_nes:
            if info_on_pix.get(ne_loc) is not None:
                ne_depths.append(info_on_pix[ne_loc][0]['depth'])
                ne_nodes.append((ne_loc[0], ne_loc[1], info_on_pix[ne_loc][0]['depth']))
        new_node = (xy[0], xy[1], float(np.mean(ne_depths)))
        if info_on_pix.get(xy) is not None and len(info_on_pix.get(xy)) > 0:
            old_depth = info_on_pix[xy][0].get('depth')
            old_node = (xy[0], xy[1], old_depth)
            mesh.remove_edges_from([(old_ne, old_node) for old_ne in mesh.neighbors(old_node)])
            mesh.add_edges_from([(zz, old_node) for zz in ne_nodes])
            mapping_dict = {old_node: new_node}
            info_on_pix, mesh = update_info(mapping_dict, info_on_pix, mesh)
        else:
            info_on_pix[xy] = []
            info_on_pix[xy][0] = info_on_pix[ne_loc[-1]][0]
            info_on_pix['color'] = image[xy[0], xy[1]]
            info_on_pix['old_color'] = image[xy[0], xy[1]]
            mesh.add_node(new_node)
            mesh.add_edges_from([(zz, new_node) for zz in ne_nodes])
        mesh.nodes[new_node]['far'] = None
        mesh.nodes[new_node]['near'] = None
    for xy in bord_nodes + corner_nodes:
        # if (xy[0], xy[1]) == (559, 60):
        #     import pdb; pdb.set_trace()
        depth[xy[0], xy[1]] = abs(info_on_pix[xy][0]['depth'])
    for xy in bord_nodes:
        cur_node = (xy[0], xy[1], info_on_pix[xy][0]['depth'])
        nes = mesh.neighbors(cur_node)
        four_nes = set([(xy[0] + 1, xy[1]), (xy[0] - 1, xy[1]), (xy[0], xy[1] + 1), (xy[0], xy[1] - 1)]) - \
                   set([(ne[0], ne[1]) for ne in nes])
        four_nes = [ne for ne in four_nes if mesh.graph['bord_up'] <= ne[0] < mesh.graph['bord_down'] and \
                                             mesh.graph['bord_left'] <= ne[1] < mesh.graph['bord_right']]
        four_nes = [(ne[0], ne[1], info_on_pix[(ne[0], ne[1])][0]['depth']) for ne in four_nes]
        mesh.nodes[cur_node]['far'] = []
        mesh.nodes[cur_node]['near'] = []
        for ne in four_nes:
            if abs(ne[2]) >= abs(cur_node[2]):
                mesh.nodes[cur_node]['far'].append(ne)
            else:
                mesh.nodes[cur_node]['near'].append(ne)

    return mesh, info_on_pix, depth

def get_union_size(mesh, dilate, *alls_cc):
    all_cc = reduce(lambda x, y: x | y, [set()] + [*alls_cc])
    min_x, min_y, max_x, max_y = mesh.graph['H'], mesh.graph['W'], 0, 0
    H, W = mesh.graph['H'], mesh.graph['W']
    for node in all_cc:
        if node[0] < min_x:
            min_x = node[0]
        if node[0] > max_x:
            max_x = node[0]
        if node[1] < min_y:
            min_y = node[1]
        if node[1] > max_y:
            max_y = node[1]
    max_x = max_x + 1
    max_y = max_y + 1
    # mask_size = dilate_valid_size(mask_size, edge_dict['mask'], dilate=[20, 20])
    osize_dict = dict()
    osize_dict['x_min'] = max(0, min_x - dilate[0])
    osize_dict['x_max'] = min(H, max_x + dilate[0])
    osize_dict['y_min'] = max(0, min_y - dilate[1])
    osize_dict['y_max'] = min(W, max_y + dilate[1])

    return osize_dict

def incomplete_node(mesh, edge_maps, info_on_pix):
    vis_map = np.zeros((mesh.graph['H'], mesh.graph['W']))

    for node in mesh.nodes:
        if mesh.nodes[node].get('synthesis') is not True:
            connect_all_flag = False
            nes = [xx for xx in mesh.neighbors(node) if mesh.nodes[xx].get('synthesis') is not True]
            if len(nes) < 3 and 0 < node[0] < mesh.graph['H'] - 1 and 0 < node[1] < mesh.graph['W'] - 1:
                if len(nes) <= 1:
                    connect_all_flag = True
                else:
                    dan_ne_node_a = nes[0]
                    dan_ne_node_b = nes[1]
                    if abs(dan_ne_node_a[0] - dan_ne_node_b[0]) > 1 or \
                        abs(dan_ne_node_a[1] - dan_ne_node_b[1]) > 1:
                        connect_all_flag = True
            if connect_all_flag == True:
                vis_map[node[0], node[1]] = len(nes)
                four_nes = [(node[0] - 1, node[1]), (node[0] + 1, node[1]), (node[0], node[1] - 1), (node[0], node[1] + 1)]
                for ne in four_nes:
                    for info in info_on_pix[(ne[0], ne[1])]:
                        ne_node = (ne[0], ne[1], info['depth'])
                        if info.get('synthesis') is not True and mesh.has_node(ne_node):
                            mesh.add_edge(node, ne_node)
                            break

    return mesh

def edge_inpainting(edge_id, context_cc, erode_context_cc, mask_cc, edge_cc, extend_edge_cc,
                    mesh, edge_map, edge_maps_with_id, config, union_size, depth_edge_model, inpaint_iter):
    edge_dict = get_edge_from_nodes(context_cc, erode_context_cc, mask_cc, edge_cc, extend_edge_cc,
                                        mesh.graph['H'], mesh.graph['W'], mesh)
    edge_dict['edge'], end_depth_maps, _ = \
        filter_irrelevant_edge_new(edge_dict['self_edge'] + edge_dict['comp_edge'],
                                edge_map,
                                edge_maps_with_id,
                                edge_id,
                                edge_dict['context'],
                                edge_dict['depth'], mesh, context_cc | erode_context_cc, spdb=True)
    patch_edge_dict = dict()
    patch_edge_dict['mask'], patch_edge_dict['context'], patch_edge_dict['rgb'], \
        patch_edge_dict['disp'], patch_edge_dict['edge'] = \
        crop_maps_by_size(union_size, edge_dict['mask'], edge_dict['context'],
                            edge_dict['rgb'], edge_dict['disp'], edge_dict['edge'])
    tensor_edge_dict = convert2tensor(patch_edge_dict)
    if require_depth_edge(patch_edge_dict['edge'], patch_edge_dict['mask']) and inpaint_iter == 0:
        with torch.no_grad():
            device = config["gpu_ids"] if isinstance(config["gpu_ids"], int) and config["gpu_ids"] >= 0 else "cpu"
            depth_edge_output = depth_edge_model.forward_3P(tensor_edge_dict['mask'],
                                                            tensor_edge_dict['context'],
                                                            tensor_edge_dict['rgb'],
                                                            tensor_edge_dict['disp'],
                                                            tensor_edge_dict['edge'],
                                                            unit_length=128,
                                                            cuda=device)
            depth_edge_output = depth_edge_output.cpu()
        tensor_edge_dict['output'] = (depth_edge_output > config['ext_edge_threshold']).float() * tensor_edge_dict['mask'] + tensor_edge_dict['edge']
    else:
        tensor_edge_dict['output'] = tensor_edge_dict['edge']
        depth_edge_output = tensor_edge_dict['edge'] + 0
    patch_edge_dict['output'] = tensor_edge_dict['output'].squeeze().data.cpu().numpy()
    edge_dict['output'] = np.zeros((mesh.graph['H'], mesh.graph['W']))
    edge_dict['output'][union_size['x_min']:union_size['x_max'], union_size['y_min']:union_size['y_max']] = \
        patch_edge_dict['output']

    return edge_dict, end_depth_maps

def depth_inpainting(context_cc, extend_context_cc, erode_context_cc, mask_cc, mesh, config, union_size, depth_feat_model, edge_output, given_depth_dict=False, spdb=False):
    if given_depth_dict is False:
        depth_dict = get_depth_from_nodes(context_cc | extend_context_cc, erode_context_cc, mask_cc, mesh.graph['H'], mesh.graph['W'], mesh, config['log_depth'])
        if edge_output is not None:
            depth_dict['edge'] = edge_output
    else:
        depth_dict = given_depth_dict
    patch_depth_dict = dict()
    patch_depth_dict['mask'], patch_depth_dict['context'], patch_depth_dict['depth'], \
        patch_depth_dict['zero_mean_depth'], patch_depth_dict['edge'] = \
            crop_maps_by_size(union_size, depth_dict['mask'], depth_dict['context'],
                                depth_dict['real_depth'], depth_dict['zero_mean_depth'], depth_dict['edge'])
    tensor_depth_dict = convert2tensor(patch_depth_dict)
    resize_mask = open_small_mask(tensor_depth_dict['mask'], tensor_depth_dict['context'], 3, 41)
    with torch.no_grad():
        device = config["gpu_ids"] if isinstance(config["gpu_ids"], int) and config["gpu_ids"] >= 0 else "cpu"
        depth_output = depth_feat_model.forward_3P(resize_mask,
                                                    tensor_depth_dict['context'],
                                                    tensor_depth_dict['zero_mean_depth'],
                                                    tensor_depth_dict['edge'],
                                                    unit_length=128,
                                                    cuda=device)
        depth_output = depth_output.cpu()
    tensor_depth_dict['output'] = torch.exp(depth_output + depth_dict['mean_depth']) * \
                                            tensor_depth_dict['mask'] + tensor_depth_dict['depth']
    patch_depth_dict['output'] = tensor_depth_dict['output'].data.cpu().numpy().squeeze()
    depth_dict['output'] = np.zeros((mesh.graph['H'], mesh.graph['W']))
    depth_dict['output'][union_size['x_min']:union_size['x_max'], union_size['y_min']:union_size['y_max']] = \
        patch_depth_dict['output']
    depth_output = depth_dict['output'] * depth_dict['mask'] + depth_dict['depth'] * depth_dict['context']
    depth_output = smooth_cntsyn_gap(depth_dict['output'].copy() * depth_dict['mask'] + depth_dict['depth'] * depth_dict['context'],
                                    depth_dict['mask'], depth_dict['context'],
                                    init_mask_region=depth_dict['mask'])
    if spdb is True:
        f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True);
        ax1.imshow(depth_output * depth_dict['mask'] + depth_dict['depth']); ax2.imshow(depth_dict['output'] * depth_dict['mask'] + depth_dict['depth']); plt.show()
        import pdb; pdb.set_trace()
    depth_dict['output'] = depth_output * depth_dict['mask'] + depth_dict['depth'] * depth_dict['context']

    return depth_dict

def update_info(mapping_dict, info_on_pix, *meshes):
    rt_meshes = []
    for mesh in meshes:
        rt_meshes.append(relabel_node(mesh, mesh.nodes, [*mapping_dict.keys()][0], [*mapping_dict.values()][0]))
    x, y, _ = [*mapping_dict.keys()][0]
    info_on_pix[(x, y)][0]['depth'] = [*mapping_dict.values()][0][2]

    return [info_on_pix] + rt_meshes

def build_connection(mesh, cur_node, dst_node):
    if (abs(cur_node[0] - dst_node[0]) + abs(cur_node[1] - dst_node[1])) < 2:
        mesh.add_edge(cur_node, dst_node)
    if abs(cur_node[0] - dst_node[0]) > 1 or abs(cur_node[1] - dst_node[1]) > 1:
        return mesh
    ne_nodes = [*mesh.neighbors(cur_node)].copy()
    for ne_node in ne_nodes:
        if mesh.has_edge(ne_node, dst_node) or ne_node == dst_node:
            continue
        else:
            mesh = build_connection(mesh, ne_node, dst_node)

    return mesh

def recursive_add_edge(edge_mesh, mesh, info_on_pix, cur_node, mark):
    ne_nodes = [(x[0], x[1]) for x in edge_mesh.neighbors(cur_node)]
    for node_xy in ne_nodes:
        node = (node_xy[0], node_xy[1], info_on_pix[node_xy][0]['depth'])
        if mark[node[0], node[1]] != 3:
            continue
        else:
            mark[node[0], node[1]] = 0
            mesh.remove_edges_from([(xx, node) for xx in mesh.neighbors(node)])
            mesh = build_connection(mesh, cur_node, node)
            re_info = dict(depth=0, count=0)
            for re_ne in mesh.neighbors(node):
                re_info['depth'] += re_ne[2]
                re_info['count'] += 1.
            try:
                re_depth = re_info['depth'] / re_info['count']
            except:
                re_depth = node[2]
            re_node = (node_xy[0], node_xy[1], re_depth)
            mapping_dict = {node: re_node}
            info_on_pix, edge_mesh, mesh = update_info(mapping_dict, info_on_pix, edge_mesh, mesh)

            edge_mesh, mesh, mark, info_on_pix = recursive_add_edge(edge_mesh, mesh, info_on_pix, re_node, mark)

    return edge_mesh, mesh, mark, info_on_pix

def resize_for_edge(tensor_dict, largest_size):
    resize_dict = {k: v.clone() for k, v in tensor_dict.items()}
    frac = largest_size / np.array([*resize_dict['edge'].shape[-2:]]).max()
    if frac < 1:
        resize_mark = torch.nn.functional.interpolate(torch.cat((resize_dict['mask'],
                                                        resize_dict['context']),
                                                        dim=1),
                                                        scale_factor=frac,
                                                        mode='bilinear')
        resize_dict['mask'] = (resize_mark[:, 0:1] > 0).float()
        resize_dict['context'] = (resize_mark[:, 1:2] == 1).float()
        resize_dict['context'][resize_dict['mask'] > 0] = 0
        resize_dict['edge'] = torch.nn.functional.interpolate(resize_dict['edge'],
                                                                scale_factor=frac,
                                                                mode='bilinear')
        resize_dict['edge'] = (resize_dict['edge'] > 0).float()
        resize_dict['edge'] = resize_dict['edge'] * resize_dict['context']
        resize_dict['disp'] = torch.nn.functional.interpolate(resize_dict['disp'],
                                                                scale_factor=frac,
                                                                mode='nearest')
        resize_dict['disp'] = resize_dict['disp'] * resize_dict['context']
        resize_dict['rgb'] = torch.nn.functional.interpolate(resize_dict['rgb'],
                                                                    scale_factor=frac,
                                                                    mode='bilinear')
        resize_dict['rgb'] = resize_dict['rgb'] * resize_dict['context']
    return resize_dict

def get_map_from_nodes(nodes, height, width):
    omap = np.zeros((height, width))
    for n in nodes:
        omap[n[0], n[1]] = 1

    return omap

def get_map_from_ccs(ccs, height, width, condition_input=None, condition=None, real_id=False, id_shift=0):
    if condition is None:
        condition = lambda x, condition_input: True

    if real_id is True:
        omap = np.zeros((height, width)) + (-1) + id_shift
    else:
        omap = np.zeros((height, width))
    for cc_id, cc in enumerate(ccs):
        for n in cc:
            if condition(n, condition_input):
                if real_id is True:
                    omap[n[0], n[1]] = cc_id + id_shift
                else:
                    omap[n[0], n[1]] = 1
    return omap

def revise_map_by_nodes(nodes, imap, operation, limit_constr=None):
    assert operation == '+' or operation == '-', "Operation must be '+' (union) or '-' (exclude)"
    omap = copy.deepcopy(imap)
    revise_flag = True
    if operation == '+':
        for n in nodes:
            omap[n[0], n[1]] = 1
        if limit_constr is not None and omap.sum() > limit_constr:
            omap = imap
            revise_flag = False
    elif operation == '-':
        for n in nodes:
            omap[n[0], n[1]] = 0
        if limit_constr is not None and omap.sum() < limit_constr:
            omap = imap
            revise_flag = False

    return omap, revise_flag

def repaint_info(mesh, cc, x_anchor, y_anchor, source_type):
    if source_type == 'rgb':
        feat = np.zeros((3, x_anchor[1] - x_anchor[0], y_anchor[1] - y_anchor[0]))
    else:
        feat = np.zeros((1, x_anchor[1] - x_anchor[0], y_anchor[1] - y_anchor[0]))
    for node in cc:
        if source_type == 'rgb':
            feat[:, node[0] - x_anchor[0], node[1] - y_anchor[0]] = np.array(mesh.nodes[node]['color']) / 255.
        elif source_type == 'd':
            feat[:, node[0] - x_anchor[0], node[1] - y_anchor[0]] = abs(node[2])

    return feat

def get_context_from_nodes(mesh, cc, H, W, source_type=''):
    if 'rgb' in source_type or 'color' in source_type:
        feat = np.zeros((H, W, 3))
    else:
        feat = np.zeros((H, W))
    context = np.zeros((H, W))
    for node in cc:
        if 'rgb' in source_type or 'color' in source_type:
            feat[node[0], node[1]] = np.array(mesh.nodes[node]['color']) / 255.
            context[node[0], node[1]] = 1
        else:
            feat[node[0], node[1]] = abs(node[2])

    return feat, context

def get_mask_from_nodes(mesh, cc, H, W):
    mask = np.zeros((H, W))
    for node in cc:
        mask[node[0], node[1]] = abs(node[2])

    return mask


def get_edge_from_nodes(context_cc, erode_context_cc, mask_cc, edge_cc, extend_edge_cc, H, W, mesh):
    context = np.zeros((H, W))
    mask = np.zeros((H, W))
    rgb = np.zeros((H, W, 3))
    disp = np.zeros((H, W))
    depth = np.zeros((H, W))
    real_depth = np.zeros((H, W))
    edge = np.zeros((H, W))
    comp_edge = np.zeros((H, W))
    fpath_map = np.zeros((H, W)) - 1
    npath_map = np.zeros((H, W)) - 1
    near_depth = np.zeros((H, W))
    for node in context_cc:
        rgb[node[0], node[1]] = np.array(mesh.nodes[node]['color'])
        disp[node[0], node[1]] = mesh.nodes[node]['disp']
        depth[node[0], node[1]] = node[2]
        context[node[0], node[1]] = 1
    for node in erode_context_cc:
        rgb[node[0], node[1]] = np.array(mesh.nodes[node]['color'])
        disp[node[0], node[1]] = mesh.nodes[node]['disp']
        depth[node[0], node[1]] = node[2]
        context[node[0], node[1]] = 1
    rgb = rgb / 255.
    disp = np.abs(disp)
    disp = disp / disp.max()
    real_depth = depth.copy()
    for node in context_cc:
        if mesh.nodes[node].get('real_depth') is not None:
            real_depth[node[0], node[1]] = mesh.nodes[node]['real_depth']
    for node in erode_context_cc:
        if mesh.nodes[node].get('real_depth') is not None:
            real_depth[node[0], node[1]] = mesh.nodes[node]['real_depth']
    for node in mask_cc:
        mask[node[0], node[1]] = 1
        near_depth[node[0], node[1]] = node[2]
    for node in edge_cc:
        edge[node[0], node[1]] = 1
    for node in extend_edge_cc:
        comp_edge[node[0], node[1]] = 1
    rt_dict = {'rgb': rgb, 'disp': disp, 'depth': depth, 'real_depth': real_depth, 'self_edge': edge, 'context': context,
               'mask': mask, 'fpath_map': fpath_map, 'npath_map': npath_map, 'comp_edge': comp_edge, 'valid_area': context + mask,
               'near_depth': near_depth}

    return rt_dict

def get_depth_from_maps(context_map, mask_map, depth_map, H, W, log_depth=False):
    context = context_map.astype(np.uint8)
    mask = mask_map.astype(np.uint8).copy()
    depth = np.abs(depth_map)
    real_depth = depth.copy()
    zero_mean_depth = np.zeros((H, W))

    if log_depth is True:
        log_depth = np.log(real_depth + 1e-8) * context
        mean_depth = np.mean(log_depth[context > 0])
        zero_mean_depth = (log_depth - mean_depth) * context
    else:
        zero_mean_depth = real_depth
        mean_depth = 0
    edge = np.zeros_like(depth)

    rt_dict = {'depth': depth, 'real_depth': real_depth, 'context': context, 'mask': mask,
               'mean_depth': mean_depth, 'zero_mean_depth': zero_mean_depth, 'edge': edge}

    return rt_dict

def get_depth_from_nodes(context_cc, erode_context_cc, mask_cc, H, W, mesh, log_depth=False):
    context = np.zeros((H, W))
    mask = np.zeros((H, W))
    depth = np.zeros((H, W))
    real_depth = np.zeros((H, W))
    zero_mean_depth = np.zeros((H, W))
    for node in context_cc:
        depth[node[0], node[1]] = node[2]
        context[node[0], node[1]] = 1
    for node in erode_context_cc:
        depth[node[0], node[1]] = node[2]
        context[node[0], node[1]] = 1
    depth = np.abs(depth)
    real_depth = depth.copy()
    for node in context_cc:
        if mesh.nodes[node].get('real_depth') is not None:
            real_depth[node[0], node[1]] = mesh.nodes[node]['real_depth']
    for node in erode_context_cc:
        if mesh.nodes[node].get('real_depth') is not None:
            real_depth[node[0], node[1]] = mesh.nodes[node]['real_depth']
    real_depth = np.abs(real_depth)
    for node in mask_cc:
        mask[node[0], node[1]] = 1
    if log_depth is True:
        log_depth = np.log(real_depth + 1e-8) * context
        mean_depth = np.mean(log_depth[context > 0])
        zero_mean_depth = (log_depth - mean_depth) * context
    else:
        zero_mean_depth = real_depth
        mean_depth = 0

    rt_dict = {'depth': depth, 'real_depth': real_depth, 'context': context, 'mask': mask,
               'mean_depth': mean_depth, 'zero_mean_depth': zero_mean_depth}

    return rt_dict

def get_rgb_from_nodes(context_cc, erode_context_cc, mask_cc, H, W, mesh):
    context = np.zeros((H, W))
    mask = np.zeros((H, W))
    rgb = np.zeros((H, W, 3))
    erode_context = np.zeros((H, W))
    for node in context_cc:
        rgb[node[0], node[1]] = np.array(mesh.nodes[node]['color'])
        context[node[0], node[1]] = 1
    rgb = rgb / 255.
    for node in mask_cc:
        mask[node[0], node[1]] = 1
    for node in erode_context_cc:
        erode_context[node[0], node[1]] = 1
        mask[node[0], node[1]] = 1
    rt_dict = {'rgb': rgb, 'context': context, 'mask': mask,
               'erode': erode_context}

    return rt_dict

def crop_maps_by_size(size, *imaps):
    omaps = []
    for imap in imaps:
        omaps.append(imap[size['x_min']:size['x_max'], size['y_min']:size['y_max']].copy())

    return omaps

def convert2tensor(input_dict):
    rt_dict = {}
    for key, value in input_dict.items():
        if 'rgb' in key or 'color' in key:
            rt_dict[key] = torch.FloatTensor(value).permute(2, 0, 1)[None, ...]
        else:
            rt_dict[key] = torch.FloatTensor(value)[None, None, ...]

    return rt_dict
