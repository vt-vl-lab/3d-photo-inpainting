import os
import numpy as np
try:
    import cynetworkx as netx
except ImportError:
    import networkx as netx
import matplotlib.pyplot as plt
from functools import partial
from vispy import scene, io
from vispy.scene import visuals
from vispy.visuals.filters import Alpha
import cv2
from moviepy.editor import ImageSequenceClip
from skimage.transform import resize
import time
import copy
import torch
import os
from utils import path_planning, open_small_mask, clean_far_edge, refine_depth_around_edge
from utils import refine_color_around_edge, filter_irrelevant_edge_new, require_depth_edge, clean_far_edge_new
from utils import create_placeholder, refresh_node, find_largest_rect
from mesh_tools import get_depth_from_maps, get_map_from_ccs, get_edge_from_nodes, get_depth_from_nodes, get_rgb_from_nodes, crop_maps_by_size, convert2tensor, recursive_add_edge, update_info, filter_edge, relabel_node, depth_inpainting
from mesh_tools import refresh_bord_depth, enlarge_border, fill_dummy_bord, extrapolate, fill_missing_node, incomplete_node, get_valid_size, dilate_valid_size, size_operation
import transforms3d
import random
from functools import reduce

def create_mesh(depth, image, int_mtx, config):
    H, W, C = image.shape
    ext_H, ext_W = H + 2 * config['extrapolation_thickness'], W + 2 * config['extrapolation_thickness']
    LDI = netx.Graph(H=ext_H, W=ext_W, noext_H=H, noext_W=W, cam_param=int_mtx)
    xy2depth = {}
    int_mtx_pix = int_mtx * np.array([[W], [H], [1.]])
    LDI.graph['cam_param_pix'], LDI.graph['cam_param_pix_inv'] = int_mtx_pix, np.linalg.inv(int_mtx_pix)
    disp = 1. / (-depth)
    LDI.graph['hoffset'], LDI.graph['woffset'] = config['extrapolation_thickness'], config['extrapolation_thickness']
    LDI.graph['bord_up'], LDI.graph['bord_down'] = LDI.graph['hoffset'] + 0, LDI.graph['hoffset'] + H
    LDI.graph['bord_left'], LDI.graph['bord_right'] = LDI.graph['woffset'] + 0, LDI.graph['woffset'] + W
    for idx in range(H):
        for idy in range(W):
            x, y = idx + LDI.graph['hoffset'], idy + LDI.graph['woffset']
            LDI.add_node((x, y, -depth[idx, idy]),
                         color=image[idx, idy],
                         disp=disp[idx, idy],
                         synthesis=False,
                         cc_id=set())
            xy2depth[(x, y)] = [-depth[idx, idy]]
    for x, y, d in LDI.nodes:
        two_nes = [ne for ne in [(x+1, y), (x, y+1)] if ne[0] < LDI.graph['bord_down'] and ne[1] < LDI.graph['bord_right']]
        [LDI.add_edge((ne[0], ne[1], xy2depth[ne][0]), (x, y, d)) for ne in two_nes]
    LDI = calculate_fov(LDI)
    image = np.pad(image,
                    pad_width=((config['extrapolation_thickness'], config['extrapolation_thickness']),
                               (config['extrapolation_thickness'], config['extrapolation_thickness']),
                               (0, 0)),
                    mode='constant')
    depth = np.pad(depth,
                    pad_width=((config['extrapolation_thickness'], config['extrapolation_thickness']),
                               (config['extrapolation_thickness'], config['extrapolation_thickness'])),
                    mode='constant')

    return LDI, xy2depth, image, depth


def tear_edges(mesh, threshold = 0.00025, xy2depth=None):
    remove_edge_list = []
    remove_horizon, remove_vertical = np.zeros((2, mesh.graph['H'], mesh.graph['W']))
    mesh_nodes = mesh.nodes
    for edge in mesh.edges:
        if abs(mesh_nodes[edge[0]]['disp'] - mesh_nodes[edge[1]]['disp']) > threshold:
            remove_edge_list.append((edge[0], edge[1]))

            near, far = edge if abs(edge[0][2]) < abs(edge[1][2]) else edge[::-1]

            mesh_nodes[far]['near'] = [] if mesh_nodes[far].get('near') is None else mesh_nodes[far]['near'].append(near)
            mesh_nodes[near]['far'] = [] if mesh_nodes[near].get('far') is None else mesh_nodes[near]['far'].append(far)

            if near[0] == far[0]:
                remove_horizon[near[0], np.minimum(near[1], far[1])] = 1
            elif near[1] == far[1]:
                remove_vertical[np.minimum(near[0], far[0]), near[1]] = 1
    mesh.remove_edges_from(remove_edge_list)

    remove_edge_list = []

    dang_horizon = np.where(np.roll(remove_horizon, 1, 0) + np.roll(remove_horizon, -1, 0) - remove_horizon == 2)
    dang_vertical = np.where(np.roll(remove_vertical, 1, 1) + np.roll(remove_vertical, -1, 1) - remove_vertical == 2)

    horizon_condition = lambda x, y: mesh.graph['bord_up'] + 1 <= x < mesh.graph['bord_down'] - 1
    vertical_condition = lambda x, y: mesh.graph['bord_left'] + 1 <= y < mesh.graph['bord_right'] - 1

    prjto3d = lambda x, y: (x, y, xy2depth[(x, y)][0])

    node_existence = lambda x, y: mesh.has_node(prjto3d(x, y))

    for x, y in zip(dang_horizon[0], dang_horizon[1]):
        if horizon_condition(x, y) and node_existence(x, y) and node_existence(x, y+1):
            remove_edge_list.append((prjto3d(x, y), prjto3d(x, y+1)))
    for x, y in zip(dang_vertical[0], dang_vertical[1]):
        if vertical_condition(x, y) and node_existence(x, y) and node_existence(x+1, y):
            remove_edge_list.append((prjto3d(x, y), prjto3d(x+1, y)))
    mesh.remove_edges_from(remove_edge_list)

    return mesh

def calculate_fov(mesh):
    k = mesh.graph['cam_param']
    mesh.graph['hFov'] = 2 * np.arctan(1. / (2*k[0, 0]))
    mesh.graph['vFov'] = 2 * np.arctan(1. / (2*k[1, 1]))
    mesh.graph['aspect'] = mesh.graph['noext_H'] / mesh.graph['noext_W']

    return mesh

def calculate_fov_FB(mesh):
    mesh.graph['aspect'] = mesh.graph['H'] / mesh.graph['W']
    if mesh.graph['H'] > mesh.graph['W']:
        mesh.graph['hFov'] = 0.508015513
        half_short = np.tan(mesh.graph['hFov']/2.0)
        half_long = half_short * mesh.graph['aspect']
        mesh.graph['vFov'] = 2.0 * np.arctan(half_long)
    else:
        mesh.graph['vFov'] = 0.508015513
        half_short = np.tan(mesh.graph['vFov']/2.0)
        half_long = half_short / mesh.graph['aspect']
        mesh.graph['hFov'] = 2.0 * np.arctan(half_long)

    return mesh

def reproject_3d_int_detail(sx, sy, z, k_00, k_02, k_11, k_12, w_offset, h_offset):
    abs_z = abs(z)
    return [abs_z * ((sy+0.5-w_offset) * k_00 + k_02), abs_z * ((sx+0.5-h_offset) * k_11 + k_12), abs_z]

def reproject_3d_int_detail_FB(sx, sy, z, w_offset, h_offset, mesh):
    if mesh.graph.get('tan_hFov') is None:
        mesh.graph['tan_hFov'] = np.tan(mesh.graph['hFov'] / 2.)
    if mesh.graph.get('tan_vFov') is None:
        mesh.graph['tan_vFov'] = np.tan(mesh.graph['vFov'] / 2.)

    ray = np.array([(-1. + 2. * ((sy+0.5-w_offset)/(mesh.graph['W'] - 1))) * mesh.graph['tan_hFov'],
                    (1. - 2. * (sx+0.5-h_offset)/(mesh.graph['H'] - 1)) * mesh.graph['tan_vFov'],
                    -1])
    point_3d = ray * np.abs(z)

    return point_3d


def reproject_3d_int(sx, sy, z, mesh):
    k = mesh.graph['cam_param_pix_inv'].copy()
    if k[0, 2] > 0:
        k = np.linalg.inv(k)
    ray = np.dot(k, np.array([sy-mesh.graph['woffset'], sx-mesh.graph['hoffset'], 1]).reshape(3, 1))

    point_3d = ray * np.abs(z)
    point_3d = point_3d.flatten()

    return point_3d

def generate_init_node(mesh, config, min_node_in_cc):
    mesh_nodes = mesh.nodes

    info_on_pix = {}

    ccs = sorted(netx.connected_components(mesh), key = len, reverse=True)
    remove_nodes = []

    for cc in ccs:

        remove_flag = True if len(cc) < min_node_in_cc else False
        if remove_flag is False:
            for (nx, ny, nd) in cc:
                info_on_pix[(nx, ny)] = [{'depth':nd,
                                          'color':mesh_nodes[(nx, ny, nd)]['color'],
                                          'synthesis':False,
                                          'disp':mesh_nodes[(nx, ny, nd)]['disp']}]
        else:
            [remove_nodes.append((nx, ny, nd)) for (nx, ny, nd) in cc]

    for node in remove_nodes:
        far_nodes = [] if mesh_nodes[node].get('far') is None else mesh_nodes[node]['far']
        for far_node in far_nodes:
            if mesh.has_node(far_node) and mesh_nodes[far_node].get('near') is not None and node in mesh_nodes[far_node]['near']:
                mesh_nodes[far_node]['near'].remove(node)
        near_nodes = [] if mesh_nodes[node].get('near') is None else mesh_nodes[node]['near']
        for near_node in near_nodes:
            if mesh.has_node(near_node) and mesh_nodes[near_node].get('far') is not None and node in mesh_nodes[near_node]['far']:
                mesh_nodes[near_node]['far'].remove(node)

    [mesh.remove_node(node) for node in remove_nodes]

    return mesh, info_on_pix

def get_neighbors(mesh, node):
    return [*mesh.neighbors(node)]

def generate_face(mesh, info_on_pix, config):
    H, W = mesh.graph['H'], mesh.graph['W']
    str_faces = []
    num_node = len(mesh.nodes)
    ply_flag = config.get('save_ply')
    def out_fmt(input, cur_id_b, cur_id_self, cur_id_a, ply_flag):
        if ply_flag is True:
            input.append(' '.join(['3', cur_id_b, cur_id_self, cur_id_a]) + '\n')
        else:
            input.append([cur_id_b, cur_id_self, cur_id_a])
    mesh_nodes = mesh.nodes
    for node in mesh_nodes:
        cur_id_self = mesh_nodes[node]['cur_id']
        ne_nodes = get_neighbors(mesh, node)
        four_dir_nes = {'up': [], 'left': [],
                        'down': [], 'right': []}
        for ne_node in ne_nodes:
            store_tuple = [ne_node, mesh_nodes[ne_node]['cur_id']]
            if ne_node[0] == node[0]:
                if ne_node[1] == ne_node[1] - 1:
                    four_dir_nes['left'].append(store_tuple)
                else:
                    four_dir_nes['right'].append(store_tuple)
            else:
                if ne_node[0] == ne_node[0] - 1:
                    four_dir_nes['up'].append(store_tuple)
                else:
                    four_dir_nes['down'].append(store_tuple)
        for node_a, cur_id_a in four_dir_nes['up']:
            for node_b, cur_id_b in four_dir_nes['right']:
                out_fmt(str_faces, cur_id_b, cur_id_self, cur_id_a, ply_flag)
        for node_a, cur_id_a in four_dir_nes['right']:
            for node_b, cur_id_b in four_dir_nes['down']:
                out_fmt(str_faces, cur_id_b, cur_id_self, cur_id_a, ply_flag)
        for node_a, cur_id_a in four_dir_nes['down']:
            for node_b, cur_id_b in four_dir_nes['left']:
                out_fmt(str_faces, cur_id_b, cur_id_self, cur_id_a, ply_flag)
        for node_a, cur_id_a in four_dir_nes['left']:
            for node_b, cur_id_b in four_dir_nes['up']:
                out_fmt(str_faces, cur_id_b, cur_id_self, cur_id_a, ply_flag)

    return str_faces

def reassign_floating_island(mesh, info_on_pix, image, depth):
    H, W = mesh.graph['H'], mesh.graph['W'],
    mesh_nodes = mesh.nodes
    bord_up, bord_down = mesh.graph['bord_up'], mesh.graph['bord_down']
    bord_left, bord_right = mesh.graph['bord_left'], mesh.graph['bord_right']
    W = mesh.graph['W']
    lost_map = np.zeros((H, W))

    '''
    (5) is_inside(x, y, xmin, xmax, ymin, ymax) : Check if a pixel(x, y) is inside the border.
    (6) get_cross_nes(x, y) : Get the four cross neighbors of pixel(x, y).
    '''
    key_exist = lambda d, k: k in d
    is_inside = lambda x, y, xmin, xmax, ymin, ymax: xmin <= x < xmax and ymin <= y < ymax
    get_cross_nes = lambda x, y: [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]
    '''
    (A) Highlight the pixels on isolated floating island.
    (B) Number those isolated floating islands with connected component analysis.
    (C) For each isolated island:
        (1) Find its longest surrounded depth edge.
        (2) Propogate depth from that depth edge to the pixels on the isolated island.
        (3) Build the connection between the depth edge and that isolated island.
    '''
    for x in range(H):
        for y in range(W):
            if is_inside(x, y, bord_up, bord_down, bord_left, bord_right) and not(key_exist(info_on_pix, (x, y))):
                lost_map[x, y] = 1
    _, label_lost_map = cv2.connectedComponents(lost_map.astype(np.uint8), connectivity=4)
    mask = np.zeros((H, W))
    mask[bord_up:bord_down, bord_left:bord_right] = 1
    label_lost_map = (label_lost_map * mask).astype(np.int)

    for i in range(1, label_lost_map.max()+1):
        lost_xs, lost_ys = np.where(label_lost_map == i)
        surr_edge_ids = {}
        for lost_x, lost_y in zip(lost_xs, lost_ys):
            if (lost_x, lost_y) == (295, 389) or (lost_x, lost_y) == (296, 389):
                import pdb; pdb.set_trace()
            for ne in get_cross_nes(lost_x, lost_y):
                if key_exist(info_on_pix, ne):
                    for info in info_on_pix[ne]:
                        ne_node = (ne[0], ne[1], info['depth'])
                        if key_exist(mesh_nodes[ne_node], 'edge_id'):
                            edge_id = mesh_nodes[ne_node]['edge_id']
                            surr_edge_ids[edge_id] = surr_edge_ids[edge_id] + [ne_node] if \
                                                key_exist(surr_edge_ids, edge_id) else [ne_node]
        if len(surr_edge_ids) == 0:
            continue
        edge_id, edge_nodes = sorted([*surr_edge_ids.items()], key=lambda x: len(x[1]), reverse=True)[0]
        edge_depth_map = np.zeros((H, W))
        for node in edge_nodes:
            edge_depth_map[node[0], node[1]] = node[2]
        lost_xs, lost_ys = np.where(label_lost_map == i)
        while lost_xs.shape[0] > 0:
            lost_xs, lost_ys = np.where(label_lost_map == i)
            for lost_x, lost_y in zip(lost_xs, lost_ys):
                propagated_depth = []
                real_nes = []
                for ne in get_cross_nes(lost_x, lost_y):
                    if not(is_inside(ne[0], ne[1], bord_up, bord_down, bord_left, bord_right)) or \
                       edge_depth_map[ne[0], ne[1]] == 0:
                        continue
                    propagated_depth.append(edge_depth_map[ne[0], ne[1]])
                    real_nes.append(ne)
                if len(real_nes) == 0:
                    continue
                reassign_depth = np.mean(propagated_depth)
                label_lost_map[lost_x, lost_y] = 0
                edge_depth_map[lost_x, lost_y] = reassign_depth
                depth[lost_x, lost_y] = -reassign_depth
                mesh.add_node((lost_x, lost_y, reassign_depth), color=image[lost_x, lost_y],
                                                            synthesis=False,
                                                            disp=1./reassign_depth,
                                                            cc_id=set())
                info_on_pix[(lost_x, lost_y)] = [{'depth':reassign_depth,
                                                  'color':image[lost_x, lost_y],
                                                  'synthesis':False,
                                                  'disp':1./reassign_depth}]
                new_connections = [((lost_x, lost_y, reassign_depth),
                                    (ne[0], ne[1], edge_depth_map[ne[0], ne[1]])) for ne in real_nes]
                mesh.add_edges_from(new_connections)

    return mesh, info_on_pix, depth

def remove_node_feat(mesh, *feats):
    mesh_nodes = mesh.nodes
    for node in mesh_nodes:
        for feat in feats:
            mesh_nodes[node][feat] = None

    return mesh

def update_status(mesh, info_on_pix, depth=None):
    '''
    (2) clear_node_feat(G, *fts) : Clear all the node feature on graph G.
    (6) get_cross_nes(x, y) : Get the four cross neighbors of pixel(x, y).
    '''
    key_exist = lambda d, k: d.get(k) is not None
    is_inside = lambda x, y, xmin, xmax, ymin, ymax: xmin <= x < xmax and ymin <= y < ymax
    get_cross_nes = lambda x, y: [(x + 1, y), (x - 1, y), (x, y - 1), (x, y + 1)]
    append_element = lambda d, k, x: d[k] + [x] if key_exist(d, k) else [x]

    def clear_node_feat(G, fts):
        le_nodes = G.nodes
        for k in le_nodes:
            v = le_nodes[k]
            for ft in fts:
                if ft in v:
                    v[ft] = None

    clear_node_feat(mesh, ['edge_id', 'far', 'near'])
    bord_up, bord_down = mesh.graph['bord_up'], mesh.graph['bord_down']
    bord_left, bord_right = mesh.graph['bord_left'], mesh.graph['bord_right']

    le_nodes = mesh.nodes

    for node_key in le_nodes:
        if mesh.neighbors(node_key).__length_hint__() == 4:
            continue
        four_nes = [xx for xx in get_cross_nes(node_key[0], node_key[1]) if
                    is_inside(xx[0], xx[1], bord_up, bord_down, bord_left, bord_right) and
                    xx in info_on_pix]
        [four_nes.remove((ne_node[0], ne_node[1])) for ne_node in mesh.neighbors(node_key)]
        for ne in four_nes:
            for info in info_on_pix[ne]:
                assert mesh.has_node((ne[0], ne[1], info['depth'])), "No node_key"
                ind_node = le_nodes[node_key]
                if abs(node_key[2]) > abs(info['depth']):
                    ind_node['near'] = append_element(ind_node, 'near', (ne[0], ne[1], info['depth']))
                else:
                    ind_node['far'] = append_element(ind_node, 'far', (ne[0], ne[1], info['depth']))
    if depth is not None:
        for key, value in info_on_pix.items():
            if depth[key[0], key[1]] != abs(value[0]['depth']):
                value[0]['disp'] = 1. / value[0]['depth']
                depth[key[0], key[1]] = abs(value[0]['depth'])

        return mesh, depth, info_on_pix
    else:
        return mesh

def group_edges(LDI, config, image, remove_conflict_ordinal, spdb=False):

    '''
    (1) add_new_node(G, node) : add "node" to graph "G"
    (2) add_new_edge(G, node_a, node_b) : add edge "node_a--node_b" to graph "G"
    (3) exceed_thre(x, y, thre) : Check if difference between "x" and "y" exceed threshold "thre"
    (4) key_exist(d, k) : Check if key "k' exists in dictionary "d"
    (5) comm_opp_bg(G, x, y) : Check if node "x" and "y" in graph "G" treat the same opposite node as background
    (6) comm_opp_fg(G, x, y) : Check if node "x" and "y" in graph "G" treat the same opposite node as foreground
    '''
    add_new_node = lambda G, node: None if G.has_node(node) else G.add_node(node)
    add_new_edge = lambda G, node_a, node_b: None if G.has_edge(node_a, node_b) else G.add_edge(node_a, node_b)
    exceed_thre = lambda x, y, thre: (abs(x) - abs(y)) > thre
    key_exist = lambda d, k: d.get(k) is not None
    comm_opp_bg = lambda G, x, y: key_exist(G.nodes[x], 'far') and key_exist(G.nodes[y], 'far') and \
                                    not(set(G.nodes[x]['far']).isdisjoint(set(G.nodes[y]['far'])))
    comm_opp_fg = lambda G, x, y: key_exist(G.nodes[x], 'near') and key_exist(G.nodes[y], 'near') and \
                                    not(set(G.nodes[x]['near']).isdisjoint(set(G.nodes[y]['near'])))
    discont_graph = netx.Graph()
    '''
    (A) Skip the pixel at image boundary, we don't want to deal with them.
    (B) Identify discontinuity by the number of its neighbor(degree).
        If the degree < 4(up/right/buttom/left). We will go through following steps:
        (1) Add the discontinuity pixel "node" to graph "discont_graph".
        (2) Find "node"'s cross neighbor(up/right/buttom/left) "ne_node".
            - If the cross neighbor "ne_node" is a discontinuity pixel(degree("ne_node") < 4),
                (a) add it to graph "discont_graph" and build the connection between "ne_node" and "node".
                (b) label its cross neighbor as invalid pixels "inval_diag_candi" to avoid building
                    connection between original discontinuity pixel "node" and "inval_diag_candi".
            - Otherwise, find "ne_node"'s cross neighbors, called diagonal candidate "diag_candi".
                - The "diag_candi" is diagonal to the original discontinuity pixel "node".
                - If "diag_candi" exists, go to step(3).
        (3) A diagonal candidate "diag_candi" will be :
            - added to the "discont_graph" if its degree < 4.
            - connected to the original discontinuity pixel "node" if it satisfied either
                one of following criterion:
                (a) the difference of disparity between "diag_candi" and "node" is smaller than default threshold.
                (b) the "diag_candi" and "node" face the same opposite pixel. (See. function "tear_edges")
                (c) Both of "diag_candi" and "node" must_connect to each other. (See. function "combine_end_node")
    (C) Aggregate each connected part in "discont_graph" into "discont_ccs" (A.K.A. depth edge).
    '''
    for node in LDI.nodes:
        if not(LDI.graph['bord_up'] + 1 <= node[0] <= LDI.graph['bord_down'] - 2 and \
               LDI.graph['bord_left'] + 1 <= node[1] <= LDI.graph['bord_right'] - 2):
            continue
        neighbors = [*LDI.neighbors(node)]
        if len(neighbors) < 4:
            add_new_node(discont_graph, node)
            diag_candi_anc, inval_diag_candi, discont_nes = set(), set(), set()
            for ne_node in neighbors:
                if len([*LDI.neighbors(ne_node)]) < 4:
                    add_new_node(discont_graph, ne_node)
                    add_new_edge(discont_graph, ne_node, node)
                    discont_nes.add(ne_node)
                else:
                    diag_candi_anc.add(ne_node)
            inval_diag_candi = set([inval_diagonal for ne_node in discont_nes for inval_diagonal in LDI.neighbors(ne_node) if \
                                     abs(inval_diagonal[0] - node[0]) < 2 and abs(inval_diagonal[1] - node[1]) < 2])
            for ne_node in diag_candi_anc:
                if ne_node[0] == node[0]:
                    diagonal_xys = [[ne_node[0] + 1, ne_node[1]], [ne_node[0] - 1, ne_node[1]]]
                elif ne_node[1] == node[1]:
                    diagonal_xys = [[ne_node[0], ne_node[1] + 1], [ne_node[0], ne_node[1] - 1]]
                for diag_candi in LDI.neighbors(ne_node):
                    if [diag_candi[0], diag_candi[1]] in diagonal_xys and LDI.degree(diag_candi) < 4:
                        if diag_candi not in inval_diag_candi:
                            if not exceed_thre(1./node[2], 1./diag_candi[2], config['depth_threshold']) or \
                               (comm_opp_bg(LDI, diag_candi, node) and comm_opp_fg(LDI, diag_candi, node)):
                                add_new_node(discont_graph, diag_candi)
                                add_new_edge(discont_graph, diag_candi, node)
                        if key_exist(LDI.nodes[diag_candi], 'must_connect') and node in LDI.nodes[diag_candi]['must_connect'] and \
                            key_exist(LDI.nodes[node], 'must_connect') and diag_candi in LDI.nodes[node]['must_connect']:
                            add_new_node(discont_graph, diag_candi)
                            add_new_edge(discont_graph, diag_candi, node)
    if spdb == True:
        import pdb; pdb.set_trace()
    discont_ccs = [*netx.connected_components(discont_graph)]
    '''
    In some corner case, a depth edge "discont_cc" will contain both
    foreground(FG) and background(BG) pixels. This violate the assumption that
    a depth edge can only composite by one type of pixel(FG or BG).
    We need to further divide this depth edge into several sub-part so that the
    assumption is satisfied.
    (A) A depth edge is invalid if both of its "far_flag"(BG) and
        "near_flag"(FG) are True.
    (B) If the depth edge is invalid, we need to do:
        (1) Find the role("oridinal") of each pixel on the depth edge.
            "-1" --> Its opposite pixels has smaller depth(near) than it.
                     It is a backgorund pixel.
            "+1" --> Its opposite pixels has larger depth(far) than it.
                     It is a foregorund pixel.
            "0"  --> Some of opposite pixels has larger depth(far) than it,
                     and some has smaller pixel than it.
                     It is an ambiguous pixel.
        (2) For each pixel "discont_node", check if its neigbhors' roles are consistent.
            - If not, break the connection between the neighbor "ne_node" that has a role
              different from "discont_node".
            - If yes, remove all the role that are inconsistent to its neighbors "ne_node".
        (3) Connected component analysis to re-identified those divided depth edge.
    (C) Aggregate each connected part in "discont_graph" into "discont_ccs" (A.K.A. depth edge).
    '''
    if remove_conflict_ordinal:
        new_discont_ccs = []
        num_new_cc = 0
        for edge_id, discont_cc in enumerate(discont_ccs):
            near_flag = False
            far_flag = False
            for discont_node in discont_cc:
                near_flag = True if key_exist(LDI.nodes[discont_node], 'far') else near_flag
                far_flag = True if key_exist(LDI.nodes[discont_node], 'near') else far_flag
                if far_flag and near_flag:
                    break
            if far_flag and near_flag:
                for discont_node in discont_cc:
                    discont_graph.nodes[discont_node]['ordinal'] = \
                        np.array([key_exist(LDI.nodes[discont_node], 'far'),
                                  key_exist(LDI.nodes[discont_node], 'near')]) * \
                        np.array([-1, 1])
                    discont_graph.nodes[discont_node]['ordinal'] = \
                        np.sum(discont_graph.nodes[discont_node]['ordinal'])
                remove_nodes, remove_edges = [], []
                for discont_node in discont_cc:
                    ordinal_relation = np.sum([discont_graph.nodes[xx]['ordinal'] \
                                               for xx in discont_graph.neighbors(discont_node)])
                    near_side = discont_graph.nodes[discont_node]['ordinal'] <= 0
                    if abs(ordinal_relation) < len([*discont_graph.neighbors(discont_node)]):
                        remove_nodes.append(discont_node)
                        for ne_node in discont_graph.neighbors(discont_node):
                            remove_flag = (near_side and not(key_exist(LDI.nodes[ne_node], 'far'))) or \
                                          (not near_side and not(key_exist(LDI.nodes[ne_node], 'near')))
                            remove_edges += [(discont_node, ne_node)] if remove_flag else []
                    else:
                        if near_side and key_exist(LDI.nodes[discont_node], 'near'):
                            LDI.nodes[discont_node].pop('near')
                        elif not(near_side) and key_exist(LDI.nodes[discont_node], 'far'):
                            LDI.nodes[discont_node].pop('far')
                discont_graph.remove_edges_from(remove_edges)
                sub_mesh = discont_graph.subgraph(list(discont_cc)).copy()
                sub_discont_ccs = [*netx.connected_components(sub_mesh)]
                is_redun_near = lambda xx: len(xx) == 1 and xx[0] in remove_nodes and key_exist(LDI.nodes[xx[0]], 'far')
                for sub_discont_cc in sub_discont_ccs:
                    if is_redun_near(list(sub_discont_cc)):
                        LDI.nodes[list(sub_discont_cc)[0]].pop('far')
                    new_discont_ccs.append(sub_discont_cc)
            else:
                new_discont_ccs.append(discont_cc)
        discont_ccs = new_discont_ccs
        new_discont_ccs = None
    if spdb == True:
        import pdb; pdb.set_trace()

    for edge_id, edge_cc in enumerate(discont_ccs):
        for node in edge_cc:
            LDI.nodes[node]['edge_id'] = edge_id

    return discont_ccs, LDI, discont_graph

def combine_end_node(mesh, edge_mesh, edge_ccs, depth):
    import collections
    mesh_nodes = mesh.nodes
    connect_dict = dict()
    for valid_edge_id, valid_edge_cc in enumerate(edge_ccs):
        connect_info = []
        for valid_edge_node in valid_edge_cc:
            single_connect = set()
            for ne_node in mesh.neighbors(valid_edge_node):
                if mesh_nodes[ne_node].get('far') is not None:
                    for fn in mesh_nodes[ne_node].get('far'):
                        if mesh.has_node(fn) and mesh_nodes[fn].get('edge_id') is not None:
                            single_connect.add(mesh_nodes[fn]['edge_id'])
                if mesh_nodes[ne_node].get('near') is not None:
                    for fn in mesh_nodes[ne_node].get('near'):
                        if mesh.has_node(fn) and mesh_nodes[fn].get('edge_id') is not None:
                            single_connect.add(mesh_nodes[fn]['edge_id'])
            connect_info.extend([*single_connect])
        connect_dict[valid_edge_id] = collections.Counter(connect_info)

    end_maps = np.zeros((mesh.graph['H'], mesh.graph['W']))
    edge_maps = np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1
    for valid_edge_id, valid_edge_cc in enumerate(edge_ccs):
        for valid_edge_node in valid_edge_cc:
            edge_maps[valid_edge_node[0], valid_edge_node[1]] = valid_edge_id
            if len([*edge_mesh.neighbors(valid_edge_node)]) == 1:
                num_ne = 1
                if num_ne == 1:
                    end_maps[valid_edge_node[0], valid_edge_node[1]] = valid_edge_node[2]
    nxs, nys = np.where(end_maps != 0)
    invalid_nodes = set()
    for nx, ny in zip(nxs, nys):
        if mesh.has_node((nx, ny, end_maps[nx, ny])) is False:
            invalid_nodes.add((nx, ny))
            continue
        four_nes = [xx for xx in [(nx - 1, ny), (nx + 1, ny), (nx, ny - 1), (nx, ny + 1)] \
                        if 0 <= xx[0] < mesh.graph['H'] and 0 <= xx[1] < mesh.graph['W'] and \
                        end_maps[xx[0], xx[1]] != 0]
        mesh_nes = [*mesh.neighbors((nx, ny, end_maps[nx, ny]))]
        remove_num = 0
        for fne in four_nes:
            if (fne[0], fne[1], end_maps[fne[0], fne[1]]) in mesh_nes:
                remove_num += 1
        if remove_num == len(four_nes):
            invalid_nodes.add((nx, ny))
    for invalid_node in invalid_nodes:
        end_maps[invalid_node[0], invalid_node[1]] = 0

    nxs, nys = np.where(end_maps != 0)
    invalid_nodes = set()
    for nx, ny in zip(nxs, nys):
        if mesh_nodes[(nx, ny, end_maps[nx, ny])].get('edge_id') is None:
            continue
        else:
            self_id = mesh_nodes[(nx, ny, end_maps[nx, ny])].get('edge_id')
            self_connect = connect_dict[self_id] if connect_dict.get(self_id) is not None else dict()
        four_nes = [xx for xx in [(nx - 1, ny), (nx + 1, ny), (nx, ny - 1), (nx, ny + 1)] \
                        if 0 <= xx[0] < mesh.graph['H'] and 0 <= xx[1] < mesh.graph['W'] and \
                        end_maps[xx[0], xx[1]] != 0]
        for fne in four_nes:
            if mesh_nodes[(fne[0], fne[1], end_maps[fne[0], fne[1]])].get('edge_id') is None:
                continue
            else:
                ne_id = mesh_nodes[(fne[0], fne[1], end_maps[fne[0], fne[1]])]['edge_id']
                if self_connect.get(ne_id) is None or self_connect.get(ne_id) == 1:
                    continue
                else:
                    invalid_nodes.add((nx, ny))
    for invalid_node in invalid_nodes:
        end_maps[invalid_node[0], invalid_node[1]] = 0
    nxs, nys = np.where(end_maps != 0)
    invalid_nodes = set()
    for nx, ny in zip(nxs, nys):
        four_nes = [xx for xx in [(nx - 1, ny), (nx + 1, ny), (nx, ny - 1), (nx, ny + 1)] \
                        if 0 <= xx[0] < mesh.graph['H'] and 0 <= xx[1] < mesh.graph['W'] and \
                        end_maps[xx[0], xx[1]] != 0]
        for fne in four_nes:
            if mesh.has_node((fne[0], fne[1], end_maps[fne[0], fne[1]])):
                node_a, node_b = (fne[0], fne[1], end_maps[fne[0], fne[1]]), (nx, ny, end_maps[nx, ny])
                mesh.add_edge(node_a, node_b)
                mesh_nodes[node_b]['must_connect'] = set() if mesh_nodes[node_b].get('must_connect') is None else mesh_nodes[node_b]['must_connect']
                mesh_nodes[node_b]['must_connect'].add(node_a)
                mesh_nodes[node_b]['must_connect'] |= set([xx for xx in [*edge_mesh.neighbors(node_a)] if \
                                                            (xx[0] - node_b[0]) < 2 and (xx[1] - node_b[1]) < 2])
                mesh_nodes[node_a]['must_connect'] = set() if mesh_nodes[node_a].get('must_connect') is None else mesh_nodes[node_a]['must_connect']
                mesh_nodes[node_a]['must_connect'].add(node_b)
                mesh_nodes[node_a]['must_connect'] |= set([xx for xx in [*edge_mesh.neighbors(node_b)] if \
                                                            (xx[0] - node_a[0]) < 2 and (xx[1] - node_a[1]) < 2])
                invalid_nodes.add((nx, ny))
    for invalid_node in invalid_nodes:
        end_maps[invalid_node[0], invalid_node[1]] = 0

    return mesh

def remove_redundant_edge(mesh, edge_mesh, edge_ccs, info_on_pix, config, redundant_number=1000, invalid=False, spdb=False):
    point_to_amount = {}
    point_to_id = {}
    end_maps = np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1
    for valid_edge_id, valid_edge_cc in enumerate(edge_ccs):
        for valid_edge_node in valid_edge_cc:
            point_to_amount[valid_edge_node] = len(valid_edge_cc)
            point_to_id[valid_edge_node] = valid_edge_id
            if edge_mesh.has_node(valid_edge_node) is True:
                if len([*edge_mesh.neighbors(valid_edge_node)]) == 1:
                    end_maps[valid_edge_node[0], valid_edge_node[1]] = valid_edge_id
    nxs, nys = np.where(end_maps > -1)
    point_to_adjoint = {}
    for nx, ny in zip(nxs, nys):
        adjoint_edges = set([end_maps[x, y] for x, y in [(nx + 1, ny), (nx - 1, ny), (nx, ny + 1), (nx, ny - 1)] if end_maps[x, y] != -1])
        point_to_adjoint[end_maps[nx, ny]] = (point_to_adjoint[end_maps[nx, ny]] | adjoint_edges) if point_to_adjoint.get(end_maps[nx, ny]) is not None else adjoint_edges
    valid_edge_ccs = filter_edge(mesh, edge_ccs, config, invalid=invalid)
    edge_canvas = np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1
    for valid_edge_id, valid_edge_cc in enumerate(valid_edge_ccs):
        for valid_edge_node in valid_edge_cc:
            edge_canvas[valid_edge_node[0], valid_edge_node[1]] = valid_edge_id
    if spdb is True:
        plt.imshow(edge_canvas); plt.show()
        import pdb; pdb.set_trace()
    for valid_edge_id, valid_edge_cc in enumerate(valid_edge_ccs):
        end_number = 0
        four_end_number = 0
        eight_end_number = 0
        db_eight_end_number = 0
        if len(valid_edge_cc) > redundant_number:
            continue
        for valid_edge_node in valid_edge_cc:
            if len([*edge_mesh.neighbors(valid_edge_node)]) == 3:
                break
            elif len([*edge_mesh.neighbors(valid_edge_node)]) == 1:
                hx, hy, hz = valid_edge_node
                if invalid is False:
                    eight_nes = [(x, y) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1),
                                                     (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                                            if info_on_pix.get((x, y)) is not None and edge_canvas[x, y] != -1 and edge_canvas[x, y] != valid_edge_id]
                    if len(eight_nes) == 0:
                        end_number += 1
                if invalid is True:
                    four_nes = []; eight_nes = []; db_eight_nes = []
                    four_nes = [(x, y) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] \
                                            if info_on_pix.get((x, y)) is not None and edge_canvas[x, y] != -1 and edge_canvas[x, y] != valid_edge_id]
                    eight_nes = [(x, y) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1), \
                                                    (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                                            if info_on_pix.get((x, y)) is not None and edge_canvas[x, y] != -1 and edge_canvas[x, y] != valid_edge_id]
                    db_eight_nes = [(x, y) for x in range(hx - 2, hx + 3) for y in range(hy - 2, hy + 3) \
                                    if info_on_pix.get((x, y)) is not None and edge_canvas[x, y] != -1 and edge_canvas[x, y] != valid_edge_id and (x, y) != (hx, hy)]
                    if len(four_nes) == 0 or len(eight_nes) == 0:
                        end_number += 1
                        if len(four_nes) == 0:
                            four_end_number += 1
                        if len(eight_nes) == 0:
                            eight_end_number += 1
                        if len(db_eight_nes) == 0:
                            db_eight_end_number += 1
            elif len([*edge_mesh.neighbors(valid_edge_node)]) == 0:
                hx, hy, hz = valid_edge_node
                four_nes = [(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] \
                                if info_on_pix.get((x, y)) is not None and \
                                    mesh.has_edge(valid_edge_node, (x, y, info_on_pix[(x, y)][0]['depth'])) is False]
                for ne in four_nes:
                    try:
                        if invalid is True or (point_to_amount.get(ne) is None or point_to_amount[ne] < redundant_number) or \
                            point_to_id[ne] in point_to_adjoint.get(point_to_id[valid_edge_node], set()):
                            mesh.add_edge(valid_edge_node, ne)
                    except:
                        import pdb; pdb.set_trace()
        if (invalid is not True and end_number >= 1) or (invalid is True and end_number >= 2 and eight_end_number >= 1 and db_eight_end_number >= 1):
            for valid_edge_node in valid_edge_cc:
                hx, hy, _ = valid_edge_node
                four_nes = [(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] \
                                if info_on_pix.get((x, y)) is not None and \
                                    mesh.has_edge(valid_edge_node, (x, y, info_on_pix[(x, y)][0]['depth'])) is False and \
                                    (edge_canvas[x, y] == -1 or edge_canvas[x, y] == valid_edge_id)]
                for ne in four_nes:
                    if invalid is True or (point_to_amount.get(ne) is None or point_to_amount[ne] < redundant_number) or \
                        point_to_id[ne] in point_to_adjoint.get(point_to_id[valid_edge_node], set()):
                        mesh.add_edge(valid_edge_node, ne)

    return mesh

def judge_dangle(mark, mesh, node):
    if not (1 <= node[0] < mesh.graph['H']-1) or not(1 <= node[1] < mesh.graph['W']-1):
        return mark
    mesh_neighbors = [*mesh.neighbors(node)]
    mesh_neighbors = [xx for xx in mesh_neighbors if 0 < xx[0] < mesh.graph['H'] - 1 and 0 < xx[1] < mesh.graph['W'] - 1]
    if len(mesh_neighbors) >= 3:
        return mark
    elif len(mesh_neighbors) <= 1:
        mark[node[0], node[1]] = (len(mesh_neighbors) + 1)
    else:
        dan_ne_node_a = mesh_neighbors[0]
        dan_ne_node_b = mesh_neighbors[1]
        if abs(dan_ne_node_a[0] - dan_ne_node_b[0]) > 1 or \
            abs(dan_ne_node_a[1] - dan_ne_node_b[1]) > 1:
            mark[node[0], node[1]] = 3

    return mark

def remove_dangling(mesh, edge_ccs, edge_mesh, info_on_pix, image, depth, config):

    tmp_edge_ccs = copy.deepcopy(edge_ccs)
    for edge_cc_id, valid_edge_cc in enumerate(tmp_edge_ccs):
        if len(valid_edge_cc) > 1 or len(valid_edge_cc) == 0:
            continue
        single_edge_node = [*valid_edge_cc][0]
        hx, hy, hz = single_edge_node
        eight_nes = set([(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1),
                         (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                         if info_on_pix.get((x, y)) is not None])
        four_nes = [(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] \
                    if info_on_pix.get((x, y)) is not None]
        sub_mesh = mesh.subgraph(eight_nes).copy()
        ccs = netx.connected_components(sub_mesh)
        four_ccs = []
        for cc_id, _cc in enumerate(ccs):
            four_ccs.append(set())
            for cc_node in _cc:
                if abs(cc_node[0] - hx) + abs(cc_node[1] - hy) < 2:
                    four_ccs[cc_id].add(cc_node)
        largest_cc = sorted(four_ccs, key=lambda x: (len(x), -np.sum([abs(xx[2] - hz) for xx in x])))[-1]
        if len(largest_cc) < 2:
            for ne in four_nes:
                mesh.add_edge(single_edge_node, ne)
        else:
            mesh.remove_edges_from([(single_edge_node, ne) for ne in mesh.neighbors(single_edge_node)])
            new_depth = np.mean([xx[2] for xx in largest_cc])
            info_on_pix[(hx, hy)][0]['depth'] = new_depth
            info_on_pix[(hx, hy)][0]['disp'] = 1./new_depth
            new_node = (hx, hy, new_depth)
            mesh = refresh_node(single_edge_node, mesh.node[single_edge_node], new_node, dict(), mesh)
            edge_ccs[edge_cc_id] = set([new_node])
            for ne in largest_cc:
                mesh.add_edge(new_node, ne)

    mark = np.zeros((mesh.graph['H'], mesh.graph['W']))
    for edge_idx, edge_cc in enumerate(edge_ccs):
        for edge_node in edge_cc:
            if not (mesh.graph['bord_up'] <= edge_node[0] < mesh.graph['bord_down']-1) or \
               not (mesh.graph['bord_left'] <= edge_node[1] < mesh.graph['bord_right']-1):
                continue
            mesh_neighbors = [*mesh.neighbors(edge_node)]
            mesh_neighbors = [xx for xx in mesh_neighbors \
                                if mesh.graph['bord_up'] < xx[0] < mesh.graph['bord_down'] - 1 and \
                                   mesh.graph['bord_left'] < xx[1] < mesh.graph['bord_right'] - 1]
            if len([*mesh.neighbors(edge_node)]) >= 3:
                continue
            elif len([*mesh.neighbors(edge_node)]) <= 1:
                mark[edge_node[0], edge_node[1]] += (len([*mesh.neighbors(edge_node)]) + 1)
            else:
                dan_ne_node_a = [*mesh.neighbors(edge_node)][0]
                dan_ne_node_b = [*mesh.neighbors(edge_node)][1]
                if abs(dan_ne_node_a[0] - dan_ne_node_b[0]) > 1 or \
                    abs(dan_ne_node_a[1] - dan_ne_node_b[1]) > 1:
                    mark[edge_node[0], edge_node[1]] += 3
    mxs, mys = np.where(mark == 1)
    conn_0_nodes = [(x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth']) for x in zip(mxs, mys) \
                        if mesh.has_node((x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth']))]
    mxs, mys = np.where(mark == 2)
    conn_1_nodes = [(x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth']) for x in zip(mxs, mys) \
                        if mesh.has_node((x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth']))]
    for node in conn_0_nodes:
        hx, hy = node[0], node[1]
        four_nes = [(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1)] \
                     if info_on_pix.get((x, y)) is not None]
        re_depth = {'value' : 0, 'count': 0}
        for ne in four_nes:
            mesh.add_edge(node, ne)
            re_depth['value'] += cc_node[2]
            re_depth['count'] += 1.
        re_depth = re_depth['value'] / re_depth['count']
        mapping_dict = {node: (node[0], node[1], re_depth)}
        info_on_pix, mesh, edge_mesh = update_info(mapping_dict, info_on_pix, mesh, edge_mesh)
        depth[node[0], node[1]] = abs(re_depth)
        mark[node[0], node[1]] = 0
    for node in conn_1_nodes:
        hx, hy = node[0], node[1]
        eight_nes = set([(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1),
                                                                           (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                        if info_on_pix.get((x, y)) is not None])
        self_nes = set([ne2 for ne1 in mesh.neighbors(node) for ne2 in mesh.neighbors(ne1) if ne2 in eight_nes])
        eight_nes = [*(eight_nes - self_nes)]
        sub_mesh = mesh.subgraph(eight_nes).copy()
        ccs = netx.connected_components(sub_mesh)
        largest_cc = sorted(ccs, key=lambda x: (len(x), -np.sum([abs(xx[0] - node[0]) + abs(xx[1] - node[1]) for xx in x])))[-1]

        mesh.remove_edges_from([(xx, node) for xx in mesh.neighbors(node)])
        re_depth = {'value' : 0, 'count': 0}
        for cc_node in largest_cc:
            if cc_node[0] == node[0] and cc_node[1] == node[1]:
                continue
            re_depth['value'] += cc_node[2]
            re_depth['count'] += 1.
            if abs(cc_node[0] - node[0]) + abs(cc_node[1] - node[1]) < 2:
                mesh.add_edge(cc_node, node)
        try:
            re_depth = re_depth['value'] / re_depth['count']
        except:
            re_depth = node[2]
        renode = (node[0], node[1], re_depth)
        mapping_dict = {node: renode}
        info_on_pix, mesh, edge_mesh = update_info(mapping_dict, info_on_pix, mesh, edge_mesh)
        depth[node[0], node[1]] = abs(re_depth)
        mark[node[0], node[1]] = 0
        edge_mesh, mesh, mark, info_on_pix = recursive_add_edge(edge_mesh, mesh, info_on_pix, renode, mark)
    mxs, mys = np.where(mark == 3)
    conn_2_nodes = [(x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth']) for x in zip(mxs, mys) \
                        if mesh.has_node((x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth'])) and \
                            mesh.degree((x[0], x[1], info_on_pix[(x[0], x[1])][0]['depth'])) == 2]
    sub_mesh = mesh.subgraph(conn_2_nodes).copy()
    ccs = netx.connected_components(sub_mesh)
    for cc in ccs:
        candidate_nodes = [xx for xx in cc if sub_mesh.degree(xx) == 1]
        for node in candidate_nodes:
            if mesh.has_node(node) is False:
                continue
            ne_node = [xx for xx in mesh.neighbors(node) if xx not in cc][0]
            hx, hy = node[0], node[1]
            eight_nes = set([(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1),
                                                                            (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                              if info_on_pix.get((x, y)) is not None and (x, y, info_on_pix[(x, y)][0]['depth']) not in cc])
            ne_sub_mesh = mesh.subgraph(eight_nes).copy()
            ne_ccs = netx.connected_components(ne_sub_mesh)
            try:
                ne_cc = [ne_cc for ne_cc in ne_ccs if ne_node in ne_cc][0]
            except:
                import pdb; pdb.set_trace()
            largest_cc = [xx for xx in ne_cc if abs(xx[0] - node[0]) + abs(xx[1] - node[1]) == 1]
            mesh.remove_edges_from([(xx, node) for xx in mesh.neighbors(node)])
            re_depth = {'value' : 0, 'count': 0}
            for cc_node in largest_cc:
                re_depth['value'] += cc_node[2]
                re_depth['count'] += 1.
                mesh.add_edge(cc_node, node)
            try:
                re_depth = re_depth['value'] / re_depth['count']
            except:
                re_depth = node[2]
            renode = (node[0], node[1], re_depth)
            mapping_dict = {node: renode}
            info_on_pix, mesh, edge_mesh = update_info(mapping_dict, info_on_pix, mesh, edge_mesh)
            depth[node[0], node[1]] = abs(re_depth)
            mark[node[0], node[1]] = 0
            edge_mesh, mesh, mark, info_on_pix = recursive_add_edge(edge_mesh, mesh, info_on_pix, renode, mark)
            break
        if len(cc) == 1:
            node = [node for node in cc][0]
            hx, hy = node[0], node[1]
            nine_nes = set([(x, y, info_on_pix[(x, y)][0]['depth']) for x, y in [(hx, hy), (hx + 1, hy), (hx - 1, hy), (hx, hy + 1), (hx, hy - 1),
                                                                                  (hx + 1, hy + 1), (hx - 1, hy - 1), (hx - 1, hy + 1), (hx + 1, hy - 1)] \
                                if info_on_pix.get((x, y)) is not None and mesh.has_node((x, y, info_on_pix[(x, y)][0]['depth']))])
            ne_sub_mesh = mesh.subgraph(nine_nes).copy()
            ne_ccs = netx.connected_components(ne_sub_mesh)
            for ne_cc in ne_ccs:
                if node in ne_cc:
                    re_depth = {'value' : 0, 'count': 0}
                    for ne in ne_cc:
                        if abs(ne[0] - node[0]) + abs(ne[1] - node[1]) == 1:
                            mesh.add_edge(node, ne)
                            re_depth['value'] += ne[2]
                            re_depth['count'] += 1.
                    re_depth = re_depth['value'] / re_depth['count']
                    mapping_dict = {node: (node[0], node[1], re_depth)}
                    info_on_pix, mesh, edge_mesh = update_info(mapping_dict, info_on_pix, mesh, edge_mesh)
                    depth[node[0], node[1]] = abs(re_depth)
                    mark[node[0], node[1]] = 0


    return mesh, info_on_pix, edge_mesh, depth, mark

def context_and_holes(mesh, edge_ccs, config, specific_edge_id, specific_edge_loc, depth_feat_model,
                      connect_points_ccs=None, inpaint_iter=0, filter_edge=False, vis_edge_id=None):
    edge_maps = np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1
    mask_info = {}
    for edge_id, edge_cc in enumerate(edge_ccs):
        for edge_node in edge_cc:
            edge_maps[edge_node[0], edge_node[1]] = edge_id

    context_ccs = [set() for x in range(len(edge_ccs))]
    extend_context_ccs = [set() for x in range(len(edge_ccs))]
    extend_erode_context_ccs = [set() for x in range(len(edge_ccs))]
    extend_edge_ccs = [set() for x in range(len(edge_ccs))]
    accomp_extend_context_ccs = [set() for x in range(len(edge_ccs))]
    erode_context_ccs = [set() for x in range(len(edge_ccs))]
    broken_mask_ccs = [set() for x in range(len(edge_ccs))]
    invalid_extend_edge_ccs = [set() for x in range(len(edge_ccs))]
    intouched_ccs = [set() for x in range(len(edge_ccs))]
    redundant_ccs = [set() for x in range(len(edge_ccs))]
    if inpaint_iter == 0:
        background_thickness = config['background_thickness']
        context_thickness = config['context_thickness']
    else:
        background_thickness = config['background_thickness_2']
        context_thickness = config['context_thickness_2']

    mesh_nodes = mesh.nodes
    for edge_id, edge_cc in enumerate(edge_ccs):
        if context_thickness == 0 or (len(specific_edge_id) > 0 and edge_id not in specific_edge_id):
            continue
        edge_group = {}
        for edge_node in edge_cc:
            far_nodes = mesh_nodes[edge_node].get('far')
            if far_nodes is None:
                continue
            for far_node in far_nodes:
                if far_node in edge_cc:
                    continue
                context_ccs[edge_id].add(far_node)
                if mesh_nodes[far_node].get('edge_id') is not None:
                    if edge_group.get(mesh_nodes[far_node]['edge_id']) is None:
                        edge_group[mesh_nodes[far_node]['edge_id']] = set()
                    edge_group[mesh_nodes[far_node]['edge_id']].add(far_node)
        if len(edge_cc) > 2:
            for edge_key in [*edge_group.keys()]:
                if len(edge_group[edge_key]) == 1:
                    context_ccs[edge_id].remove([*edge_group[edge_key]][0])
    for edge_id, edge_cc in enumerate(edge_ccs):
        if inpaint_iter != 0:
            continue
        tmp_intouched_nodes = set()
        for edge_node in edge_cc:
            raw_intouched_nodes = set(mesh_nodes[edge_node].get('near')) if mesh_nodes[edge_node].get('near') is not None else set()
            tmp_intouched_nodes |= set([xx for xx in raw_intouched_nodes if mesh_nodes[xx].get('edge_id') is not None and \
                                                                         len(context_ccs[mesh_nodes[xx].get('edge_id')]) > 0])
        intouched_ccs[edge_id] |= tmp_intouched_nodes
        tmp_intouched_nodes = None
    mask_ccs = copy.deepcopy(edge_ccs)
    forbidden_len = 3
    forbidden_map = np.ones((mesh.graph['H'] - forbidden_len, mesh.graph['W'] - forbidden_len))
    forbidden_map = np.pad(forbidden_map, ((forbidden_len, forbidden_len), (forbidden_len, forbidden_len)), mode='constant').astype(np.bool)
    cur_tmp_mask_map = np.zeros_like(forbidden_map).astype(np.bool)
    passive_background = 10 if 10 is not None else background_thickness
    passive_context = 1 if 1 is not None else context_thickness

    for edge_id, edge_cc in enumerate(edge_ccs):
        cur_mask_cc = None; cur_mask_cc = []
        cur_context_cc = None; cur_context_cc = []
        cur_accomp_near_cc = None; cur_accomp_near_cc = []
        cur_invalid_extend_edge_cc = None; cur_invalid_extend_edge_cc = []
        cur_comp_far_cc = None; cur_comp_far_cc = []
        tmp_erode = []
        if len(context_ccs[edge_id]) == 0 or (len(specific_edge_id) > 0 and edge_id not in specific_edge_id):
            continue
        for i in range(max(background_thickness, context_thickness)):
            cur_tmp_mask_map.fill(False)
            if i == 0:
                tmp_mask_nodes = copy.deepcopy(mask_ccs[edge_id])
                tmp_intersect_nodes = []
                tmp_intersect_context_nodes = []
                mask_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
                context_depth = np.zeros((mesh.graph['H'], mesh.graph['W']))
                comp_cnt_depth = np.zeros((mesh.graph['H'], mesh.graph['W']))
                connect_map = np.zeros((mesh.graph['H'], mesh.graph['W']))
                for node in tmp_mask_nodes:
                    mask_map[node[0], node[1]] = True
                    depth_count = 0
                    if mesh_nodes[node].get('far') is not None:
                        for comp_cnt_node in mesh_nodes[node]['far']:
                            comp_cnt_depth[node[0], node[1]] += abs(comp_cnt_node[2])
                            depth_count += 1
                    if depth_count > 0:
                        comp_cnt_depth[node[0], node[1]] = comp_cnt_depth[node[0], node[1]] / depth_count
                    connect_node = []
                    if mesh_nodes[node].get('connect_point_id') is not None:
                        connect_node.append(mesh_nodes[node]['connect_point_id'])
                    connect_point_id = np.bincount(connect_node).argmax() if len(connect_node) > 0 else -1
                    if connect_point_id > -1 and connect_points_ccs is not None:
                        for xx in connect_points_ccs[connect_point_id]:
                            if connect_map[xx[0], xx[1]] == 0:
                                connect_map[xx[0], xx[1]] = xx[2]
                    if mesh_nodes[node].get('connect_point_exception') is not None:
                        for xx in mesh_nodes[node]['connect_point_exception']:
                            if connect_map[xx[0], xx[1]] == 0:
                                connect_map[xx[0], xx[1]] = xx[2]
                tmp_context_nodes = [*context_ccs[edge_id]]
                tmp_erode.append([*context_ccs[edge_id]])
                context_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
                if (context_map.astype(np.uint8) * mask_map.astype(np.uint8)).max() > 0:
                    import pdb; pdb.set_trace()
                for node in tmp_context_nodes:
                    context_map[node[0], node[1]] = True
                    context_depth[node[0], node[1]] = node[2]
                context_map[mask_map == True] = False
                if (context_map.astype(np.uint8) * mask_map.astype(np.uint8)).max() > 0:
                    import pdb; pdb.set_trace()
                tmp_intouched_nodes = [*intouched_ccs[edge_id]]
                intouched_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
                for node in tmp_intouched_nodes: intouched_map[node[0], node[1]] = True
                intouched_map[mask_map == True] = False
                tmp_redundant_nodes = set()
                tmp_noncont_nodes = set()
                noncont_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
                intersect_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
                intersect_context_map = np.zeros((mesh.graph['H'], mesh.graph['W']), dtype=np.bool)
            if i > passive_background and inpaint_iter == 0:
                new_tmp_intersect_nodes = None
                new_tmp_intersect_nodes = []
                for node in tmp_intersect_nodes:
                    nes = mesh.neighbors(node)
                    for ne in nes:
                        if bool(context_map[ne[0], ne[1]]) is False and \
                        bool(mask_map[ne[0], ne[1]]) is False and \
                        bool(forbidden_map[ne[0], ne[1]]) is True and \
                        bool(intouched_map[ne[0], ne[1]]) is False and\
                        bool(intersect_map[ne[0], ne[1]]) is False and\
                        bool(intersect_context_map[ne[0], ne[1]]) is False:
                            break_flag = False
                            if (i - passive_background) % 2 == 0 and (i - passive_background) % 8 != 0:
                                four_nes = [xx for xx in[[ne[0] - 1, ne[1]], [ne[0] + 1, ne[1]], [ne[0], ne[1] - 1], [ne[0], ne[1] + 1]] \
                                                if 0 <= xx[0] < mesh.graph['H'] and 0 <= xx[1] < mesh.graph['W']]
                                for fne in four_nes:
                                    if bool(mask_map[fne[0], fne[1]]) is True:
                                        break_flag = True
                                        break
                                if break_flag is True:
                                    continue
                            intersect_map[ne[0], ne[1]] = True
                            new_tmp_intersect_nodes.append(ne)
                tmp_intersect_nodes = None
                tmp_intersect_nodes = new_tmp_intersect_nodes

            if i > passive_context and inpaint_iter == 1:
                new_tmp_intersect_context_nodes = None
                new_tmp_intersect_context_nodes = []
                for node in tmp_intersect_context_nodes:
                    nes = mesh.neighbors(node)
                    for ne in nes:
                        if bool(context_map[ne[0], ne[1]]) is False and \
                        bool(mask_map[ne[0], ne[1]]) is False and \
                        bool(forbidden_map[ne[0], ne[1]]) is True and \
                        bool(intouched_map[ne[0], ne[1]]) is False and\
                        bool(intersect_map[ne[0], ne[1]]) is False and \
                        bool(intersect_context_map[ne[0], ne[1]]) is False:
                            intersect_context_map[ne[0], ne[1]] = True
                            new_tmp_intersect_context_nodes.append(ne)
                tmp_intersect_context_nodes = None
                tmp_intersect_context_nodes = new_tmp_intersect_context_nodes

            new_tmp_mask_nodes = None
            new_tmp_mask_nodes = []
            for node in tmp_mask_nodes:
                four_nes = {xx:[] for xx in [(node[0] - 1, node[1]), (node[0] + 1, node[1]), (node[0], node[1] - 1), (node[0], node[1] + 1)] if \
                            0 <= xx[0] < connect_map.shape[0] and 0 <= xx[1] < connect_map.shape[1]}
                if inpaint_iter > 0:
                    for ne in four_nes.keys():
                        if connect_map[ne[0], ne[1]] == True:
                            tmp_context_nodes.append((ne[0], ne[1], connect_map[ne[0], ne[1]]))
                            context_map[ne[0], ne[1]] = True
                nes = mesh.neighbors(node)
                if inpaint_iter > 0:
                    for ne in nes: four_nes[(ne[0], ne[1])].append(ne[2])
                    nes = []
                    for kfne, vfnes in four_nes.items(): vfnes.sort(key = lambda xx: abs(xx), reverse=True)
                    for kfne, vfnes in four_nes.items():
                        for vfne in vfnes: nes.append((kfne[0], kfne[1], vfne))
                for ne in nes:
                    if bool(context_map[ne[0], ne[1]]) is False and \
                       bool(mask_map[ne[0], ne[1]]) is False and \
                       bool(forbidden_map[ne[0], ne[1]]) is True and \
                       bool(intouched_map[ne[0], ne[1]]) is False and \
                       bool(intersect_map[ne[0], ne[1]]) is False and \
                       bool(intersect_context_map[ne[0], ne[1]]) is False:
                        if i == passive_background and inpaint_iter == 0:
                            if np.any(context_map[max(ne[0] - 1, 0):min(ne[0] + 2, mesh.graph['H']), max(ne[1] - 1, 0):min(ne[1] + 2, mesh.graph['W'])]) == True:
                                intersect_map[ne[0], ne[1]] = True
                                tmp_intersect_nodes.append(ne)
                                continue
                        if i < background_thickness:
                            if inpaint_iter == 0:
                                cur_mask_cc.append(ne)
                            elif mesh_nodes[ne].get('inpaint_id') == 1:
                                cur_mask_cc.append(ne)
                            else:
                                continue
                            mask_ccs[edge_id].add(ne)
                            if inpaint_iter == 0:
                                if comp_cnt_depth[node[0], node[1]] > 0 and comp_cnt_depth[ne[0], ne[1]] == 0:
                                    comp_cnt_depth[ne[0], ne[1]] = comp_cnt_depth[node[0], node[1]]
                                if mesh_nodes[ne].get('far') is not None:
                                    for comp_far_node in mesh_nodes[ne]['far']:
                                        cur_comp_far_cc.append(comp_far_node)
                                        cur_accomp_near_cc.append(ne)
                                        cur_invalid_extend_edge_cc.append(comp_far_node)
                                if mesh_nodes[ne].get('edge_id') is not None and \
                                    len(context_ccs[mesh_nodes[ne].get('edge_id')]) > 0:
                                    intouched_fars = set(mesh_nodes[ne].get('far')) if mesh_nodes[ne].get('far') is not None else set()
                                    accum_intouched_fars = set(intouched_fars)
                                    for intouched_far in intouched_fars:
                                        accum_intouched_fars |= set([*mesh.neighbors(intouched_far)])
                                    for intouched_far in accum_intouched_fars:
                                        if bool(mask_map[intouched_far[0], intouched_far[1]]) is True or \
                                        bool(context_map[intouched_far[0], intouched_far[1]]) is True:
                                            continue
                                        tmp_redundant_nodes.add(intouched_far)
                                        intouched_map[intouched_far[0], intouched_far[1]] = True
                                if mesh_nodes[ne].get('near') is not None:
                                    intouched_nears = set(mesh_nodes[ne].get('near'))
                                    for intouched_near in intouched_nears:
                                        if bool(mask_map[intouched_near[0], intouched_near[1]]) is True or \
                                        bool(context_map[intouched_near[0], intouched_near[1]]) is True:
                                            continue
                                        tmp_redundant_nodes.add(intouched_near)
                                        intouched_map[intouched_near[0], intouched_near[1]] = True
                        if not (mesh_nodes[ne].get('inpaint_id') != 1 and inpaint_iter == 1):
                            new_tmp_mask_nodes.append(ne)
                            mask_map[ne[0], ne[1]] = True
            tmp_mask_nodes = new_tmp_mask_nodes

            new_tmp_context_nodes = None
            new_tmp_context_nodes = []
            for node in tmp_context_nodes:
                nes = mesh.neighbors(node)
                if inpaint_iter > 0:
                    four_nes = {(node[0] - 1, node[1]):[], (node[0] + 1, node[1]):[], (node[0], node[1] - 1):[], (node[0], node[1] + 1):[]}
                    for ne in nes: four_nes[(ne[0], ne[1])].append(ne[2])
                    nes = []
                    for kfne, vfnes in four_nes.items(): vfnes.sort(key = lambda xx: abs(xx), reverse=True)
                    for kfne, vfnes in four_nes.items():
                        for vfne in vfnes: nes.append((kfne[0], kfne[1], vfne))
                for ne in nes:
                    mask_flag = (bool(mask_map[ne[0], ne[1]]) is False)
                    if bool(context_map[ne[0], ne[1]]) is False and mask_flag and \
                       bool(forbidden_map[ne[0], ne[1]]) is True and bool(noncont_map[ne[0], ne[1]]) is False and \
                       bool(intersect_context_map[ne[0], ne[1]]) is False:
                        if i == passive_context and inpaint_iter == 1:
                            mnes = mesh.neighbors(ne)
                            if any([mask_map[mne[0], mne[1]] == True for mne in mnes]) is True:
                                intersect_context_map[ne[0], ne[1]] = True
                                tmp_intersect_context_nodes.append(ne)
                                continue
                        if False and mesh_nodes[ne].get('near') is not None and mesh_nodes[ne].get('edge_id') != edge_id:
                            noncont_nears = set(mesh_nodes[ne].get('near'))
                            for noncont_near in noncont_nears:
                                if bool(context_map[noncont_near[0], noncont_near[1]]) is False:
                                    tmp_noncont_nodes.add(noncont_near)
                                    noncont_map[noncont_near[0], noncont_near[1]] = True
                        new_tmp_context_nodes.append(ne)
                        context_map[ne[0], ne[1]] = True
                        context_depth[ne[0], ne[1]] = ne[2]
            cur_context_cc.extend(new_tmp_context_nodes)
            tmp_erode.append(new_tmp_context_nodes)
            tmp_context_nodes = None
            tmp_context_nodes = new_tmp_context_nodes
            new_tmp_intouched_nodes = None; new_tmp_intouched_nodes = []

            for node in tmp_intouched_nodes:
                if bool(context_map[node[0], node[1]]) is True or bool(mask_map[node[0], node[1]]) is True:
                    continue
                nes = mesh.neighbors(node)

                for ne in nes:
                    if bool(context_map[ne[0], ne[1]]) is False and \
                       bool(mask_map[ne[0], ne[1]]) is False and \
                       bool(intouched_map[ne[0], ne[1]]) is False and \
                       bool(forbidden_map[ne[0], ne[1]]) is True:
                        new_tmp_intouched_nodes.append(ne)
                        intouched_map[ne[0], ne[1]] = True
            tmp_intouched_nodes = None
            tmp_intouched_nodes = set(new_tmp_intouched_nodes)
            new_tmp_redundant_nodes = None; new_tmp_redundant_nodes = []
            for node in tmp_redundant_nodes:
                if bool(context_map[node[0], node[1]]) is True or \
                   bool(mask_map[node[0], node[1]]) is True:
                    continue
                nes = mesh.neighbors(node)

                for ne in nes:
                    if bool(context_map[ne[0], ne[1]]) is False and \
                       bool(mask_map[ne[0], ne[1]]) is False and \
                       bool(intouched_map[ne[0], ne[1]]) is False and \
                       bool(forbidden_map[ne[0], ne[1]]) is True:
                        new_tmp_redundant_nodes.append(ne)
                        intouched_map[ne[0], ne[1]] = True
            tmp_redundant_nodes = None
            tmp_redundant_nodes = set(new_tmp_redundant_nodes)
            new_tmp_noncont_nodes = None; new_tmp_noncont_nodes = []
            for node in tmp_noncont_nodes:
                if bool(context_map[node[0], node[1]]) is True or \
                   bool(mask_map[node[0], node[1]]) is True:
                    continue
                nes = mesh.neighbors(node)
                rmv_flag = False
                for ne in nes:
                    if bool(context_map[ne[0], ne[1]]) is False and \
                       bool(mask_map[ne[0], ne[1]]) is False and \
                       bool(noncont_map[ne[0], ne[1]]) is False and \
                       bool(forbidden_map[ne[0], ne[1]]) is True:
                        patch_context_map = context_map[max(ne[0] - 1, 0):min(ne[0] + 2, context_map.shape[0]),
                                                        max(ne[1] - 1, 0):min(ne[1] + 2, context_map.shape[1])]
                        if bool(np.any(patch_context_map)) is True:
                            new_tmp_noncont_nodes.append(ne)
                            noncont_map[ne[0], ne[1]] = True
            tmp_noncont_nodes = None
            tmp_noncont_nodes = set(new_tmp_noncont_nodes)
        if inpaint_iter == 0:
            depth_dict = get_depth_from_maps(context_map, mask_map, context_depth, mesh.graph['H'], mesh.graph['W'], log_depth=config['log_depth'])
            mask_size = get_valid_size(depth_dict['mask'])
            mask_size = dilate_valid_size(mask_size, depth_dict['mask'], dilate=[20, 20])
            context_size = get_valid_size(depth_dict['context'])
            context_size = dilate_valid_size(context_size, depth_dict['context'], dilate=[20, 20])
            union_size = size_operation(mask_size, context_size, operation='+')
            depth_dict = depth_inpainting(None, None, None, None, mesh, config, union_size, depth_feat_model, None, given_depth_dict=depth_dict, spdb=False)
            near_depth_map, raw_near_depth_map = np.zeros((mesh.graph['H'], mesh.graph['W'])), np.zeros((mesh.graph['H'], mesh.graph['W']))
            filtered_comp_far_cc, filtered_accomp_near_cc = set(), set()
            for node in cur_accomp_near_cc:
                near_depth_map[node[0], node[1]] = depth_dict['output'][node[0], node[1]]
                raw_near_depth_map[node[0], node[1]] = node[2]
            for node in cur_comp_far_cc:
                four_nes = [xx for xx in [(node[0] - 1, node[1]), (node[0] + 1, node[1]), (node[0], node[1] - 1), (node[0], node[1] + 1)] \
                            if 0 <= xx[0] < mesh.graph['H'] and 0 <= xx[1] < mesh.graph['W'] and \
                            near_depth_map[xx[0], xx[1]] != 0 and \
                            abs(near_depth_map[xx[0], xx[1]]) < abs(node[2])]
                if len(four_nes) > 0:
                    filtered_comp_far_cc.add(node)
                for ne in four_nes:
                    filtered_accomp_near_cc.add((ne[0], ne[1], -abs(raw_near_depth_map[ne[0], ne[1]])))
            cur_comp_far_cc, cur_accomp_near_cc = filtered_comp_far_cc, filtered_accomp_near_cc
        mask_ccs[edge_id] |= set(cur_mask_cc)
        context_ccs[edge_id] |= set(cur_context_cc)
        accomp_extend_context_ccs[edge_id] |= set(cur_accomp_near_cc).intersection(cur_mask_cc)
        extend_edge_ccs[edge_id] |= set(cur_accomp_near_cc).intersection(cur_mask_cc)
        extend_context_ccs[edge_id] |= set(cur_comp_far_cc)
        invalid_extend_edge_ccs[edge_id] |= set(cur_invalid_extend_edge_cc)
        erode_size = [0]
        for tmp in tmp_erode:
            erode_size.append(len(tmp))
            if len(erode_size) > 1:
                erode_size[-1] += erode_size[-2]
        if inpaint_iter == 0:
            tmp_width = config['depth_edge_dilate']
        else:
            tmp_width = 0
        while float(erode_size[tmp_width]) / (erode_size[-1] + 1e-6) > 0.3:
            tmp_width = tmp_width - 1
        try:
            if tmp_width == 0:
                erode_context_ccs[edge_id] = set([])
            else:
                erode_context_ccs[edge_id] = set(reduce(lambda x, y : x + y, [] + tmp_erode[:tmp_width]))
        except:
            import pdb; pdb.set_trace()
        erode_context_cc = copy.deepcopy(erode_context_ccs[edge_id])
        for erode_context_node in erode_context_cc:
            if (inpaint_iter != 0 and (mesh_nodes[erode_context_node].get('inpaint_id') is None or
                                        mesh_nodes[erode_context_node].get('inpaint_id') == 0)):
                erode_context_ccs[edge_id].remove(erode_context_node)
            else:
                context_ccs[edge_id].remove(erode_context_node)
        context_map = np.zeros((mesh.graph['H'], mesh.graph['W']))
        for context_node in context_ccs[edge_id]:
            context_map[context_node[0], context_node[1]] = 1
        extend_context_ccs[edge_id] = extend_context_ccs[edge_id] - mask_ccs[edge_id] - accomp_extend_context_ccs[edge_id]
    if inpaint_iter == 0:
        all_ecnt_cc = set()
        for ecnt_id, ecnt_cc in enumerate(extend_context_ccs):
            constraint_context_ids = set()
            constraint_context_cc = set()
            constraint_erode_context_cc = set()
            tmp_mask_cc = set()
            accum_context_cc = None; accum_context_cc = []
            for ecnt_node in accomp_extend_context_ccs[ecnt_id]:
                if edge_maps[ecnt_node[0], ecnt_node[1]] > -1:
                    constraint_context_ids.add(int(round(edge_maps[ecnt_node[0], ecnt_node[1]])))
            constraint_erode_context_cc = erode_context_ccs[ecnt_id]
            for constraint_context_id in constraint_context_ids:
                constraint_context_cc = constraint_context_cc | context_ccs[constraint_context_id] | erode_context_ccs[constraint_context_id]
                constraint_erode_context_cc = constraint_erode_context_cc | erode_context_ccs[constraint_context_id]
            for i in range(background_thickness):
                if i == 0:
                    tmp_context_nodes = copy.deepcopy(ecnt_cc)
                    tmp_invalid_context_nodes = copy.deepcopy(invalid_extend_edge_ccs[ecnt_id])
                    tmp_mask_nodes = copy.deepcopy(accomp_extend_context_ccs[ecnt_id])
                    tmp_context_map = np.zeros((mesh.graph['H'], mesh.graph['W'])).astype(np.bool)
                    tmp_mask_map = np.zeros((mesh.graph['H'], mesh.graph['W'])).astype(np.bool)
                    tmp_invalid_context_map = np.zeros((mesh.graph['H'], mesh.graph['W'])).astype(np.bool)
                    for node in tmp_mask_nodes:
                        tmp_mask_map[node[0], node[1]] = True
                    for node in context_ccs[ecnt_id]:
                        tmp_context_map[node[0], node[1]] = True
                    for node in erode_context_ccs[ecnt_id]:
                        tmp_context_map[node[0], node[1]] = True
                    for node in extend_context_ccs[ecnt_id]:
                        tmp_context_map[node[0], node[1]] = True
                    for node in invalid_extend_edge_ccs[ecnt_id]:
                        tmp_invalid_context_map[node[0], node[1]] = True
                    init_invalid_context_map = tmp_invalid_context_map.copy()
                    init_context_map = tmp
                    if (tmp_mask_map.astype(np.uint8) * tmp_context_map.astype(np.uint8)).max() > 0:
                        import pdb; pdb.set_trace()
                    if vis_edge_id is not None and ecnt_id == vis_edge_id:
                        f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True)
                        ax1.imshow(tmp_context_map * 1); ax2.imshow(init_invalid_context_map * 1 + tmp_context_map * 2)
                        plt.show()
                        import pdb; pdb.set_trace()
                else:
                    tmp_context_nodes = new_tmp_context_nodes
                    new_tmp_context_nodes = None
                    tmp_mask_nodes = new_tmp_mask_nodes
                    new_tmp_mask_nodes = None
                    tmp_invalid_context_nodes = new_tmp_invalid_context_nodes
                    new_tmp_invalid_context_nodes = None
                new_tmp_context_nodes = None
                new_tmp_context_nodes = []
                new_tmp_invalid_context_nodes = None
                new_tmp_invalid_context_nodes = []
                new_tmp_mask_nodes = set([])
                for node in tmp_context_nodes:
                    for ne in mesh.neighbors(node):
                        if ne in constraint_context_cc and \
                            bool(tmp_mask_map[ne[0], ne[1]]) is False and \
                            bool(tmp_context_map[ne[0], ne[1]]) is False and \
                            bool(forbidden_map[ne[0], ne[1]]) is True:
                            new_tmp_context_nodes.append(ne)
                            tmp_context_map[ne[0], ne[1]] = True
                accum_context_cc.extend(new_tmp_context_nodes)
                for node in tmp_invalid_context_nodes:
                    for ne in mesh.neighbors(node):
                        if bool(tmp_mask_map[ne[0], ne[1]]) is False and \
                           bool(tmp_context_map[ne[0], ne[1]]) is False and \
                           bool(tmp_invalid_context_map[ne[0], ne[1]]) is False and \
                           bool(forbidden_map[ne[0], ne[1]]) is True:
                            tmp_invalid_context_map[ne[0], ne[1]] = True
                            new_tmp_invalid_context_nodes.append(ne)
                for node in tmp_mask_nodes:
                    for ne in mesh.neighbors(node):
                        if bool(tmp_mask_map[ne[0], ne[1]]) is False and \
                           bool(tmp_context_map[ne[0], ne[1]]) is False and \
                           bool(tmp_invalid_context_map[ne[0], ne[1]]) is False and \
                           bool(forbidden_map[ne[0], ne[1]]) is True:
                            new_tmp_mask_nodes.add(ne)
                            tmp_mask_map[ne[0], ne[1]] = True
            init_invalid_context_map[tmp_context_map] = False
            _, tmp_label_map = cv2.connectedComponents((init_invalid_context_map | tmp_context_map).astype(np.uint8), connectivity=8)
            tmp_label_ids = set(np.unique(tmp_label_map[init_invalid_context_map]))
            if (tmp_mask_map.astype(np.uint8) * tmp_context_map.astype(np.uint8)).max() > 0:
                import pdb; pdb.set_trace()
            if vis_edge_id is not None and ecnt_id == vis_edge_id:
                f, ((ax1, ax2)) = plt.subplots(1, 2, sharex=True, sharey=True)
                ax1.imshow(tmp_label_map); ax2.imshow(init_invalid_context_map * 1 + tmp_context_map * 2)
                plt.show()
                import pdb; pdb.set_trace()
            extend_context_ccs[ecnt_id] |= set(accum_context_cc)
            extend_context_ccs[ecnt_id] = extend_context_ccs[ecnt_id] - mask_ccs[ecnt_id]
            extend_erode_context_ccs[ecnt_id] = extend_context_ccs[ecnt_id] & constraint_erode_context_cc
            extend_context_ccs[ecnt_id] = extend_context_ccs[ecnt_id] - extend_erode_context_ccs[ecnt_id] - erode_context_ccs[ecnt_id]
            tmp_context_cc = context_ccs[ecnt_id] - extend_erode_context_ccs[ecnt_id] - erode_context_ccs[ecnt_id]
            if len(tmp_context_cc) > 0:
                context_ccs[ecnt_id] = tmp_context_cc
            tmp_mask_cc = tmp_mask_cc - context_ccs[ecnt_id] - erode_context_ccs[ecnt_id]
            mask_ccs[ecnt_id] = mask_ccs[ecnt_id] | tmp_mask_cc

    return context_ccs, mask_ccs, broken_mask_ccs, edge_ccs, erode_context_ccs, invalid_extend_edge_ccs, edge_maps, extend_context_ccs, extend_edge_ccs, extend_erode_context_ccs

def DL_inpaint_edge(mesh,
                    info_on_pix,
                    config,
                    image,
                    depth,
                    context_ccs,
                    erode_context_ccs,
                    extend_context_ccs,
                    extend_erode_context_ccs,
                    mask_ccs,
                    broken_mask_ccs,
                    edge_ccs,
                    extend_edge_ccs,
                    init_mask_connect,
                    edge_maps,
                    rgb_model=None,
                    depth_edge_model=None,
                    depth_edge_model_init=None,
                    depth_feat_model=None,
                    specific_edge_id=-1,
                    specific_edge_loc=None,
                    inpaint_iter=0):

    if isinstance(config["gpu_ids"], int) and (config["gpu_ids"] >= 0):
        device = config["gpu_ids"]
    else:
        device = "cpu"

    edge_map = np.zeros_like(depth)
    new_edge_ccs = [set() for _ in range(len(edge_ccs))]
    edge_maps_with_id = edge_maps
    edge_condition = lambda x, m: m.nodes[x].get('far') is not None and len(m.nodes[x].get('far')) > 0
    edge_map = get_map_from_ccs(edge_ccs, mesh.graph['H'], mesh.graph['W'], mesh, edge_condition)
    np_depth, np_image = depth.copy(), image.copy()
    image_c = image.shape[-1]
    image = torch.FloatTensor(image.transpose(2, 0, 1)).unsqueeze(0).to(device)
    if depth.ndim < 3:
        depth = depth[..., None]
    depth = torch.FloatTensor(depth.transpose(2, 0, 1)).unsqueeze(0).to(device)
    mesh.graph['max_edge_id'] = len(edge_ccs)
    connnect_points_ccs = [set() for _ in range(len(edge_ccs))]
    gp_time, tmp_mesh_time, bilateral_time = 0, 0, 0
    edges_infos = dict()
    edges_in_mask = [set() for _ in range(len(edge_ccs))]
    tmp_specific_edge_id = []
    for edge_id, (context_cc, mask_cc, erode_context_cc, extend_context_cc, edge_cc) in enumerate(zip(context_ccs, mask_ccs, erode_context_ccs, extend_context_ccs, edge_ccs)):
        if len(specific_edge_id) > 0:
            if edge_id not in specific_edge_id:
                continue
        if len(context_cc) < 1 or len(mask_cc) < 1:
            continue
        edge_dict = get_edge_from_nodes(context_cc | extend_context_cc, erode_context_cc | extend_erode_context_ccs[edge_id], mask_cc, edge_cc, extend_edge_ccs[edge_id],
                                        mesh.graph['H'], mesh.graph['W'], mesh)
        edge_dict['edge'], end_depth_maps, _ = \
            filter_irrelevant_edge_new(edge_dict['self_edge'], edge_dict['comp_edge'],
                                    edge_map,
                                    edge_maps_with_id,
                                    edge_id,
                                    edge_dict['context'],
                                    edge_dict['depth'], mesh, context_cc | erode_context_cc | extend_context_cc | extend_erode_context_ccs[edge_id], spdb=False)
        if specific_edge_loc is not None and \
            (specific_edge_loc is not None and edge_dict['mask'][specific_edge_loc[0], specific_edge_loc[1]] == 0):
            continue
        mask_size = get_valid_size(edge_dict['mask'])
        mask_size = dilate_valid_size(mask_size, edge_dict['mask'], dilate=[20, 20])
        context_size = get_valid_size(edge_dict['context'])
        context_size = dilate_valid_size(context_size, edge_dict['context'], dilate=[20, 20])
        union_size = size_operation(mask_size, context_size, operation='+')
        patch_edge_dict = dict()
        patch_edge_dict['mask'], patch_edge_dict['context'], patch_edge_dict['rgb'], \
            patch_edge_dict['disp'], patch_edge_dict['edge'] = \
            crop_maps_by_size(union_size, edge_dict['mask'], edge_dict['context'],
                                edge_dict['rgb'], edge_dict['disp'], edge_dict['edge'])
        x_anchor, y_anchor = [union_size['x_min'], union_size['x_max']], [union_size['y_min'], union_size['y_max']]
        tensor_edge_dict = convert2tensor(patch_edge_dict)
        input_edge_feat = torch.cat((tensor_edge_dict['rgb'],
                                        tensor_edge_dict['disp'],
                                        tensor_edge_dict['edge'],
                                        1 - tensor_edge_dict['context'],
                                        tensor_edge_dict['mask']), dim=1)
        if require_depth_edge(patch_edge_dict['edge'], patch_edge_dict['mask']) and inpaint_iter == 0:
            with torch.no_grad():
                depth_edge_output = depth_edge_model.forward_3P(tensor_edge_dict['mask'],
                                                                tensor_edge_dict['context'],
                                                                tensor_edge_dict['rgb'],
                                                                tensor_edge_dict['disp'],
                                                                tensor_edge_dict['edge'],
                                                                unit_length=128,
                                                                cuda=device)
                depth_edge_output = depth_edge_output.cpu()
            tensor_edge_dict['output'] = (depth_edge_output> config['ext_edge_threshold']).float() * tensor_edge_dict['mask'] + tensor_edge_dict['edge']
        else:
            tensor_edge_dict['output'] = tensor_edge_dict['edge']
            depth_edge_output = tensor_edge_dict['edge'] + 0
        patch_edge_dict['output'] = tensor_edge_dict['output'].squeeze().data.cpu().numpy()
        edge_dict['output'] = np.zeros((mesh.graph['H'], mesh.graph['W']))
        edge_dict['output'][union_size['x_min']:union_size['x_max'], union_size['y_min']:union_size['y_max']] = \
            patch_edge_dict['output']
        if require_depth_edge(patch_edge_dict['edge'], patch_edge_dict['mask']) and inpaint_iter == 0:
            if ((depth_edge_output> config['ext_edge_threshold']).float() * tensor_edge_dict['mask']).max() > 0:
                try:
                    edge_dict['fpath_map'], edge_dict['npath_map'], break_flag, npaths, fpaths, invalid_edge_id = \
                        clean_far_edge_new(edge_dict['output'], end_depth_maps, edge_dict['mask'], edge_dict['context'], mesh, info_on_pix, edge_dict['self_edge'], inpaint_iter, config)
                except:
                    import pdb; pdb.set_trace()
                pre_npath_map = edge_dict['npath_map'].copy()
                if config.get('repeat_inpaint_edge') is True:
                    for _ in range(2):
                        tmp_input_edge = ((edge_dict['npath_map'] > -1) + edge_dict['edge']).clip(0, 1)
                        patch_tmp_input_edge = crop_maps_by_size(union_size, tmp_input_edge)[0]
                        tensor_input_edge = torch.FloatTensor(patch_tmp_input_edge)[None, None, ...]
                        depth_edge_output = depth_edge_model.forward_3P(tensor_edge_dict['mask'],
                                                    tensor_edge_dict['context'],
                                                    tensor_edge_dict['rgb'],
                                                    tensor_edge_dict['disp'],
                                                    tensor_input_edge,
                                                    unit_length=128,
                                                    cuda=device)
                        depth_edge_output = depth_edge_output.cpu()
                        depth_edge_output = (depth_edge_output> config['ext_edge_threshold']).float() * tensor_edge_dict['mask'] + tensor_edge_dict['edge']
                        depth_edge_output = depth_edge_output.squeeze().data.cpu().numpy()
                        full_depth_edge_output = np.zeros((mesh.graph['H'], mesh.graph['W']))
                        full_depth_edge_output[union_size['x_min']:union_size['x_max'], union_size['y_min']:union_size['y_max']] = \
                            depth_edge_output
                        edge_dict['fpath_map'], edge_dict['npath_map'], break_flag, npaths, fpaths, invalid_edge_id = \
                            clean_far_edge_new(full_depth_edge_output, end_depth_maps, edge_dict['mask'], edge_dict['context'], mesh, info_on_pix, edge_dict['self_edge'], inpaint_iter, config)
                for nid in npaths.keys():
                    npath, fpath = npaths[nid], fpaths[nid]
                    start_mx, start_my, end_mx, end_my = -1, -1, -1, -1
                    if end_depth_maps[npath[0][0], npath[0][1]] != 0:
                        start_mx, start_my = npath[0][0], npath[0][1]
                    if end_depth_maps[npath[-1][0], npath[-1][1]] != 0:
                        end_mx, end_my = npath[-1][0], npath[-1][1]
                    if start_mx == -1:
                        import pdb; pdb.set_trace()
                    valid_end_pt = () if end_mx == -1 else (end_mx, end_my, info_on_pix[(end_mx, end_my)][0]['depth'])
                    new_edge_info = dict(fpath=fpath,
                                         npath=npath,
                                         cont_end_pts=valid_end_pt,
                                         mask_id=edge_id,
                                         comp_edge_id=nid,
                                         depth=end_depth_maps[start_mx, start_my])
                    if edges_infos.get((start_mx, start_my)) is None:
                        edges_infos[(start_mx, start_my)] = []
                    edges_infos[(start_mx, start_my)].append(new_edge_info)
                    edges_in_mask[edge_id].add((start_mx, start_my))
                    if len(valid_end_pt) > 0:
                        new_edge_info = dict(fpath=fpath[::-1],
                                             npath=npath[::-1],
                                             cont_end_pts=(start_mx, start_my, info_on_pix[(start_mx, start_my)][0]['depth']),
                                             mask_id=edge_id,
                                             comp_edge_id=nid,
                                             depth=end_depth_maps[end_mx, end_my])
                        if edges_infos.get((end_mx, end_my)) is None:
                            edges_infos[(end_mx, end_my)] = []
                        edges_infos[(end_mx, end_my)].append(new_edge_info)
                        edges_in_mask[edge_id].add((end_mx, end_my))
    for edge_id, (context_cc, mask_cc, erode_context_cc, extend_context_cc, edge_cc) in enumerate(zip(context_ccs, mask_ccs, erode_context_ccs, extend_context_ccs, edge_ccs)):
        if len(specific_edge_id) > 0:
            if edge_id not in specific_edge_id:
                continue
        if len(context_cc) < 1 or len(mask_cc) < 1:
            continue
        edge_dict = get_edge_from_nodes(context_cc | extend_context_cc, erode_context_cc | extend_erode_context_ccs[edge_id], mask_cc, edge_cc, extend_edge_ccs[edge_id],
                                        mesh.graph['H'], mesh.graph['W'], mesh)
        if specific_edge_loc is not None and \
            (specific_edge_loc is not None and edge_dict['mask'][specific_edge_loc[0], specific_edge_loc[1]] == 0):
            continue
        else:
            tmp_specific_edge_id.append(edge_id)
        edge_dict['edge'], end_depth_maps, _ = \
            filter_irrelevant_edge_new(edge_dict['self_edge'], edge_dict['comp_edge'],
                                    edge_map,
                                    edge_maps_with_id,
                                    edge_id,
                                    edge_dict['context'],
                                    edge_dict['depth'], mesh, context_cc | erode_context_cc | extend_context_cc | extend_erode_context_ccs[edge_id], spdb=False)
        discard_map = np.zeros_like(edge_dict['edge'])
        mask_size = get_valid_size(edge_dict['mask'])
        mask_size = dilate_valid_size(mask_size, edge_dict['mask'], dilate=[20, 20])
        context_size = get_valid_size(edge_dict['context'])
        context_size = dilate_valid_size(context_size, edge_dict['context'], dilate=[20, 20])
        union_size = size_operation(mask_size, context_size, operation='+')
        patch_edge_dict = dict()
        patch_edge_dict['mask'], patch_edge_dict['context'], patch_edge_dict['rgb'], \
            patch_edge_dict['disp'], patch_edge_dict['edge'] = \
            crop_maps_by_size(union_size, edge_dict['mask'], edge_dict['context'],
                                edge_dict['rgb'], edge_dict['disp'], edge_dict['edge'])
        x_anchor, y_anchor = [union_size['x_min'], union_size['x_max']], [union_size['y_min'], union_size['y_max']]
        tensor_edge_dict = convert2tensor(patch_edge_dict)
        input_edge_feat = torch.cat((tensor_edge_dict['rgb'],
                                        tensor_edge_dict['disp'],
                                        tensor_edge_dict['edge'],
                                        1 - tensor_edge_dict['context'],
                                        tensor_edge_dict['mask']), dim=1)
        edge_dict['output'] = edge_dict['edge'].copy()

        if require_depth_edge(patch_edge_dict['edge'], patch_edge_dict['mask']) and inpaint_iter == 0:
            edge_dict['fpath_map'], edge_dict['npath_map'] = edge_dict['fpath_map'] * 0 - 1, edge_dict['npath_map'] * 0 - 1
            end_pts = edges_in_mask[edge_id]
            for end_pt in end_pts:
                cur_edge_infos = edges_infos[(end_pt[0], end_pt[1])]
                cur_info = [xx for xx in cur_edge_infos if xx['mask_id'] == edge_id][0]
                other_infos = [xx for xx in cur_edge_infos if xx['mask_id'] != edge_id and len(xx['cont_end_pts']) > 0]
                if len(cur_info['cont_end_pts']) > 0 or (len(cur_info['cont_end_pts']) == 0 and len(other_infos) == 0):
                    for fnode in cur_info['fpath']:
                        edge_dict['fpath_map'][fnode[0], fnode[1]] = cur_info['comp_edge_id']
                    for fnode in cur_info['npath']:
                        edge_dict['npath_map'][fnode[0], fnode[1]] = cur_info['comp_edge_id']
            fnmap = edge_dict['fpath_map'] * 1
            fnmap[edge_dict['npath_map'] != -1] = edge_dict['npath_map'][edge_dict['npath_map'] != -1]
            for end_pt in end_pts:
                cur_edge_infos = edges_infos[(end_pt[0], end_pt[1])]
                cur_info = [xx for xx in cur_edge_infos if xx['mask_id'] == edge_id][0]
                cur_depth = cur_info['depth']
                other_infos = [xx for xx in cur_edge_infos if xx['mask_id'] != edge_id and len(xx['cont_end_pts']) > 0]
                comp_edge_id = cur_info['comp_edge_id']
                if len(cur_info['cont_end_pts']) == 0 and len(other_infos) > 0:
                    other_infos = sorted(other_infos, key=lambda aa: abs(abs(aa['cont_end_pts'][2]) - abs(cur_depth)))
                    for other_info in other_infos:
                        tmp_fmap, tmp_nmap = np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1, np.zeros((mesh.graph['H'], mesh.graph['W'])) - 1
                        for fnode in other_info['fpath']:
                            if fnmap[fnode[0], fnode[1]] != -1:
                                tmp_fmap = tmp_fmap * 0 - 1
                                break
                            else:
                                tmp_fmap[fnode[0], fnode[1]] = comp_edge_id
                        if fnmap[fnode[0], fnode[1]] != -1:
                            continue
                        for fnode in other_info['npath']:
                            if fnmap[fnode[0], fnode[1]] != -1:
                                tmp_nmap = tmp_nmap * 0 - 1
                                break
                            else:
                                tmp_nmap[fnode[0], fnode[1]] = comp_edge_id
                        if fnmap[fnode[0], fnode[1]] != -1:
                            continue
                        break
                    if min(tmp_fmap.max(), tmp_nmap.max()) != -1:
                        edge_dict['fpath_map'] = tmp_fmap
                        edge_dict['fpath_map'][edge_dict['valid_area'] == 0] = -1
                        edge_dict['npath_map'] = tmp_nmap
                        edge_dict['npath_map'][edge_dict['valid_area'] == 0] = -1
                        discard_map = ((tmp_nmap != -1).astype(np.uint8) + (tmp_fmap != -1).astype(np.uint8)) * edge_dict['mask']
                    else:
                        for fnode in cur_info['fpath']:
                            edge_dict['fpath_map'][fnode[0], fnode[1]] = cur_info['comp_edge_id']
                        for fnode in cur_info['npath']:
                            edge_dict['npath_map'][fnode[0], fnode[1]] = cur_info['comp_edge_id']
            if edge_dict['npath_map'].min() == 0 or edge_dict['fpath_map'].min() == 0:
                import pdb; pdb.set_trace()
            edge_dict['output'] = (edge_dict['npath_map'] > -1) * edge_dict['mask'] + edge_dict['context'] * edge_dict['edge']
        mesh, _, _, _ = create_placeholder(edge_dict['context'], edge_dict['mask'],
                                  edge_dict['depth'], edge_dict['fpath_map'],
                                  edge_dict['npath_map'], mesh, inpaint_iter,
                                  edge_ccs,
                                  extend_edge_ccs[edge_id],
                                  edge_maps_with_id,
                                  edge_id)

        dxs, dys = np.where(discard_map != 0)
        for dx, dy in zip(dxs, dys):
            mesh.nodes[(dx, dy)]['inpaint_twice'] = False
        depth_dict = depth_inpainting(context_cc, extend_context_cc, erode_context_cc | extend_erode_context_ccs[edge_id], mask_cc, mesh, config, union_size, depth_feat_model, edge_dict['output'])
        refine_depth_output = depth_dict['output']*depth_dict['mask']
        for near_id in np.unique(edge_dict['npath_map'])[1:]:
            refine_depth_output = refine_depth_around_edge(refine_depth_output.copy(),
                                                            (edge_dict['fpath_map'] == near_id).astype(np.uint8) * edge_dict['mask'],
                                                            (edge_dict['fpath_map'] == near_id).astype(np.uint8),
                                                            (edge_dict['npath_map'] == near_id).astype(np.uint8) * edge_dict['mask'],
                                                            depth_dict['mask'].copy(),
                                                            depth_dict['output'] * depth_dict['context'],
                                                            config)
        depth_dict['output'][depth_dict['mask'] > 0] = refine_depth_output[depth_dict['mask'] > 0]
        rgb_dict = get_rgb_from_nodes(context_cc | extend_context_cc,
                                      erode_context_cc | extend_erode_context_ccs[edge_id], mask_cc, mesh.graph['H'], mesh.graph['W'], mesh)
        if np.all(rgb_dict['mask'] == edge_dict['mask']) is False:
            import pdb; pdb.set_trace()
        rgb_dict['edge'] = edge_dict['output']
        patch_rgb_dict = dict()
        patch_rgb_dict['mask'], patch_rgb_dict['context'], patch_rgb_dict['rgb'], \
            patch_rgb_dict['edge'] = crop_maps_by_size(union_size, rgb_dict['mask'],
                                                        rgb_dict['context'], rgb_dict['rgb'],
                                                        rgb_dict['edge'])
        tensor_rgb_dict = convert2tensor(patch_rgb_dict)
        resize_rgb_dict = {k: v.clone() for k, v in tensor_rgb_dict.items()}
        max_hw = np.array([*patch_rgb_dict['mask'].shape[-2:]]).max()
        init_frac = config['largest_size'] / (np.array([*patch_rgb_dict['mask'].shape[-2:]]).prod() ** 0.5)
        resize_hw = [patch_rgb_dict['mask'].shape[-2] * init_frac, patch_rgb_dict['mask'].shape[-1] * init_frac]
        resize_max_hw = max(resize_hw)
        frac = (np.floor(resize_max_hw / 128.) * 128.) / max_hw
        if frac < 1:
            resize_mark = torch.nn.functional.interpolate(torch.cat((resize_rgb_dict['mask'],
                                                            resize_rgb_dict['context']),
                                                            dim=1),
                                                            scale_factor=frac,
                                                            mode='area')
            resize_rgb_dict['mask'] = (resize_mark[:, 0:1] > 0).float()
            resize_rgb_dict['context'] = (resize_mark[:, 1:2] == 1).float()
            resize_rgb_dict['context'][resize_rgb_dict['mask'] > 0] = 0
            resize_rgb_dict['rgb'] = torch.nn.functional.interpolate(resize_rgb_dict['rgb'],
                                                                        scale_factor=frac,
                                                                        mode='area')
            resize_rgb_dict['rgb'] = resize_rgb_dict['rgb'] * resize_rgb_dict['context']
            resize_rgb_dict['edge'] = torch.nn.functional.interpolate(resize_rgb_dict['edge'],
                                                                        scale_factor=frac,
                                                                        mode='area')
            resize_rgb_dict['edge'] = (resize_rgb_dict['edge'] > 0).float() * 0
            resize_rgb_dict['edge'] = resize_rgb_dict['edge'] * (resize_rgb_dict['context'] + resize_rgb_dict['mask'])
        rgb_input_feat = torch.cat((resize_rgb_dict['rgb'], resize_rgb_dict['edge']), dim=1)
        rgb_input_feat[:, 3] = 1 - rgb_input_feat[:, 3]
        resize_mask = open_small_mask(resize_rgb_dict['mask'], resize_rgb_dict['context'], 3, 41)
        specified_hole = resize_mask
        with torch.no_grad():
            rgb_output = rgb_model.forward_3P(specified_hole,
                                            resize_rgb_dict['context'],
                                            resize_rgb_dict['rgb'],
                                            resize_rgb_dict['edge'],
                                            unit_length=128,
                                            cuda=device)
            rgb_output = rgb_output.cpu()
            if config.get('gray_image') is True:
                rgb_output = rgb_output.mean(1, keepdim=True).repeat((1,3,1,1))
            rgb_output = rgb_output.cpu()
        resize_rgb_dict['output'] = rgb_output * resize_rgb_dict['mask'] + resize_rgb_dict['rgb']
        tensor_rgb_dict['output'] = resize_rgb_dict['output']
        if frac < 1:
            tensor_rgb_dict['output'] = torch.nn.functional.interpolate(tensor_rgb_dict['output'],
                                                                        size=tensor_rgb_dict['mask'].shape[-2:],
                                                                        mode='bicubic')
            tensor_rgb_dict['output'] = tensor_rgb_dict['output'] * \
                                         tensor_rgb_dict['mask'] + (tensor_rgb_dict['rgb'] * tensor_rgb_dict['context'])
        patch_rgb_dict['output'] = tensor_rgb_dict['output'].data.cpu().numpy().squeeze().transpose(1,2,0)
        rgb_dict['output'] = np.zeros((mesh.graph['H'], mesh.graph['W'], 3))
        rgb_dict['output'][union_size['x_min']:union_size['x_max'], union_size['y_min']:union_size['y_max']] = \
            patch_rgb_dict['output']

        if require_depth_edge(patch_edge_dict['edge'], patch_edge_dict['mask']) or inpaint_iter > 0:
            edge_occlusion = True
        else:
            edge_occlusion = False
        for node in erode_context_cc:
            if rgb_dict['mask'][node[0], node[1]] > 0:
                for info in info_on_pix[(node[0], node[1])]:
                    if abs(info['depth']) == abs(node[2]):
                        info['update_color'] = (rgb_dict['output'][node[0], node[1]] * 255).astype(np.uint8)
        if frac < 1.:
            depth_edge_dilate_2_color_flag = False
        else:
            depth_edge_dilate_2_color_flag = True
        hxs, hys = np.where((rgb_dict['mask'] > 0) & (rgb_dict['erode'] == 0))
        for hx, hy in zip(hxs, hys):
            real_depth = None
            if abs(depth_dict['output'][hx, hy]) <= abs(np_depth[hx, hy]):
                depth_dict['output'][hx, hy] = np_depth[hx, hy] + 0.01
            node = (hx, hy, -depth_dict['output'][hx, hy])
            if info_on_pix.get((node[0], node[1])) is not None:
                for info in info_on_pix.get((node[0], node[1])):
                    if info.get('inpaint_id') is None or abs(info['inpaint_id'] < mesh.nodes[(hx, hy)]['inpaint_id']):
                        pre_depth = info['depth'] if info.get('real_depth') is None else info['real_depth']
                        if abs(node[2]) < abs(pre_depth):
                            node = (node[0], node[1], -(abs(pre_depth) + 0.001))
            if mesh.has_node(node):
                real_depth = node[2]
            while True:
                if mesh.has_node(node):
                    node = (node[0], node[1], -(abs(node[2]) + 0.001))
                else:
                    break
            if real_depth == node[2]:
                real_depth = None
            cur_disp = 1./node[2]
            if not(mesh.has_node(node)):
                if not mesh.has_node((node[0], node[1])):
                    print("2D node not found.")
                    import pdb; pdb.set_trace()
                if inpaint_iter == 1:
                    paint = (rgb_dict['output'][hx, hy] * 255).astype(np.uint8)
                else:
                    paint = (rgb_dict['output'][hx, hy] * 255).astype(np.uint8)
                ndict = dict(color=paint,
                                synthesis=True,
                                disp=cur_disp,
                                cc_id=set([edge_id]),
                                overlap_number=1.0,
                                refine_depth=False,
                                edge_occlusion=edge_occlusion,
                                depth_edge_dilate_2_color_flag=depth_edge_dilate_2_color_flag,
                                real_depth=real_depth)
                mesh, _, _ = refresh_node((node[0], node[1]), mesh.nodes[(node[0], node[1])], node, ndict, mesh, stime=True)
                if inpaint_iter == 0 and mesh.degree(node) < 4:
                    connnect_points_ccs[edge_id].add(node)
            if info_on_pix.get((hx, hy)) is None:
                info_on_pix[(hx, hy)] = []
            new_info = {'depth':node[2],
                        'color': paint,
                        'synthesis':True,
                        'disp':cur_disp,
                        'cc_id':set([edge_id]),
                        'inpaint_id':inpaint_iter + 1,
                        'edge_occlusion':edge_occlusion,
                        'overlap_number':1.0,
                        'real_depth': real_depth}
            info_on_pix[(hx, hy)].append(new_info)
    specific_edge_id = tmp_specific_edge_id
    for erode_id, erode_context_cc in enumerate(erode_context_ccs):
        if len(specific_edge_id) > 0 and erode_id not in specific_edge_id:
            continue
        for erode_node in erode_context_cc:
            for info in info_on_pix[(erode_node[0], erode_node[1])]:
                if info['depth'] == erode_node[2]:
                    info['color'] = info['update_color']
                    mesh.nodes[erode_node]['color'] = info['update_color']
                    np_image[(erode_node[0], erode_node[1])] = info['update_color']
    new_edge_ccs = [set() for _ in range(mesh.graph['max_edge_id'] + 1)]
    for node in mesh.nodes:
        if len(node) == 2:
            mesh.remove_node(node)
            continue
        if mesh.nodes[node].get('edge_id') is not None and mesh.nodes[node].get('inpaint_id') == inpaint_iter + 1:
            if mesh.nodes[node].get('inpaint_twice') is False:
                continue
            try:
                new_edge_ccs[mesh.nodes[node].get('edge_id')].add(node)
            except:
                import pdb; pdb.set_trace()
    specific_mask_nodes = None
    if inpaint_iter == 0:
        mesh, info_on_pix = refine_color_around_edge(mesh, info_on_pix, new_edge_ccs, config, False)

    return mesh, info_on_pix, specific_mask_nodes, new_edge_ccs, connnect_points_ccs, np_image


def write_ply(image,
              depth,
              int_mtx,
              ply_name,
              config,
              rgb_model,
              depth_edge_model,
              depth_edge_model_init,
              depth_feat_model):
    depth = depth.astype(np.float64)
    input_mesh, xy2depth, image, depth = create_mesh(depth, image, int_mtx, config)

    H, W = input_mesh.graph['H'], input_mesh.graph['W']
    input_mesh = tear_edges(input_mesh, config['depth_threshold'], xy2depth)
    input_mesh, info_on_pix = generate_init_node(input_mesh, config, min_node_in_cc=200)
    edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=False)
    edge_canvas = np.zeros((H, W)) - 1

    input_mesh, info_on_pix, depth = reassign_floating_island(input_mesh, info_on_pix, image, depth)
    input_mesh = update_status(input_mesh, info_on_pix)
    specific_edge_id = []
    edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
    pre_depth = depth.copy()
    input_mesh, info_on_pix, edge_mesh, depth, aft_mark = remove_dangling(input_mesh, edge_ccs, edge_mesh, info_on_pix, image, depth, config)

    input_mesh, depth, info_on_pix = update_status(input_mesh, info_on_pix, depth)
    edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
    edge_canvas = np.zeros((H, W)) - 1

    mesh, info_on_pix, depth = fill_missing_node(input_mesh, info_on_pix, image, depth)
    if config['extrapolate_border'] is True:
        pre_depth = depth.copy()
        input_mesh, info_on_pix, depth = refresh_bord_depth(input_mesh, info_on_pix, image, depth)
        input_mesh = remove_node_feat(input_mesh, 'edge_id')
        aft_depth = depth.copy()
        input_mesh, info_on_pix, depth, image = enlarge_border(input_mesh, info_on_pix, depth, image, config)
        noext_H, noext_W = H, W
        H, W = image.shape[:2]
        input_mesh, info_on_pix = fill_dummy_bord(input_mesh, info_on_pix, image, depth, config)
        edge_ccs, input_mesh, edge_mesh = \
            group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
        input_mesh = combine_end_node(input_mesh, edge_mesh, edge_ccs, depth)
        input_mesh, depth, info_on_pix = update_status(input_mesh, info_on_pix, depth)
        edge_ccs, input_mesh, edge_mesh = \
            group_edges(input_mesh, config, image, remove_conflict_ordinal=True, spdb=False)
        input_mesh = remove_redundant_edge(input_mesh, edge_mesh, edge_ccs, info_on_pix, config, redundant_number=config['redundant_number'], spdb=False)
        input_mesh, depth, info_on_pix = update_status(input_mesh, info_on_pix, depth)
        edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
        input_mesh = combine_end_node(input_mesh, edge_mesh, edge_ccs, depth)
        input_mesh = remove_redundant_edge(input_mesh, edge_mesh, edge_ccs, info_on_pix, config, redundant_number=config['redundant_number'], invalid=True, spdb=False)
        input_mesh, depth, info_on_pix = update_status(input_mesh, info_on_pix, depth)
        edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
        input_mesh = combine_end_node(input_mesh, edge_mesh, edge_ccs, depth)
        input_mesh, depth, info_on_pix = update_status(input_mesh, info_on_pix, depth)
        edge_ccs, input_mesh, edge_mesh = group_edges(input_mesh, config, image, remove_conflict_ordinal=True)
        edge_condition = lambda x, m: m.nodes[x].get('far') is not None and len(m.nodes[x].get('far')) > 0
        edge_map = get_map_from_ccs(edge_ccs, input_mesh.graph['H'], input_mesh.graph['W'], input_mesh, edge_condition)
        other_edge_with_id = get_map_from_ccs(edge_ccs, input_mesh.graph['H'], input_mesh.graph['W'], real_id=True)
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="up")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="left")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="down")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="right")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="right-up")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="right-down")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="left-up")
        info_on_pix, input_mesh, image, depth, edge_ccs = extrapolate(input_mesh, info_on_pix, image, depth, other_edge_with_id, edge_map, edge_ccs,
                                                depth_edge_model, depth_feat_model, rgb_model, config, direc="left-down")
    specific_edge_loc = None
    specific_edge_id = []
    vis_edge_id = None
    context_ccs, mask_ccs, broken_mask_ccs, edge_ccs, erode_context_ccs, \
        init_mask_connect, edge_maps, extend_context_ccs, extend_edge_ccs, extend_erode_context_ccs = \
                                                                                context_and_holes(input_mesh,
                                                                                            edge_ccs,
                                                                                            config,
                                                                                            specific_edge_id,
                                                                                            specific_edge_loc,
                                                                                            depth_feat_model,
                                                                                            inpaint_iter=0,
                                                                                            vis_edge_id=vis_edge_id)
    edge_canvas = np.zeros((H, W))
    mask = np.zeros((H, W))
    context = np.zeros((H, W))
    vis_edge_ccs = filter_edge(input_mesh, edge_ccs, config)
    edge_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W'])) - 1
    specific_edge_loc = None
    FG_edge_maps = edge_maps.copy()
    edge_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W'])) - 1
    # for cc_id, cc in enumerate(edge_ccs):
    #     for node in cc:
    #         edge_canvas[node[0], node[1]] = cc_id
    # f, ((ax0, ax1, ax2)) = plt.subplots(1, 3, sharex=True, sharey=True); ax0.imshow(1./depth); ax1.imshow(image); ax2.imshow(edge_canvas); plt.show()
    input_mesh, info_on_pix, specific_edge_nodes, new_edge_ccs, connect_points_ccs, image = DL_inpaint_edge(input_mesh,
                                                                                                            info_on_pix,
                                                                                                            config,
                                                                                                            image,
                                                                                                            depth,
                                                                                                            context_ccs,
                                                                                                            erode_context_ccs,
                                                                                                            extend_context_ccs,
                                                                                                            extend_erode_context_ccs,
                                                                                                            mask_ccs,
                                                                                                            broken_mask_ccs,
                                                                                                            edge_ccs,
                                                                                                            extend_edge_ccs,
                                                                                                            init_mask_connect,
                                                                                                            edge_maps,
                                                                                                            rgb_model,
                                                                                                            depth_edge_model,
                                                                                                            depth_edge_model_init,
                                                                                                            depth_feat_model,
                                                                                                            specific_edge_id,
                                                                                                            specific_edge_loc,
                                                                                                            inpaint_iter=0)
    specific_edge_id = []
    edge_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W']))
    connect_points_ccs = [set() for _ in connect_points_ccs]
    context_ccs, mask_ccs, broken_mask_ccs, edge_ccs, erode_context_ccs, init_mask_connect, \
        edge_maps, extend_context_ccs, extend_edge_ccs, extend_erode_context_ccs = \
            context_and_holes(input_mesh, new_edge_ccs, config, specific_edge_id, specific_edge_loc, depth_feat_model, connect_points_ccs, inpaint_iter=1)
    mask_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W']))
    context_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W']))
    erode_context_ccs_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W']))
    edge_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W']))
    # edge_canvas = np.zeros((input_mesh.graph['H'], input_mesh.graph['W'])) - 1
    # for cc_id, cc in enumerate(edge_ccs):
    #     for node in cc:
    #         edge_canvas[node[0], node[1]] = cc_id
    specific_edge_id = []
    input_mesh, info_on_pix, specific_edge_nodes, new_edge_ccs, _, image = DL_inpaint_edge(input_mesh,
                                                                                    info_on_pix,
                                                                                    config,
                                                                                    image,
                                                                                    depth,
                                                                                    context_ccs,
                                                                                    erode_context_ccs,
                                                                                    extend_context_ccs,
                                                                                    extend_erode_context_ccs,
                                                                                    mask_ccs,
                                                                                    broken_mask_ccs,
                                                                                    edge_ccs,
                                                                                    extend_edge_ccs,
                                                                                    init_mask_connect,
                                                                                    edge_maps,
                                                                                    rgb_model,
                                                                                    depth_edge_model,
                                                                                    depth_edge_model_init,
                                                                                    depth_feat_model,
                                                                                    specific_edge_id,
                                                                                    specific_edge_loc,
                                                                                    inpaint_iter=1)
    vertex_id = 0
    input_mesh.graph['H'], input_mesh.graph['W'] = input_mesh.graph['noext_H'], input_mesh.graph['noext_W']
    background_canvas = np.zeros((input_mesh.graph['H'],
                                  input_mesh.graph['W'],
                                  3))
    ply_flag = config.get('save_ply')
    if ply_flag is True:
        node_str_list = []
    else:
        node_str_color = []
        node_str_point = []
    out_fmt = lambda x, x_flag: str(x) if x_flag is True else x
    point_time = 0
    hlight_time = 0
    cur_id_time = 0
    node_str_time = 0
    generate_face_time = 0
    point_list = []
    k_00, k_02, k_11, k_12 = \
        input_mesh.graph['cam_param_pix_inv'][0, 0], input_mesh.graph['cam_param_pix_inv'][0, 2], \
        input_mesh.graph['cam_param_pix_inv'][1, 1], input_mesh.graph['cam_param_pix_inv'][1, 2]
    w_offset = input_mesh.graph['woffset']
    h_offset = input_mesh.graph['hoffset']
    for pix_xy, pix_list in info_on_pix.items():
        for pix_idx, pix_info in enumerate(pix_list):
            pix_depth = pix_info['depth'] if pix_info.get('real_depth') is None else pix_info['real_depth']
            str_pt = [out_fmt(x, ply_flag) for x in reproject_3d_int_detail(pix_xy[0], pix_xy[1], pix_depth,
                      k_00, k_02, k_11, k_12, w_offset, h_offset)]
            if input_mesh.has_node((pix_xy[0], pix_xy[1], pix_info['depth'])) is False:
                return False
                continue
            if pix_info.get('overlap_number') is not None:
                str_color = [out_fmt(x, ply_flag) for x in (pix_info['color']/pix_info['overlap_number']).astype(np.uint8).tolist()]
            else:
                str_color = [out_fmt(x, ply_flag) for x in pix_info['color'].tolist()]
            if pix_info.get('edge_occlusion') is True:
                str_color.append(out_fmt(4, ply_flag))
            else:
                if pix_info.get('inpaint_id') is None:
                    str_color.append(out_fmt(1, ply_flag))
                else:
                    str_color.append(out_fmt(pix_info.get('inpaint_id') + 1, ply_flag))
            if pix_info.get('modified_border') is True or pix_info.get('ext_pixel') is True:
                if len(str_color) == 4:
                    str_color[-1] = out_fmt(5, ply_flag)
                else:
                    str_color.append(out_fmt(5, ply_flag))
            pix_info['cur_id'] = vertex_id
            input_mesh.nodes[(pix_xy[0], pix_xy[1], pix_info['depth'])]['cur_id'] = out_fmt(vertex_id, ply_flag)
            vertex_id += 1
            if ply_flag is True:
                node_str_list.append(' '.join(str_pt) + ' ' + ' '.join(str_color) + '\n')
            else:
                node_str_color.append(str_color)
                node_str_point.append(str_pt)
    str_faces = generate_face(input_mesh, info_on_pix, config)
    if config['save_ply'] is True:
        print("Writing mesh file %s ..." % ply_name)
        with open(ply_name, 'w') as ply_fi:
            ply_fi.write('ply\n' + 'format ascii 1.0\n')
            ply_fi.write('comment H ' + str(int(input_mesh.graph['H'])) + '\n')
            ply_fi.write('comment W ' + str(int(input_mesh.graph['W'])) + '\n')
            ply_fi.write('comment hFov ' + str(float(input_mesh.graph['hFov'])) + '\n')
            ply_fi.write('comment vFov ' + str(float(input_mesh.graph['vFov'])) + '\n')
            ply_fi.write('element vertex ' + str(len(node_str_list)) + '\n')
            ply_fi.write('property float x\n' + \
                         'property float y\n' + \
                         'property float z\n' + \
                         'property uchar red\n' + \
                         'property uchar green\n' + \
                         'property uchar blue\n' + \
                         'property uchar alpha\n')
            ply_fi.write('element face ' + str(len(str_faces)) + '\n')
            ply_fi.write('property list uchar int vertex_index\n')
            ply_fi.write('end_header\n')
            ply_fi.writelines(node_str_list)
            ply_fi.writelines(str_faces)
        ply_fi.close()
        return input_mesh
    else:
        H = int(input_mesh.graph['H'])
        W = int(input_mesh.graph['W'])
        hFov = input_mesh.graph['hFov']
        vFov = input_mesh.graph['vFov']
        node_str_color = np.array(node_str_color).astype(np.float32)
        node_str_color[..., :3] = node_str_color[..., :3] / 255.
        node_str_point = np.array(node_str_point)
        str_faces = np.array(str_faces)

        return node_str_point, node_str_color, str_faces, H, W, hFov, vFov

def read_ply(mesh_fi):
    ply_fi = open(mesh_fi, 'r')
    Height = None
    Width = None
    hFov = None
    vFov = None
    while True:
        line = ply_fi.readline().split('\n')[0]
        if line.startswith('element vertex'):
            num_vertex = int(line.split(' ')[-1])
        elif line.startswith('element face'):
            num_face = int(line.split(' ')[-1])
        elif line.startswith('comment'):
            if line.split(' ')[1] == 'H':
                Height = int(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'W':
                Width = int(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'hFov':
                hFov = float(line.split(' ')[-1].split('\n')[0])
            if line.split(' ')[1] == 'vFov':
                vFov = float(line.split(' ')[-1].split('\n')[0])
        elif line.startswith('end_header'):
            break
    contents = ply_fi.readlines()
    vertex_infos = contents[:num_vertex]
    face_infos = contents[num_vertex:]
    verts = []
    colors = []
    faces = []
    for v_info in vertex_infos:
        str_info = [float(v) for v in v_info.split('\n')[0].split(' ')]
        if len(str_info) == 6:
            vx, vy, vz, r, g, b = str_info
        else:
            vx, vy, vz, r, g, b, hi = str_info
        verts.append([vx, vy, vz])
        colors.append([r, g, b, hi])
    verts = np.array(verts)
    try:
        colors = np.array(colors)
        colors[..., :3] = colors[..., :3]/255.
    except:
        import pdb
        pdb.set_trace()

    for f_info in face_infos:
        _, v1, v2, v3 = [int(f) for f in f_info.split('\n')[0].split(' ')]
        faces.append([v1, v2, v3])
    faces = np.array(faces)


    return verts, colors, faces, Height, Width, hFov, vFov


class Canvas_view():
    def __init__(self,
                 fov,
                 verts,
                 faces,
                 colors,
                 canvas_size,
                 factor=1,
                 bgcolor='gray',
                 proj='perspective',
                 ):
        self.canvas = scene.SceneCanvas(bgcolor=bgcolor, size=(canvas_size*factor, canvas_size*factor))
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = 'perspective'
        self.view.camera.fov = fov
        self.mesh = visuals.Mesh(shading=None)
        self.mesh.attach(Alpha(1.0))
        self.view.add(self.mesh)
        self.tr = self.view.camera.transform
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])
        self.translate([0,0,0])
        self.rotate(axis=[1,0,0], angle=180)
        self.view_changed()

    def translate(self, trans=[0,0,0]):
        self.tr.translate(trans)

    def rotate(self, axis=[1,0,0], angle=0):
        self.tr.rotate(axis=axis, angle=angle)

    def view_changed(self):
        self.view.camera.view_changed()

    def render(self):
        return self.canvas.render()

    def reinit_mesh(self, verts, faces, colors):
        self.mesh.set_data(vertices=verts, faces=faces, vertex_colors=colors[:, :3])

    def reinit_camera(self, fov):
        self.view.camera.fov = fov
        self.view.camera.view_changed()


def output_3d_photo(verts, colors, faces, Height, Width, hFov, vFov, tgt_poses, video_traj_types, ref_pose,
                    output_dir, ref_image, int_mtx, config, image, videos_poses, video_basename, original_H=None, original_W=None,
                    border=None, depth=None, normal_canvas=None, all_canvas=None, mean_loc_depth=None):

    cam_mesh = netx.Graph()
    cam_mesh.graph['H'] = Height
    cam_mesh.graph['W'] = Width
    cam_mesh.graph['original_H'] = original_H
    cam_mesh.graph['original_W'] = original_W
    int_mtx_real_x = int_mtx[0] * Width
    int_mtx_real_y = int_mtx[1] * Height
    cam_mesh.graph['hFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_W']) / int_mtx_real_x[0]))
    cam_mesh.graph['vFov'] = 2 * np.arctan((1. / 2.) * ((cam_mesh.graph['original_H']) / int_mtx_real_y[1]))
    colors = colors[..., :3]

    fov_in_rad = max(cam_mesh.graph['vFov'], cam_mesh.graph['hFov'])
    fov = (fov_in_rad * 180 / np.pi)
    print("fov: " + str(fov))
    init_factor = 1
    if config.get('anti_flickering') is True:
        init_factor = 3
    if (cam_mesh.graph['original_H'] is not None) and (cam_mesh.graph['original_W'] is not None):
        canvas_w = cam_mesh.graph['original_W']
        canvas_h = cam_mesh.graph['original_H']
    else:
        canvas_w = cam_mesh.graph['W']
        canvas_h = cam_mesh.graph['H']
    canvas_size = max(canvas_h, canvas_w)
    if normal_canvas is None:
        normal_canvas = Canvas_view(fov,
                                    verts,
                                    faces,
                                    colors,
                                    canvas_size=canvas_size,
                                    factor=init_factor,
                                    bgcolor='gray',
                                    proj='perspective')
    else:
        normal_canvas.reinit_mesh(verts, faces, colors)
        normal_canvas.reinit_camera(fov)
    img = normal_canvas.render()
    backup_img, backup_all_img, all_img_wo_bound = img.copy(), img.copy() * 0, img.copy() * 0
    img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)
    if border is None:
        border = [0, img.shape[0], 0, img.shape[1]]
    H, W = cam_mesh.graph['H'], cam_mesh.graph['W']
    if (cam_mesh.graph['original_H'] is not None) and (cam_mesh.graph['original_W'] is not None):
        aspect_ratio = cam_mesh.graph['original_H'] / cam_mesh.graph['original_W']
    else:
        aspect_ratio = cam_mesh.graph['H'] / cam_mesh.graph['W']
    if aspect_ratio > 1:
        img_h_len = cam_mesh.graph['H'] if cam_mesh.graph.get('original_H') is None else cam_mesh.graph['original_H']
        img_w_len = img_h_len / aspect_ratio
        anchor = [0,
                  img.shape[0],
                  int(max(0, int((img.shape[1])//2 - img_w_len//2))),
                  int(min(int((img.shape[1])//2 + img_w_len//2), (img.shape[1])-1))]
    elif aspect_ratio <= 1:
        img_w_len = cam_mesh.graph['W'] if cam_mesh.graph.get('original_W') is None else cam_mesh.graph['original_W']
        img_h_len = img_w_len * aspect_ratio
        anchor = [int(max(0, int((img.shape[0])//2 - img_h_len//2))),
                  int(min(int((img.shape[0])//2 + img_h_len//2), (img.shape[0])-1)),
                  0,
                  img.shape[1]]
    anchor = np.array(anchor)
    plane_width = np.tan(fov_in_rad/2.) * np.abs(mean_loc_depth)
    for video_pose, video_traj_type in zip(videos_poses, video_traj_types):
        stereos = []
        tops = []; buttoms = []; lefts = []; rights = []
        for tp_id, tp in enumerate(video_pose):
            rel_pose = np.linalg.inv(np.dot(tp, np.linalg.inv(ref_pose)))
            axis, angle = transforms3d.axangles.mat2axangle(rel_pose[0:3, 0:3])
            normal_canvas.rotate(axis=axis, angle=(angle*180)/np.pi)
            normal_canvas.translate(rel_pose[:3,3])
            new_mean_loc_depth = mean_loc_depth - float(rel_pose[2, 3])
            if 'dolly' in video_traj_type:
                new_fov = float((np.arctan2(plane_width, np.array([np.abs(new_mean_loc_depth)])) * 180. / np.pi) * 2)
                normal_canvas.reinit_camera(new_fov)
            else:
                normal_canvas.reinit_camera(fov)
            normal_canvas.view_changed()
            img = normal_canvas.render()
            img = cv2.GaussianBlur(img,(int(init_factor//2 * 2 + 1), int(init_factor//2 * 2 + 1)), 0)
            img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)), interpolation=cv2.INTER_AREA)
            img = img[anchor[0]:anchor[1], anchor[2]:anchor[3]]
            img = img[int(border[0]):int(border[1]), int(border[2]):int(border[3])]

            if any(np.array(config['crop_border']) > 0.0):
                H_c, W_c, _ = img.shape
                o_t = int(H_c * config['crop_border'][0])
                o_l = int(W_c * config['crop_border'][1])
                o_b = int(H_c * config['crop_border'][2])
                o_r = int(W_c * config['crop_border'][3])
                img = img[o_t:H_c-o_b, o_l:W_c-o_r]
                img = cv2.resize(img, (W_c, H_c), interpolation=cv2.INTER_CUBIC)

            """
            img = cv2.resize(img, (int(img.shape[1] / init_factor), int(img.shape[0] / init_factor)), interpolation=cv2.INTER_CUBIC)
            img = img[anchor[0]:anchor[1], anchor[2]:anchor[3]]
            img = img[int(border[0]):int(border[1]), int(border[2]):int(border[3])]

            if config['crop_border'] is True:
                top, buttom, left, right = find_largest_rect(img, bg_color=(128, 128, 128))
                tops.append(top); buttoms.append(buttom); lefts.append(left); rights.append(right)
            """
            stereos.append(img[..., :3])
            normal_canvas.translate(-rel_pose[:3,3])
            normal_canvas.rotate(axis=axis, angle=-(angle*180)/np.pi)
            normal_canvas.view_changed()
        """
        if config['crop_border'] is True:
            atop, abuttom = min(max(tops), img.shape[0]//2 - 10), max(min(buttoms), img.shape[0]//2 + 10)
            aleft, aright = min(max(lefts), img.shape[1]//2 - 10), max(min(rights), img.shape[1]//2 + 10)
            atop -= atop % 2; abuttom -= abuttom % 2; aleft -= aleft % 2; aright -= aright % 2
        else:
            atop = 0; abuttom = img.shape[0] - img.shape[0] % 2; aleft = 0; aright = img.shape[1] - img.shape[1] % 2
        """
        atop = 0; abuttom = img.shape[0] - img.shape[0] % 2; aleft = 0; aright = img.shape[1] - img.shape[1] % 2
        crop_stereos = []
        for stereo in stereos:
            crop_stereos.append((stereo[atop:abuttom, aleft:aright, :3] * 1).astype(np.uint8))
            stereos = crop_stereos
        clip = ImageSequenceClip(stereos, fps=config['fps'])
        if isinstance(video_basename, list):
            video_basename = video_basename[0]
        clip.write_videofile(os.path.join(output_dir, video_basename + '_' + video_traj_type + '.mp4'), fps=config['fps'])



    return normal_canvas, all_canvas
