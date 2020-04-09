import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob

from collections import namedtuple
from operator import mul
from functools import reduce
import shutil

def max_size(mat, value=0):
    if not (mat and mat[0]): return (0, 0)
    it = iter(mat)
    prev = [(el==value) for el in next(it)]
    max_size = max_rectangle_size(prev)
    for row in it:
        hist = [(1+h) if el == value else 0 for h, el in zip(prev, row)]
        max_size = max(max_size, max_rectangle_size(hist), key=get_area)
        prev = hist                                               
    return max_size

def max_rectangle_size(histogram):
    Info = namedtuple('Info', 'start height')
    stack = []
    top = lambda: stack[-1]
    max_size = (0, 0) # height, width of the largest rectangle
    pos = 0 # current position in the histogram
    for pos, height in enumerate(histogram):
        start = pos # position where rectangle starts
        while True:
            if not stack or height > top().height:
                stack.append(Info(start, height)) # push
            if stack and height < top().height:
                max_size = max(max_size, (top().height, (pos-top().start)),
                               key=get_area)
                start, _ = stack.pop()
                continue
            break # height == top().height goes here
                
    pos += 1
    for start, height in stack:
        max_size = max(max_size, (height, (pos-start)),
                       key=get_area)

    return max_size

def get_area(size):
    return reduce(mul, size)

def find_anchors(matrix):
    matrix = [[*x] for x in matrix]
    mh, mw = max_size(matrix)
    matrix = np.array(matrix)
    # element = np.zeros((mh, mw))
    for i in range(matrix.shape[0] + 1 - mh):
        for j in range(matrix.shape[1] + 1 - mw):
            if matrix[i:i + mh, j:j + mw].max() == 0:
                return i, i + mh, j, j + mw

src_our_dir = 'DPS/dst_gray_bk2_conn'
src_baseline_dir = 'DPS/baseline_result'
src_tgt_dir = 'DPS/tgt'
filenames = sorted(glob.glob(os.path.join(src_our_dir, '*_pred.png')))
all_occ_filenames = sorted(glob.glob(os.path.join(src_our_dir, '*_occ.png')))
edge_occ_filenames = sorted(glob.glob(os.path.join(src_our_dir, '*_occ.png')))
assert len(filenames) == len(all_occ_filenames) == len(edge_occ_filenames), "Length of filenames is not consistent."

dst_dir = 'DPS/our_result_crop' # 'DPS/test_our_result_crop'
if os.path.exists(dst_dir):
    shutil.rmtree(dst_dir)
os.makedirs(dst_dir)
baseline_dir = 'DPS/baseline_result_crop' # 'DPS/test_baseline1_result_crop'
if os.path.exists(baseline_dir):
    shutil.rmtree(baseline_dir)
os.makedirs(baseline_dir)
tgt_dir = 'DPS/tgt_crop' # 'DPS/test_tgt_crop'
if os.path.exists(tgt_dir):
    shutil.rmtree(tgt_dir)
os.makedirs(tgt_dir)
tgt_shape = (720, 405)
for filename, all_occ_filename, edge_occ_filename in zip(filenames, all_occ_filenames, edge_occ_filenames):
    filename = os.path.basename(filename)
    all_occ_filename = os.path.basename(all_occ_filename)
    edge_occ_filename = os.path.basename(edge_occ_filename)
    dst_img = cv2.imread(os.path.join(src_our_dir, filename))
    # dst_img = cv2.resize(dst_img, (720, 400), interpolation=cv2.INTER_NEAREST)
    dst_img = cv2.resize(dst_img, tgt_shape, interpolation=cv2.INTER_NEAREST)
    valid = np.any(dst_img != (128, 128, 128), axis=-1)
    ret, labels = cv2.connectedComponents(np.uint8(valid == False))
    assert(labels.shape == (tgt_shape[1], tgt_shape[0]))
    red_mat = np.zeros_like(labels)
    # denoise
    for i in range(1, np.max(labels)+1, 1):
        x, y, w, h = cv2.boundingRect(np.uint8(labels==i))
        if x == 0 or (x+w) == tgt_shape[0] or y == 0 or (y+h) == tgt_shape[1]:
            red_mat[labels==i] = 1
    # crop
    t, b, l, r = find_anchors(red_mat)
    try:
        dst_img_gray = cv2.imread(os.path.join(src_our_dir, filename))
    except:
        dst_img_gray = cv2.imread(os.path.join(src_our_dir, filename.replace("_pred", "")))
    dst_img_gray = cv2.resize(dst_img_gray, (720, 400), interpolation=cv2.INTER_AREA)
    dst_img_gray = cv2.resize(dst_img_gray, tgt_shape, interpolation=cv2.INTER_AREA)
    # import pdb; pdb.set_trace()
    try:
        baseline_img = cv2.imread(os.path.join(src_baseline_dir, filename))
    finally:
        baseline_img = cv2.imread(os.path.join(src_baseline_dir, filename.replace("_pred", "")))
    baseline_img = cv2.resize(baseline_img, tgt_shape, interpolation=cv2.INTER_AREA)
    # dst_img_gray = cv2.resize(dst_img_gray, (720, 400), interpolation=cv2.INTER_AREA)
    # dst_img_gray = cv2.resize(dst_img_gray, (720, 405), interpolation=cv2.INTER_CUBIC)
    dst_all_occ_gray = cv2.imread(os.path.join(src_our_dir, all_occ_filename))
    dst_all_occ_gray = cv2.resize(dst_all_occ_gray, tgt_shape, interpolation=cv2.INTER_NEAREST)
    dst_edge_occ_gray = cv2.imread(os.path.join(src_our_dir, edge_occ_filename))
    dst_edge_occ_gray = cv2.resize(dst_edge_occ_gray, tgt_shape, interpolation=cv2.INTER_NEAREST)
    # dst_img_crop = dst_img[t:b, l:r]
    dst_gray_crop = dst_img_gray[t:b, l:r]
    dst_all_occ_crop = dst_all_occ_gray[t:b, l:r]
    dst_edge_occ_crop = dst_edge_occ_gray[t:b, l:r]
    # baseline_img = cv2.resize(baseline_img, tgt_shape, interpolation=cv2.INTER_AREA)
    baseline_img_crop = baseline_img[t:b, l:r]
    tgt_img = cv2.imread(os.path.join(src_tgt_dir, filename.replace("_pred", "")))
    tgt_img = cv2.resize(tgt_img, (720, 405), interpolation=cv2.INTER_AREA)
    tgt_img_crop = tgt_img[t:b, l:r]
    # cv2.imwrite(os.path.join("dst_crop", filename.replace("_pred_00", "")), dst_img_crop)
    cv2.imwrite(os.path.join(baseline_dir, filename.replace("_pred", "")), baseline_img_crop)
    cv2.imwrite(os.path.join(tgt_dir, filename.replace("_pred", "")), tgt_img_crop)
    cv2.imwrite(os.path.join(dst_dir, filename.replace("_pred", "")), dst_gray_crop)
    cv2.imwrite(os.path.join(dst_dir, all_occ_filename), dst_all_occ_crop)
    cv2.imwrite(os.path.join(dst_dir, edge_occ_filename), dst_edge_occ_crop)
