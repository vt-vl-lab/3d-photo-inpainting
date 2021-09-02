import glob
import os

import cv2
import imageio
import numpy as np
from MiDaS.MiDaS_utils import write_depth

BOOST_BASE = "BoostingMonocularDepth"

BOOST_INPUTS = "inputs"
BOOST_OUTPUTS = "outputs"


def run_boostmonodepth(img_names, src_folder, depth_folder):

    if not isinstance(img_names, list):
        img_names = [img_names]

    # remove irrelevant files first
    clean_folder(os.path.join(BOOST_BASE, BOOST_INPUTS))
    clean_folder(os.path.join(BOOST_BASE, BOOST_OUTPUTS))

    tgt_names = []
    for img_name in img_names:
        base_name = os.path.basename(replace_ext(img_name, ".png"))
        input_file = os.path.join(BOOST_BASE, BOOST_INPUTS, base_name)
        os.system(f"cp {img_name} {input_file}")
        tgt_names.append(base_name)

        # keep only the file name here.
        # they save all depth as .png file

    os.system(
        f"cd {BOOST_BASE} && python run.py --Final --data_dir {BOOST_INPUTS}/  --output_dir {BOOST_OUTPUTS} --depthNet 0"
    )

    for i, (img_name, tgt_name) in enumerate(zip(img_names, tgt_names)):
        img = imageio.imread(img_name)
        H, W = img.shape[:2]
        scale = 640.0 / max(H, W)

        # resize and save depth
        target_height, target_width = int(round(H * scale)), int(round(W * scale))
        depth = imageio.imread(os.path.join(BOOST_BASE, BOOST_OUTPUTS, tgt_name))
        depth = np.array(depth).astype(np.float32)
        depth = resize_depth(depth, target_width, target_height)
        np.save(os.path.join(depth_folder, replace_ext(tgt_name, ".npy")), depth / 32768.0 - 1.0)
        write_depth(os.path.join(depth_folder, replace_ext(tgt_name, "")), depth)


def clean_folder(folder, img_exts=None):
    img_exts = img_exts or [".png", ".jpg", ".npy"]
    for img_ext in img_exts:
        paths_to_check = os.path.join(folder, f"*{img_ext}")
        if len(glob.glob(paths_to_check)) == 0:
            continue
        print(paths_to_check)
        os.system(f"rm {paths_to_check}")


def resize_depth(depth, width, height):
    """Resize numpy (or image read by imageio) depth map

    Args:
        depth (numpy): depth
        width (int): image width
        height (int): image height

    Returns:
        array: processed depth
    """
    depth = cv2.blur(depth, (3, 3))
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_AREA)


def replace_ext(filename: str, extension: str):
    extension = extension.replace(".", "")
    return f"{os.path.splitext(filename)[0]}.{extension}"
