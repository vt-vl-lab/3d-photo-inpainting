# Documentation

## Python scripts

These files are for our monocular 3D Tracking pipeline:

`main.py` Execute 3D photo inpainting

`mesh.py` Functions about context-aware depth inpainting

`mesh_tools.py` Some common functions used in `mesh.py`

`utils.py` Some common functions used in image preprocessing, data loading

`networks.py` Network architectures of inpainting model


MiDaS/

`run.py` Execute depth estimation

`monodepth_net.py` Network architecture of depth estimation model

`MiDaS_utils.py` Some common functions in depth estimation


## Configuration

```bash
argument.yml
```

- `depth_edge_model_ckpt: checkpoints/EdgeModel.pth`
    - Pretrained model of depth-edge inpainting
- `depth_feat_model_ckpt: checkpoints/DepthModel.pth`
    - Pretrained model of depth inpainting
- `rgb_feat_model_ckpt: checkpoints/ColorModel.pth`
    - Pretrained model of color inpainting
- `MiDaS_model_ckpt: MiDaS/model.pt`
    - Pretrained model of depth estimation
- `fps: 40`
    - Frame per second of output rendered video
- `num_frames: 240`
    - Total number of frames in output rendered video
- `x_shift_range: [-0.03, -0.03, -0.03]`
    - The translations on x-axis of output rendered videos.
    - This parameter is a list. Each element corresponds to a specific camera motion.
- `y_shift_range: [-0.00, -0.00, -0.03]`
    - The translations on y-axis of output rendered videos.
    - This parameter is a list. Each element corresponds to a specific camera motion.
- `z_shift_range: [-0.07, -0.07, -0.07]`
    - The translations on z-axis of output rendered videos.
    - This parameter is a list. Each element corresponds to a specific camera motion.
- `traj_types: ['straight-line', 'circle', 'circle']`
    - The type of camera trajectory.
    - This parameter is a list.
    - Currently, we only privode `straight-line` and `circle`.
-  `video_postfix: ['zoom-in', 'swing', 'circle']`
    - The postfix of video.
    - This parameter is a list.
- Note that the number of elements in `x_shift_range`,  `y_shift_range`, `z_shift_range`, `traj_types` and `video_postfix` should be equal.
- `specific: '' `
    - The specific image name, use this to specify the image to be executed. By default, all the image in the folder will    be executed.
- `longer_side_len: 960`
    - The length of larger dimension in output resolution.
- `src_folder: image`
    - Input image directory. 
- `depth_folder: depth`
    - Estimated depth directory.
- `mesh_folder: mesh`
    - Output 3-D mesh directory.
- `video_folder: video`
    - Output rendered video directory
- `load_ply: False`
    - Action to load existed mesh (.ply) file
- `save_ply: True`
    - Action to store the output mesh (.ply) file
    - Disable this option `save_ply: False` to reduce the computational time.
- `inference_video: True`
    - Action to rendered the output video
- `gpu_ids: 0`
    - The ID of working GPU. Leave it blank or negative to use CPU.
- `offscreen_rendering: True`
    - If you're executing the process in a remote server (via ssh), please switch on this flag. 
    - Sometimes, using off-screen rendering result in longer execution time.
- `img_format: '.jpg'`
    - Input image format.
- `depth_threshold: 0.04`
    - A threshold in disparity, adjacent two pixels are discontinuity pixels 
      if the difference between them excceed this number.
- `ext_edge_threshold: 0.002`
    - The threshold to define inpainted depth edge. A pixel in inpainted edge 
      map belongs to extended depth edge if the value of that pixel exceeds this number,
- `sparse_iter: 5`
    - Total iteration numbers of bilateral median filter
- `filter_size: [7, 7, 5, 5, 5]`
    - Window size of bilateral median filter in each iteration.
- `sigma_s: 4.0`
    - Intensity term of bilateral median filter
- `sigma_r: 0.5`
    - Spatial term of bilateral median filter
- `redundant_number: 12`
    - The number defines short segments. If a depth edge is shorter than this number, 
      it is a short segment and removed.
- `background_thickness: 70`
    - The thickness of synthesis area.
- `context_thickness: 140`
    - The thickness of context area.
- `background_thickness_2: 70`
    - The thickness of synthesis area when inpaint second time.
- `context_thickness_2: 70`
    - The thickness of context area when inpaint second time.
- `discount_factor: 1.00`
- `log_depth: True`
    - The scale of depth inpainting. If true, performing inpainting in log scale. 
      Otherwise, performing in linear scale.
- `largest_size: 512`
    - The largest size of inpainted image patch.
- `depth_edge_dilate: 10`
    - The thickness of dilated synthesis area.
- `depth_edge_dilate_2: 5`
    - The thickness of dilated synthesis area when inpaint second time.
- `extrapolate_border: True`
    - Action to extrapolate out-side the border.
- `extrapolation_thickness: 60`
    - The thickness of extrapolated area.
- `repeat_inpaint_edge: True`
    - Action to apply depth edge inpainting model repeatedly. Sometimes inpainting depth 
      edge once results in short inpinated edge, apply depth edge inpainting repeatedly 
      could help you prolong the inpainted depth edge. 
- `crop_border: [0.03, 0.03, 0.05, 0.03]`
    - The fraction of pixels to crop out around the borders `[top, left, bottom, right]`.
- `anti_flickering: True`
    - Action to avoid flickering effect in the output video. 
    - This may result in longer computational time in rendering phase.
