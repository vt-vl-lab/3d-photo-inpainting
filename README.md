# [CVPR 2020] 3D Photography using Context-aware Layered Depth Inpainting

**This is a modified version for Windows use. All credits still belong to the original researchers.**

[![Open 3DPhotoInpainting in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1706ToQrkIZshRSJSHvZ1RuCiM__YX3Bz)

### [[Paper](https://arxiv.org/abs/2004.04727)] [[Project Website](https://shihmengli.github.io/3D-Photo-Inpainting/)] [[Google Colab](https://colab.research.google.com/drive/1706ToQrkIZshRSJSHvZ1RuCiM__YX3Bz)]

<p align='center'>
<img src='https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/3DPhoto_teaser.jpg' width='900'/>
</p>

We propose a method for converting a single RGB-D input image into a 3D photo, i.e., a multi-layer representation for novel view synthesis that contains hallucinated color and depth structures in regions occluded in the original view. We use a Layered Depth Image with explicit pixel connectivity as underlying representation, and present a learning-based inpainting model that iteratively synthesizes new local color-and-depth content into the occluded region in a spatial context-aware manner. The resulting 3D photos can be efficiently rendered with motion parallax using standard graphics engines. We validate the effectiveness of our method on a wide range of challenging everyday scenes and show fewer artifacts when compared with the state-of-the-arts.
<br/>

**3D Photography using Context-aware Layered Depth Inpainting**
<br/>
[Meng-Li Shih](https://shihmengli.github.io/), 
[Shih-Yang Su](https://lemonatsu.github.io/), 
[Johannes Kopf](https://johanneskopf.de/), and
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/)
<br/>
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.


## Prerequisites
- [Anaconda3](https://www.anaconda.com/products/individual)

You can follow a video tutorial [here](https://youtu.be/wyYK82C6W88) if it's more convenient for you.

- To get started, please run the following steps:
- Clone this [page](https://github.com/bycloudai/3d-photo-inpainting-Windows) 
- Clone this [page](https://github.com/compphoto/BoostingMonocularDepth.git)

    Unzip, drag and drop it here, follow the below folder structure(change the file names too):
    ```
    📂3d-photo-inpainting/
    ├── 📂BoostingMonocularDepth/ <--
    │...
    ```
Change your Anaconda directory to that folder with `cd YOUR_FILE_DIRECTORY/3d-photo-inpainting/`

- Next, follow these installation steps under Anaconda Prompt:
    ```bash
    conda create -n 3DP python=3.7
    conda activate 3DP
    ```
    For any other GPU that's < RTX 30 series:
    ```bash
    conda install pytorch==1.5.0 torchvision==0.6.0 cudatoolkit=10.2 -c pytorch
    ```
    For any RTX 30 series:
    ```bash
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c conda-forge
    ```    
    Continue:
    ```bash
    pip install decorator
    pip install -r requirements.txt
    pip install pyyaml
    pip install cython
    pip install pyqt5
    ```
    If you cannot install Cython, cynetworkx successfully, please install VS C++ Toolkit. Tutorial [here](https://www.notion.so/How-To-Install-Microsoft-C-Build-Tools-f79ca6796a524674878b80e998c88b02).
    
- Next, please download the following models & weights and put them in the right file directory (from my backup [drive](https://drive.google.com/drive/folders/1oiH0qN1yuogDe_zCcaAhECngZtkVi7hz?usp=sharing)):
    - color-model.pth
    - depth-model.pth
    - edge-model.pth
    - model.pt
    - model-f46da743.pt
    - latest_net_G.pth


    ```
    📂3d-photo-inpainting/
    ├── 📂checkpoints/
    │   ├── 📜color-model.pth
    │   ├── 📜depth-model.pth
    │   └── 📜edge-model.pth
    ├── 📂MiDaS/
    │   └── 📜model.pt
    ├── 📂BoostingMonocularDepth/
    │   ├── 📂midas/
    │   │   └── 📜model-f46da743.pt (rename to model.pt after dragged in)
    │   └── 📂pix2pix/
    │       └── 📂checkpoints/
    │           └── 📂mergemodel/
    │               └── 📜latest_net_G.pth
    │...
    ```
    

## Quick start
Please follow the instructions in this section. 
This should allow to execute our results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Execute
1. Put ```.jpg``` files (e.g., test.jpg) into the ```image``` folder.  Don't use any space in the image name (eg. "image 1.jpg" would not work)
    - E.g., `image/moon.jpg`
2. Run the following command
    ```bash
    python main.py --config argument.yml
    ```
    - Note: The 3D photo generation process usually takes about 2-3 minutes depending on the available computing resources.
3. The results are stored in the following directories:
    - Corresponding depth map estimated by [MiDaS](https://github.com/intel-isl/MiDaS.git) 
        - E.g. ```depth/moon.npy```, ```depth/moon.png```
        - User could edit ```depth/moon.png``` manually. 
            - Remember to set the following two flags as listed below if user wants to use manually edited ```depth/moon.png``` as input for 3D Photo.
                - `depth_format: '.png'`
                - `require_midas: False`
    - Inpainted 3D mesh (Optional: User need to switch on the flag `save_ply`)
        - E.g. ```mesh/moon.ply```
    - Rendered videos with zoom-in motion
        - E.g. ```video/moon_zoom-in.mp4```
    - Rendered videos with swing motion
        - E.g. ```video/moon_swing.mp4```
    - Rendered videos with circle motion
        - E.g. ```video/moon_circle.mp4```         
    - Rendered videos with dolly zoom-in effect
        - E.g. ```video/moon_dolly-zoom-in.mp4```
        - Note: We assume that the object of focus is located at the center of the image.
4. (Optional) If you want to change the default configuration. Please read [`DOCUMENTATION.md`](DOCUMENTATION.md) and modified ```argument.yml```.


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

If you find our code/models useful, please consider citing our paper:
```
@inproceedings{Shih3DP20,
  author = {Shih, Meng-Li and Su, Shih-Yang and Kopf, Johannes and Huang, Jia-Bin},
  title = {3D Photography using Context-aware Layered Depth Inpainting},
  booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2020}
}
```

## Acknowledgments
- We thank Pratul Srinivasan for providing clarification of the method [Srinivasan et al. CVPR 2019](https://people.eecs.berkeley.edu/~pratul/publication/mpi_extrapolation/).
- We thank the author of [Zhou et al. 2018](https://people.eecs.berkeley.edu/~tinghuiz/projects/mpi/), [Choi et al. 2019](https://github.com/NVlabs/extreme-view-synth/), [Mildenhall et al. 2019](https://github.com/Fyusion/LLFF), [Srinivasan et al. 2019](https://github.com/google-research/google-research/tree/ac9b04e1dbdac468fda53e798a326fe9124e49fe/mpi_extrapolation), [Wiles et al. 2020](http://www.robots.ox.ac.uk/~ow/synsin.html), [Niklaus et al. 2019](https://github.com/sniklaus/3d-ken-burns) for providing their implementations online.
- Our code builds upon [EdgeConnect](https://github.com/knazeri/edge-connect), [MiDaS](https://github.com/intel-isl/MiDaS.git) and [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)
