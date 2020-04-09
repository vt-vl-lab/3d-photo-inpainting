# [CVPR 2020] 3D Photography using Context-aware Layered Depth Inpainting

We propose a method for converting a single RGB-D input image into a 3D photo, i.e., a multi-layer representation for novel view synthesis that contains hallucinated color and depth structures in regions occluded in the original view. We use a Layered Depth Image with explicit pixel connectivity as underlying representation, and present a learning-based inpainting model that iteratively synthesizes new local color-and-depth content into the occluded region in a spatial context-aware manner. The resulting 3D photos can be efficiently rendered with motion parallax using standard graphics engines. We validate the effectiveness of our method on a wide range of challenging everyday scenes and show less artifacts when compared with the state-of-the-arts.
<br/>

**3D Photography using Context-aware Layered Depth Inpainting**
<br/>
[Meng-Li Shih](https://shihmengli.github.io/), 
[Shih-Yang Su](https://lemonatsu.github.io/), 
[Johannes Kopf](https://johanneskopf.de/), 
[Jia-Bin Huang](https://filebox.ece.vt.edu/~jbhuang/)
<br/>
In CVPR, 2020.

[Paper](https://drive.google.com/file/d/17ki_YAL1k5CaHHP3pIBFWvw-ztF4CCPP/view?usp=sharing)
[Website](https://shihmengli.github.io/3D-Photo-Inpainting/)

## Prerequisites

- Linux (tested on Ubuntu 18.04.4 LTS)
- Anaconda
- Python 3.7 (tested on 3.7.4)
- PyTorch 1.4.0 (tested on 1.4.0 for execution)

and Python dependencies list in [requirements.txt](requirements.txt)
- To prepare the environment, you could run the following commands
    ```bash
    conda create -n 3DP python=3.7 anaconda
    pip install -r requirements.txt
    conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
    ```
- Our method requires learning-based model. Please download them and put them in correct directory.
    ```bash
    wget edge-model.pth checkpoints/
    wget depth-model.pth checkpoints/
    wget color-model.pth checkpoints/
    wget model.pt MiDaS/
    ```    

## Quick start
To get started as quickly as possible, follow the instructions in this section. 
This should allow to execute our results.
For more detailed instructions, please refer to [`DOCUMENTATION.md`](DOCUMENTATION.md).

## Execute
1. Put ```.jpg``` files (e.g. test.jpg) into ```image``` folder. 
    - E.g. `image/moon.jpg`
2. Run the following command
    - Note: The whole process may take 3 minutes or more depends on the computing resource.
    ```bash
    python demo.py --config argument.yml
    ```
3. The outputs file are in the ```output``` folder
    - Corresponding depth map estimated by [MiDaS](https://github.com/intel-isl/MiDaS.git) 
        - E.g. ```depth/moon.npy```
    - Inpainted 3D mesh
        - E.g. ```mesh/moon.ply```
    - Rendered videos with circular motion
        - E.g. ```mesh/moon.mp4```
4. (Optional) If you want to change the default configuration. Please read [`DOCUMENTATION.md`](DOCUMENTATION.md) and modified ```argument.yml```.


## License
This work is licensed under MIT License. See [LICENSE](LICENSE) for details. 

If you use our code/models in your research, please cite our paper:
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
- Part of these code are borrowing from [EdgeConnect](https://github.com/knazeri/edge-connect), [MiDaS](https://github.com/intel-isl/MiDaS.git) and [pytorch-inpainting-with-partial-conv](https://github.com/naoto0804/pytorch-inpainting-with-partial-conv)