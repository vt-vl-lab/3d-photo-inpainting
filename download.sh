#!/bin/sh
fb_status=$(wget --spider -S https://filebox.ece.vt.edu/ 2>&1 | grep  "HTTP/1.1 200 OK")

mkdir checkpoints

echo "downloading from filebox ..."
curl -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
curl -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
curl -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
curl -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt

mv color-model.pth checkpoints/.
mv depth-model.pth checkpoints/.
mv edge-model.pth checkpoints/.
mv model.pt MiDaS/.

echo "cloning from BoostingMonocularDepth ..."
git clone https://github.com/compphoto/BoostingMonocularDepth.git
mkdir -p BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/

echo "downloading mergenet weights ..."
curl -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/latest_net_G.pth
mv latest_net_G.pth BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
curl -O https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt
mv model-f46da743.pt BoostingMonocularDepth/midas/model.pt

python patch.py
python config.py
mkdir -p ./KenBurns/Input
mkdir -p ./KenBurns/Output

pip install vispy
pip install moviepy
pip install transforms3d
pip install networkx
pip install opencv-python
pip install pyqt5

python main.py --config argument.yml