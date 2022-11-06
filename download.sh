#!/bin/sh
fb_status=$(wget --spider -S https://filebox.ece.vt.edu/ 2>&1 | grep  "HTTP/1.1 200 OK")

mkdir checkpoints

echo "downloading from filebox ..."
curl --retry 5 -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
curl --retry 5 -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
curl --retry 5 -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
curl --retry 5 -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt

mv color-model.pth checkpoints/.
mv depth-model.pth checkpoints/.
mv edge-model.pth checkpoints/.
mv model.pt MiDaS/.

echo "cloning from BoostingMonocularDepth ..."
git clone https://github.com/compphoto/BoostingMonocularDepth.git
mkdir -p BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/

echo "downloading mergenet weights ..."
curl --retry 5 -O https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/latest_net_G.pth
mv latest_net_G.pth BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/
curl --retry 5 -O https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt
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
pip install scipy==1.9.2
pip install pyyaml==6.0

file1=checkpoints/color-model.pth
minimumsize1=90000
actualsize1=$(wc -c <"$file1")
if [ $actualsize1 -ge $minimumsize1 ]; then
    echo $file1 ok
else
    echo $file1 size is under $minimumsize1 bytes
    exit
fi

file2=checkpoints/depth-model.pth
minimumsize2=90000
actualsize2=$(wc -c <"$file2")
if [ $actualsize2 -ge $minimumsize2 ]; then
    echo $file2 ok
else
    echo $file2 size is under $minimumsize2 bytes
    exit
fi

file3=checkpoints/edge-model.pth
minimumsize3=90000
actualsize3=$(wc -c <"$file3")
if [ $actualsize3 -ge $minimumsize3 ]; then
    echo $file3 ok
else
    echo $file3 size is under $minimumsize3 bytes
    exit
fi

file4=MiDaS/model.pt
minimumsize4=90000
actualsize4=$(wc -c <"$file4")
if [ $actualsize4 -ge $minimumsize4 ]; then
    echo $file4 ok
else
    echo $file4 size is under $minimumsize4 bytes
    exit
fi

file5=BoostingMonocularDepth/pix2pix/checkpoints/mergemodel/latest_net_G.pth
minimumsize5=90000
actualsize5=$(wc -c <"$file5")
if [ $actualsize5 -ge $minimumsize5 ]; then
    echo $file5 ok
else
    echo $file5 size is under $minimumsize5 bytes
    exit
fi

file6=BoostingMonocularDepth/midas/model.pt
minimumsize6=90000
actualsize6=$(wc -c <"$file6")
if [ $actualsize6 -ge $minimumsize6 ]; then
    echo $file6 ok
else
    echo $file6 size is under $minimumsize6 bytes
    exit
fi

# Can only run this if no error after above setup.
python main.py --config argument.yml
