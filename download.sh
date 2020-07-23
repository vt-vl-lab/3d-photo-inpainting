#!/bin/sh
fb_status=$(wget --spider -S https://filebox.ece.vt.edu/ 2>&1 | grep  "HTTP/1.1 200 OK")

mkdir checkpoints
if [ -z "$fb_status" ]
then
    echo "filebox is down. Use alternative urls."
    wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/color-model.pth
    wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/depth-model.pth
    wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/edge-model.pth
    wget https://github.com/LemonATsu/3d-photo-model-weights/raw/master/model.pt
else
    echo "downloading from filebox ..."
    wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/color-model.pth
    wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/depth-model.pth
    wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/edge-model.pth
    wget https://filebox.ece.vt.edu/~jbhuang/project/3DPhoto/model/model.pt
fi

mv color-model.pth checkpoints/.
mv depth-model.pth checkpoints/.
mv edge-model.pth checkpoints/.
mv model.pt MiDaS/.
