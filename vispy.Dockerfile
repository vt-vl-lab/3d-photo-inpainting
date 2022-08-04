FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-devel
RUN apt-get update
RUN apt install -y libfontconfig1-dev wget ffmpeg libsm6 libxext6 libxrender-dev mesa-utils-extra libegl1-mesa-dev libgles2-mesa-dev xvfb
ENV DISPLAY=:0
RUN pip install opencv-python==4.2.0.32 vispy==0.6.4 moviepy==1.0.2 transforms3d==0.3.1 networkx==2.3  scikit-image

RUN pip install scipy matplotlib scikit-image
RUN pip install jupyterlab
ENV MKL_SERVICE_FORCE_INTEL=true
RUN pip install pydantic
RUN pip install -U ipywidgets