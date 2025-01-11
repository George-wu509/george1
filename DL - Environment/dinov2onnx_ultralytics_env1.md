
$ conda create -n dinov2onnx_ultralytics_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install --trusted-host pypi.nvidia.com --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) cuml-cu11

$ pip install ultralytics

$ pip install omegaconf torchmetrics==0.10.3 fvcore iopath submitit

$ pip install -U xformers==0.0.18

$ conda install anaconda::pillow

$ sudo apt update

$ sudo apt install g++

$ git clone [https://github.com/sefaburakokcu/dinov2.git](https://github.com/sefaburakokcu/dinov2.git)

之後進入dinov2 folder, 執行pip install .

$ pip install GPUtil

$ conda install scikit-learn -c conda-forge

$ pip install -U openmim

$ pip install onnxruntime==1.16.3

$ pip install opencv-python==4.8.0.76

$ pip install timm effdet

$ mim install mmengine --trusted-host download.openmmlab.com

$ pip install mmcv==2.2.0 -f [https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html](https://download.openmmlab.com/mmcv/dist/cu117/torch2.0/index.html) --trusted-host download.openmmlab.com

$ pip install mmsegmentation

$ mim install mmdet  --trusted-host download.openmmlab.com