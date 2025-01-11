
@@check we have g++ and cuda 11.7 installed

$ conda create -n dinov2_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install --trusted-host pypi.nvidia.com --extra-index-url https://pypi.nvidia.com cuml-cu11

$ pip install omegaconf torchmetrics==0.10.3 fvcore iopath submitit

$ pip install -U xformers==0.0.18

$ pip install mmcv-full==1.5.0

$ pip install mmsegmentation==0.27.0

$ git clone [https://github.com/facebookresearch/dinov2.git](https://github.com/facebookresearch/dinov2.git)

之後進入dinov2 folder, 執行pip install .

>>> add AzureML libraries)

$ pip install azureml.core

>>>> add GPU memory measure

$ pip install GPUtil

>>>> add sklearn

$ conda install scikit-learn -c conda-forge

>>>> add mmdetection

$ pip install -U openmim

$ mim install mmengine

$ mim install "mmcv>=2.0.0"      (如果沒有安裝mmfull/mmcv)

$ git clone [https://github.com/open-mmlab/mmdetection.git](https://github.com/open-mmlab/mmdetection.git)

$ cd mmdetection

$ pip install -v -e .

$ mim download mmdet --ignore-ssl --config mask-rcnn_r50-caffe_fpn_ms-poly-3x_coco --dest ./checkpoints

@@ 如果mmcv-full install出現問題

@如果顯示沒有g++

step: install g++

$ sudo apt update

$ sudo apt install g++

@如果顯示cuda mismatch:

step: check cuda安裝正確cuda 11.7

$ wget --no-check-certificate [https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run](https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda_11.7.0_515.43.04_linux.run)

$ sudo sh cuda_11.7.0_515.43.04_linux.run

nano ~/.bashrc

export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}