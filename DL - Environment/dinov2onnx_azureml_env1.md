
Run RINOv2 inference, local training, GPU usage, latest mmdetection and mmsegmentation

$ conda create -n dinov2onnx_azureml_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install --trusted-host pypi.nvidia.com --extra-index-url https://pypi.nvidia.com cuml-cu11

$ pip install omegaconf torchmetrics==0.10.3 fvcore iopath submitit

$ pip install -U xformers==0.0.18

$ pip install matplotlib

$ sudo apt update

$ sudo apt install g++

$ pip install mmcv-full==1.5.0

$ pip install mmsegmentation==0.27.0

$ git clone https://github.com/sefaburakokcu/dinov2.git

之後進入dinov2 folder, 執行pip install .

$ pip install GPUtil

$ conda install scikit-learn -c conda-forge

$ pip install -U openmim

$ mim install mmengine --trusted-host download.openmmlab.com

$ git clone --branch v2.28.2 --single-branch https://github.com/open-mmlab/mmdetection.git

之後進入mmdetection folder, 執行pip install .

$ pip install onnxruntime==1.16.3

$ pip install opencv-python==4.8.0.76

$ pip install azureml-sdk azureml-core azureml-defaults azureml-telemetry azureml-train-restclients-hyperdrive azureml-train-core tensorboard future

$ pip install numpy==1.26.4