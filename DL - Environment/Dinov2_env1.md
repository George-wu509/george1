
Run RINOv2 inference, local training, GPU usage, mmdetection

$ conda create -n dinov2_env1 python=3.9

$ conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia

$ pip install --trusted-host pypi.nvidia.com --extra-index-url [https://pypi.nvidia.com](https://pypi.nvidia.com) cuml-cu11

$ pip install omegaconf torchmetrics==0.10.3 fvcore iopath submitit

$ pip install -U xformers==0.0.18

$ sudo apt update

$ sudo apt install g++

$ pip install mmcv-full==1.5.0

$ pip install mmsegmentation==0.27.0

$ conda install anaconda::pillow

$ git clone [https://github.com/facebookresearch/dinov2.git](https://github.com/facebookresearch/dinov2.git)

之後進入dinov2 folder, 執行pip install .

$ pip install GPUtil

$ conda install scikit-learn -c conda-forge

$ pip install -U openmim

$ mim install mmengine --trusted-host download.openmmlab.com

$ git clone --branch v2.28.2 --single-branch [https://github.com/open-mmlab/mmdetection.git](https://github.com/open-mmlab/mmdetection.git)

之後進入mmdetection folder, 執行pip install .

$ pip install onnxruntime==1.16.3

$ pip install opencv-python==4.8.0.76

@ 如果安裝mmcv-full顯示問題可能是沒安裝CUDA Toolkit 11.7

$ wget --no-check-certificate [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin)

$ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600

$ wget --no-check-certificate [https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb](https://developer.download.nvidia.com/compute/cuda/11.7.0/local_installers/cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb)

$ sudo dpkg -i cuda-repo-wsl-ubuntu-11-7-local_11.7.0-1_amd64.deb

$ sudo cp /var/cuda-repo-wsl-ubuntu-11-7-local/cuda-*-keyring.gpg /usr/share/keyrings/

$ sudo apt-get update

$ sudo apt-get -y install cuda

@如果安裝CUDA toolkit有apt-key問題

$ wget --no-check-certificate -c [https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub](https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/3bf863cc.pub) -O ./nvidia-cuda.pub

$ cat ./nvidia-cuda.pub | sudo gpg --import --no-default-keyring --keyring gnupg-ring:/etc/apt/trusted.gpg.d/nvidia-cuda.gpg

$ sudo chmod a+r /etc/apt/trusted.gpg.d/nvidia-cuda.gpg

Ref: [https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local](https://developer.nvidia.com/cuda-11-7-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)

@如果nvvv -V顯示not found可能是要設定path

$ nano ~/.bashrc

在末行加上

export PATH=/usr/local/cuda-11.7/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH

儲存