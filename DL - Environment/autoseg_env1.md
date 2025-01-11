
```python
# 創建新的Conda環境 
conda create -n autoseg_env1 python=3.10 -y

# 激活環境 
conda activate autoseg_env1 

# 安裝PyTorch和torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安裝其他依賴 
conda install numpy=1.26 -c conda-forge 
conda install scipy=1.11 -c conda-forge 
conda install matplotlib=3.8 -c conda-forge 
conda install pillow opencv jupyter -c conda-forge
pip install decord
conda install conda-forge::tqdm
conda install conda-forge::loguru
conda install conda-forge::imageio
pip install hydra-core

# 克隆並安裝SAM
git clone https://github.com/facebookresearch/segment-anything.git
cd segment-anything
pip install -e .

# 克隆並安裝SAM2 
git clone https://github.com/facebookresearch/sam2.git 
cd sam2 
pip install -e .
```