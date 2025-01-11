
```python
# 創建新的Conda環境 
conda create -n SD_llama_env1 python=3.10 -y

# 激活環境 
conda activate SD_llama_env1

# 安裝PyTorch和torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 克隆並安裝SAM2 
git clone https://github.com/facebookresearch/sam2.git 
cd sam2 
pip install -e . 

# 安裝stable diffusion
pip install sympy==1.13.1
pip install transformers diffusers

# 安裝其他依賴 
conda install numpy=1.26 -c conda-forge 
conda install scipy=1.11 -c conda-forge 
conda install matplotlib=3.8 -c conda-forge 
conda install pillow opencv jupyter -c conda-forge
pip install --upgrade accelerate

pip install controlnet-aux
pip install basicsr
pip install realesrgan


```