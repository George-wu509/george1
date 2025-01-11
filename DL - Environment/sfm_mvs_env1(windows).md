
```python
# 創建新的Conda環境 
conda create -n sfm_mvs_env1 python=3.10 -y

# 激活環境 
conda activate sfm_mvs_env1

# 安裝PyTorch和torchvision
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia


# 安裝其他依賴 
conda install numpy=1.26 -c conda-forge 
conda install scipy=1.11 -c conda-forge 
conda install matplotlib=3.8 -c conda-forge 
conda install pillow opencv jupyter -c conda-forge
pip install pyvista open3d

```