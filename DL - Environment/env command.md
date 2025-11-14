
Remove env
```python
# create python environment
conda create -n ENV_NAME python=3.9

# Remove environment
conda remove ENV_NAME --all

# install library from local github repo
pip install -e .

```



```python
conda env create -f watch_ocr_env.yaml

conda activate watch_ocr_env

pip install -r requirements.txt

conda env remove --name watch_ocr_env


**（如果需要）手動刪除損壞的環境**：

- 執行 `conda info --envs` 找到 `watch-ocr-env` 的資料夾路徑。
- 在 Windows 檔案總管中**手動刪除**該資料夾 (如果 `conda env remove` 命令失敗)。
  
  conda clean --all
```