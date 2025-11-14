
name: watch_ocr_env

channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  # --- 1. Python & Core (from Conda) ---
  - python=3.10
  - pip
  - pyyaml
  - tqdm
  - pytesseract
  - pillow
  - transformers  # 這些套件是純 Python 或依賴 PyTorch
  - timm
  - sentencepiece
  # 'numpy', 'scipy', 'scikit-image' 已被移至 pip 區塊

  # --- 2. PyTorch (from Conda) ---
  - pytorch=2.1
  - torchvision
  - torchaudio
  - pytorch-cuda=11.8 # Or 12.1 if your setup supports it

  # --- 3. Pip-Only Dependencies (包含所有科學計算套件) ---
  - pip:
      # --- 核心科學計算堆疊 (全部來自 Pip) ---
      - numpy
      - scipy
      - scikit-image
      
      # --- 核心 CV 堆疊 (全部來自 Pip) ---
      - opencv-contrib-python-headless
      - easyocr
      - python-doctr
      
      # --- 其他 Libs ---
      - surya-ocr
      
      # --- Paddle (Pinned) ---
      - paddlepaddle-gpu==2.5.2
      - paddleocr==2.6.1.3
      
      # --- MMLab (Pinned) ---
      - mmengine==0.8.4
      - mmcv==2.0.1
      - mmdet==3.0.0
      - mmocr==1.0.1