
# 一、Studio Lab 快速認識（重點）

- **介面**：標準 JupyterLab。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html?utm_source=chatgpt.com)
    
- **環境**：以 **conda** 管理，**可長期保存**（你的專案磁碟會保留下次重開仍在）。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-manage.html?utm_source=chatgpt.com)
    
- **時數**：**GPU 最長 4 小時/次、每日上限 4 小時**；**CPU 最長 4 小時/次、每日上限 8 小時**（官方說明可能隨時間調整）。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-overview.html?utm_source=chatgpt.com)
    
- **與 AWS 帳號的關係**：Studio Lab 本身是免費服務；若要存取 **S3**，你可以在 Studio Lab 內 **設定 AWS 憑證** 連回你自己的 AWS 帳號資源。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html?utm_source=chatgpt.com)[AWS in Plain English](https://aws.plainenglish.io/how-to-connect-amazon-sagemaker-studio-lab-with-s3-128b0804ee6d?utm_source=chatgpt.com)[GitHub](https://github.com/aws/studio-lab-examples/blob/main/connect-to-aws/Access_AWS_from_Studio_Lab.ipynb?utm_source=chatgpt.com)
    

---

# 二、建立專案與基礎環境（一次性）

1. **登入 Studio Lab**  
    到 `https://studiolab.sagemaker.aws/` 以你的帳號登入，進入 Project 概覽後 **Open project** 進入 JupyterLab。[AWS 文檔+1](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html?utm_source=chatgpt.com)
    
2. **建立持久化 Conda 環境＋Kernel**（建議做；以後不必每次重裝）  
    在 JupyterLab 裡開一個 Terminal，執行：
    

`# 建環境（名稱可自取，例如 mlgpu） conda create -y -n mlgpu python=3.10 conda activate mlgpu  # 安裝常用套件（視你模型需要調整） pip install -U pip wheel pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 pip install transformers accelerate safetensors sentencepiece pip install opencv-python pillow numpy matplotlib # 視需要：pip install timm bitsandbytes xformers #（Qwen2-VL 省顯存時常用） # SAM 原生實作 pip install git+https://github.com/facebookresearch/segment-anything.git  # Jupyter kernel 註冊（之後在右上角就能選到） python -m ipykernel install --user --name "mlgpu" --display-name "Python (mlgpu)"`

Studio Lab 使用 conda 管理環境並會保存到你的專案儲存，重啟後依然在。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-manage.html?utm_source=chatgpt.com)

3. **（可選）把安裝動作寫成一個安裝筆記本**  
    之後新專案或新成員只要「Run All」就能一鍵裝好。
    

---

# 三、在 Studio Lab 內設定 S3（連你的 AWS 帳號）

> 目標：讓你能把 **GitHub 下載的 library**、**模型 checkpoints**、**datasets** 放到 **S3**（長期保存、跨環境共享），也能從 S3 抓回到 Studio Lab。

## A. 在 AWS 端準備（一次性）

1. **建立一個專用 S3 bucket**（例如 `s3://george-ml-artifacts`）。
    
2. **建立精簡權限的 IAM 使用者**（只允許存取上述 bucket）：
    
    - 產生 **Access key ID / Secret access key**（只會出現一次，請保存）。
        
    - 掛一個最小權限的自訂政策（bucket ARN 換成你的）：
        
    
    `{   "Version": "2012-10-17",   "Statement": [{     "Effect": "Allow",     "Action": ["s3:PutObject","s3:GetObject","s3:ListBucket","s3:DeleteObject"],     "Resource": [       "arn:aws:s3:::george-ml-artifacts",       "arn:aws:s3:::george-ml-artifacts/*"     ]   }] }`
    

## B. 在 Studio Lab 內設定憑證

方法 1（最簡單）：在 Terminal 內安裝 CLI 後 `aws configure`

`pip install awscli boto3 s3fs aws configure # 依序輸入：Access key、Secret key、Region（例如 us-east-1）、Output（json）`

方法 2（手動建立檔案）：建立 `~/.aws/credentials` 與 `~/.aws/config`：

`mkdir -p ~/.aws printf "[default]\naws_access_key_id=AKIA...\naws_secret_access_key=...\n" > ~/.aws/credentials printf "[default]\nregion=us-east-1\noutput=json\n" > ~/.aws/config`

這是官方與社群皆常用的做法，用後就能在筆記本內透過 `boto3` / `aws s3` / `s3fs` 操作 S3。[AWS in Plain English](https://aws.plainenglish.io/how-to-connect-amazon-sagemaker-studio-lab-with-s3-128b0804ee6d?utm_source=chatgpt.com)[GitHub](https://github.com/aws/studio-lab-examples/blob/main/connect-to-aws/Access_AWS_from_Studio_Lab.ipynb?utm_source=chatgpt.com)

## C. 常用 S3 操作（Studio Lab 端）

`# 上傳本機資料夾到 S3 aws s3 sync /home/studio-lab-user/projects/myproj s3://george-ml-artifacts/myproj  # 從 S3 拉回（例如 checkpoints） aws s3 cp s3://george-ml-artifacts/checkpoints/sam_vit_h.pth ./checkpoints/ --no-progress`

Python 內直接讀 S3（不必先下載）：

`import pandas as pd # 需 pip install s3fs df = pd.read_csv("s3://george-ml-artifacts/datasets/labels.csv")`

> 註：Studio Lab 是免費服務，但 **S3 是你 AWS 帳號下的資源，照 S3 標準計費**（常見為標準儲存 $0.023/GB-月，傳輸/請求額外計費）。Studio Lab 本身與 Jupyter、環境保存等請見官方文件。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab.html?utm_source=chatgpt.com)

---

# 四、三個模型的最小推論腳本（在 Studio Lab）

> 先確認 GPU：

`import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))`

## A. SAM（Segment Anything）單張影像推論（示意）

`import torch, cv2 from segment_anything import sam_model_registry, SamPredictor  # 若權重放在 S3：先用 aws s3 cp 拉到 ./checkpoints sam_ckpt = "./checkpoints/sam_vit_h.pth"  # or vit_l / vit_b sam = sam_model_registry["vit_h"](checkpoint=sam_ckpt).to("cuda").eval() predictor = SamPredictor(sam)  img = cv2.imread("test.jpg")[:, :, ::-1] predictor.set_image(img)  # 以點提示為例 import numpy as np input_points = np.array([[400, 250]]) input_labels = np.array([1]) masks, scores, _ = predictor.predict(point_coords=input_points,                                      point_labels=input_labels,                                      multimask_output=True) print([m.shape for m in masks], scores)`

## B. Qwen2-VL（小尺寸權重示意；建議 FP16 / 低精度）

`import torch from transformers import AutoProcessor, AutoModelForVision2Seq  model_id = "Qwen/Qwen2-VL-2B-Instruct"   # 視你的模型大小調整 dtype = torch.float16 if torch.cuda.is_available() else torch.float32  processor = AutoProcessor.from_pretrained(model_id) model = AutoModelForVision2Seq.from_pretrained(     model_id, torch_dtype=dtype, device_map="auto" )  from PIL import Image image = Image.open("frame.jpg") prompt = "Describe the scene." inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device) out = model.generate(**inputs, max_new_tokens=64) print(processor.batch_decode(out, skip_special_tokens=True)[0])`

> 若顯存吃緊，可安裝 `bitsandbytes` 並用 8-bit 量化（`load_in_8bit=True`），或改用更小的 0.5B / 1.5B 權重。

## C. DINOv3（特徵抽取／分類示意）

> DINOv3 有多個實作路徑（原始 GitHub、Hugging Face 權重、或 `timm` 近似 backbone）。以下示意用 HF `AutoModel` / `AutoImageProcessor`，若你的特定模型卡在處理器不支援，就改用該 repo 的前處理或 `timm`。

`import torch from PIL import Image from transformers import AutoImageProcessor, AutoModel  model_id = "facebook/dinov3-large"  # 依你實際用的權重調整 processor = AutoImageProcessor.from_pretrained(model_id) model = AutoModel.from_pretrained(model_id).to("cuda").eval()  image = Image.open("test.jpg").convert("RGB") inputs = processor(images=image, return_tensors="pt").to("cuda") with torch.no_grad():     feats = model(**inputs).last_hidden_state  # 取特徵 print(tuple(feats.shape))`

---

# 五、把「Colab 程式」換到 Studio Lab 的常見修改

|目的|Colab 寫法|Studio Lab 寫法|
|---|---|---|
|掛雲端硬碟|`from google.colab import drive; drive.mount('/content/drive')`|用 **S3**：`aws s3 sync/cp` 或 `boto3` / `s3fs`（見上節）|
|安裝套件|每次 `!pip install ...`|**一次性安裝到 conda 環境**，之後直接用；或保留安裝 cell 但之後基本不需重跑|
|路徑|`/content/...`|你的專案路徑（例 `/home/studio-lab-user/projects/...`）或相對路徑|
|顯示影像|`from google.colab.patches import cv2_imshow`|直接用 `matplotlib` 或 `IPython.display`|
|上傳檔案|Colab 的「檔案上傳」工具|`aws s3 cp`/`sync`、JupyterLab 檔案面板拖拉、或 `wget/curl`|
|秘鑰/Token|`from google.colab import userdata`|放在 `~/.aws/credentials`、環境變數、或 `.env`（**不要**硬編在 notebook）|

> 小提醒：Studio Lab **一次下載的 checkpoints / datasets 會留在你的專案磁碟**，下次打開不用重下；只要你沒手動刪除即可。[AWS 文檔](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-manage.html?utm_source=chatgpt.com)

---

# 六、常見排錯與最佳實務

- **CUDA/顯存不足**：
    
    - Qwen2-VL 請選小模型或用 `load_in_8bit=True`；SAM 請縮小輸入影像或用 `vit_b/l` 權重；DINOv3 批次別開太大。
        
- **AWS 認證錯誤**：
    
    - 確認 `~/.aws/credentials`／`aws configure` 已設定且 **region** 有填；`boto3.client('s3')` 能列出 bucket。[Stack Overflow](https://stackoverflow.com/questions/70378085/sagemaker-studio-lab-how-do-i-configure-aws-for-running-in-sagemaker-studio-lab?utm_source=chatgpt.com)
        
- **速度／持久化**：
    
    - 模型與資料放 S3；Studio Lab 拉下來到本地再推論，或直接用 `s3fs` 流式讀取。
        
- **安全**：
    
    - 以 **最小權限 IAM** 管理 S3；不要把 Access Key 寫死在 notebook。
        

---

如果你願意，我可以幫你把：

1. **專案資料夾結構**（`src/`, `checkpoints/`, `datasets/`, `notebooks/`），
    
2. **conda 環境 .yml**（針對 SAM / Qwen2-VL / DINOv3），
    
3. **S3 同步腳本**（`sync_up.sh` / `sync_down.sh`）  
    一次生出來，直接拷貝到 Studio Lab 使用。需要就說一聲 🙌
    

（參考：Studio Lab 環境管理與保存、GPU/CPU 使用限制與總覽、在 Studio Lab 內連到 AWS/S3 的做法與範例。）[AWS 文檔+2AWS 文檔+2](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-lab-use-manage.html?utm_source=chatgpt.com)[AWS in Plain English](https://aws.plainenglish.io/how-to-connect-amazon-sagemaker-studio-lab-with-s3-128b0804ee6d?utm_source=chatgpt.com)[GitHub](https://github.com/aws/studio-lab-examples/blob/main/connect-to-aws/Access_AWS_from_Studio_Lab.ipynb?utm_source=chatgpt.com)