
|                              |     |
| ---------------------------- | --- |
| [[### MLOPs using AzureML ]] |     |
| [[###### MLOPs using AWS]]   |     |


### MLOPs using AzureML 

詳細說明如何使用 Azure Machine Learning (Azure ML) 實作一個物件偵測 (Object Detection) 模型的 MLOps 流程，並提供具體的中文說明和示例概念。

**MLOps 的核心目標**：是將 DevOps 的原則應用於機器學習系統的開發和部署，以實現模型生命週期的自動化、標準化、可追蹤性和高品質。

**場景設定**

假設我們要為一個零售商店開發一個物件偵測模型，用於識別貨架上的商品，例如：水果（蘋果、香蕉）、飲料（可樂、水瓶）等。

**Azure Machine Learning 在 MLOps 中的核心組件**

1. **工作區 (Workspace):** Azure ML 的頂層資源，集中管理所有機器學習資產。
2. **資料資產 (Data Assets):** 管理和版本化您的數據集。
3. **環境 (Environments):** 定義和管理模型訓練和部署所需的 Python 環境、Docker 映像等。
4. **計算資源 (Compute):**
    - **計算叢集 (Compute Clusters):** 用於模型訓練的可擴展計算資源 (CPU/GPU)。
    - **計算執行個體 (Compute Instances):** 開發用的雲端工作站。
    - **推斷叢集 (Inference Clusters) / 受控線上端點 (Managed Online Endpoints):** 用於模型部署。
5. **作業 (Jobs):** 執行訓練腳本或其他機器學習任務。
6. **模型 (Models):** 註冊、版本化和管理訓練好的模型。
7. **端點 (Endpoints):** 將模型部署為可供應用程式呼叫的 Web 服務。
    - **線上端點 (Online Endpoints):** 用於即時推斷。
    - **批次端點 (Batch Endpoints):** 用於對大量數據進行非即時推斷。
8. **管線 (Pipelines):** 將機器學習工作流程的各個步驟（如數據準備、訓練、註冊、部署）串聯起來，實現自動化。
9. **元件 (Components):** 可重用的管線步驟。

**MLOps 實作步驟詳解 (以物件偵測為例)**

**步驟 1：專案設定與原始碼管理**

- **工具：** Git (例如 Azure Repos, GitHub)
- **說明：**
    - 建立一個 Git 儲存庫來管理所有程式碼（訓練腳本、部署腳本、管線定義檔等）、設定檔以及模型相關的 YAML 文件。
    - 良好的分支策略（如 `main`, `dev`, `feature/*`）有助於協同作業和版本控制。
- **示例結構：**
    
    ```
    object-detection-mlops/
    ├── .azureml/              # Azure ML CLI 設定檔 (可選)
    ├── data/                  # (可選) 本地測試用的小型數據集
    ├── src/                   # 訓練、評估、推斷等 Python 腳本
    │   ├── train.py
    │   ├── evaluate.py
    │   ├── score.py           # 線上端點的推斷腳本
    ├── environments/          # 環境定義 YAML 檔案
    │   └── yolov8_env.yml
    ├── components/            # 自訂管線元件 YAML 檔案 (可選)
    ├── pipelines/             # 管線定義 YAML 檔案
    │   └── training_pipeline.yml
    ├── endpoints/             # 端點和部署定義 YAML 檔案
    │   ├── online_endpoint.yml
    │   └── online_deployment.yml
    ├── tests/                 # 測試腳本
    └── README.md
    ```
    

**步驟 2：數據準備與管理 (Data Preparation & Management)**

- **Azure ML 組件：** Azure Blob Storage, Azure ML 資料資產 (Data Assets)
    
- **說明：**
    
    1. **數據收集與標註：**
        - 收集包含目標物件（水果、飲料）的圖像。
        - 使用標註工具（如 LabelImg, CVAT, 或 Azure ML 內建的數據標註工具）為圖像中的物件加上邊界框 (bounding box) 和類別標籤。常見的標註格式有 Pascal VOC (XML), COCO (JSON), YOLO (TXT)。
    2. **數據上傳至 Azure Blob Storage：** 將原始圖像和標註文件上傳到 Azure Blob Storage 的容器中。建議將圖像和標註分開存放或有清晰的組織結構。
    3. **建立 Azure ML 資料資產：**
        - 在 Azure ML 工作區中，將儲存在 Blob Storage 中的圖像和標註註冊為資料資產。這使得數據可以被版本化、追蹤，並在 Azure ML 作業和管線中輕鬆引用。
        - 對於物件偵測，您可能需要兩個資料資產：一個指向圖像文件，一個指向標註文件；或者，如果使用特定格式（如 COCO JSON），一個資產可能同時包含圖像路徑和標註。Azure ML 也支援 `mltable` 格式，可以更好地組織複雜數據。
- **示例 (Azure ML CLI v2)：**
    
    - 假設圖像在 `azureml://datastores/workspaceblobstore/paths/fridge_items/images/`
    - 假設標註 (COCO JSON 格式) 在 `azureml://datastores/workspaceblobstore/paths/fridge_items/annotations/coco_annotations.json`
    
    建立圖像資料資產的 YAML (`image_data.yml`):
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/asset.schema.json
    name: fridge-item-images
    version: 1 # 版本控制
    description: Images of fridge items for object detection.
    type: uri_folder
    path: azureml://datastores/workspaceblobstore/paths/fridge_items/images/
    ```
    
    建立標註資料資產的 YAML (`annotation_data.yml`):
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/asset.schema.json
    name: fridge-item-annotations-coco
    version: 1
    description: COCO annotations for fridge items.
    type: uri_file
    path: azureml://datastores/workspaceblobstore/paths/fridge_items/annotations/coco_annotations.json
    ```
    
    使用 CLI 創建資料資產：
    
    Bash
    
    ```
    az ml data create -f image_data.yml --resource-group <your-rg> --workspace-name <your-ws>
    az ml data create -f annotation_data.yml --resource-group <your-rg> --workspace-name <your-ws>
    ```
    

**步驟 3：開發訓練腳本 (Training Script Development)**

- **Azure ML 組件：** 計算執行個體 (Compute Instance) 作為開發環境。
- **說明：**
    - 使用您選擇的物件偵測框架（如 YOLOv8, TensorFlow Object Detection API, PyTorch Faster R-CNN）編寫 Python 訓練腳本 (`src/train.py`)。
    - **腳本功能：**
        - 接收參數：如數據路徑、學習率、訓練週期數等。
        - 加載數據：從 Azure ML 資料資產路徑讀取圖像和標註。
        - 數據預處理與增強。
        - 模型定義與初始化（或加載預訓練權重）。
        - 執行模型訓練。
        - **記錄指標 (Logging Metrics)：** 使用 MLflow (Azure ML 自動集成) 或 `azureml.core.Run.get_context().log()` (SDK v1 風格) / `MLClient` (SDK v2 風格) 記錄訓練過程中的損失 (loss)、準確率 (accuracy)、mAP (mean Average Precision) 等關鍵指標。這是 MLOps 的核心，用於追蹤實驗和模型比較。
        - **保存模型：** 將訓練好的模型文件（例如 `.pt` for YOLO, `saved_model` for TensorFlow）保存到指定的輸出目錄 (Azure ML 會自動捕獲此目錄下的內容)。
- **示例 (`src/train.py` 概念 - 使用 YOLOv8 和 MLflow)：**
    
    Python
    
    ```
    # src/train.py
    import argparse
    import os
    from ultralytics import YOLO
    import mlflow
    import yaml
    
    def main(args):
        # 使用 MLflow 自動記錄 (Azure ML 會自動設定 MLflow tracking URI)
        mlflow.autolog() # 可以自動記錄許多 Pytorch 指標和模型
    
        # 載入數據集設定 (假設有一個 data.yaml 描述了數據路徑)
        # 在實際的 Azure ML 作業中，args.data_config_path 會是掛載的數據資產路徑
        # data_yaml_path = os.path.join(args.data_config_path, 'data.yaml')
        # with open(data_yaml_path, 'r') as f:
        #     data_config = yaml.safe_load(f)
        # train_images_path = os.path.join(args.image_data_path, data_config['train'])
        # val_images_path = os.path.join(args.image_data_path, data_config['val'])
        # coco_annotations_path = args.coco_annotations_path # 假設是 COCO JSON 檔案路徑
    
        # 這裡簡化為直接使用 YOLOv8 的 COCO128 作為示例
        # 在實際場景中，你需要準備符合 YOLO 格式的數據集描述檔 (data.yaml)
        # 並將 image_data_path 和 coco_annotations_path (或其他標註格式) 轉換為 YOLO 需要的格式
        # 或者直接使用支援 COCO JSON 的框架功能
    
        # 載入一個預訓練模型 (例如 YOLOv8n)
        model = YOLO('yolov8n.pt')
    
        print(f"Starting training with epochs: {args.epochs}, batch_size: {args.batch_size}, learning_rate: {args.learning_rate}")
    
        # 訓練模型 (YOLOv8 的 train 方法)
        # 實際使用時，你需要配置 data 參數指向你的數據集描述檔
        # 例如: model.train(data='path/to/your_dataset.yaml', epochs=args.epochs, ...)
        results = model.train(
            data='coco128.yaml', # 示例，實際應替換為你的數據集配置文件
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=640,
            name='yolov8n_fridge_custom' # 實驗名稱
            # lr0=args.learning_rate # YOLOv8 可能有自己的學習率參數名稱
        )
    
        print(f"Training completed. Best mAP50-95: {results.results_dict['metrics/mAP50-95(B)']}")
        print(f"Model saved to: {results.save_dir}")
    
        # 將訓練好的模型（通常是 best.pt）保存到 Azure ML 指定的輸出路徑
        # Azure ML 會將 'outputs' 資料夾的內容自動上傳
        os.makedirs(args.model_output_path, exist_ok=True)
        best_model_path_source = os.path.join(results.save_dir, 'weights/best.pt')
        best_model_path_dest = os.path.join(args.model_output_path, 'best.pt')
        os.rename(best_model_path_source, best_model_path_dest)
        print(f"Best model saved to Azure ML output path: {best_model_path_dest}")
    
        # (可選) 使用 MLflow 明確記錄模型 (如果 autolog 沒有完全滿足需求)
        # mlflow.pytorch.log_model(model, "yolov8_fridge_model", registered_model_name="yolov8-fridge-detector")
    
    if __name__ == '__main__':
        parser = argparse.ArgumentParser()
        parser.add_argument('--image_data_path', type=str, required=False, help='Path to image data asset') # 示例參數
        parser.add_argument('--coco_annotations_path', type=str, required=False, help='Path to COCO annotation file asset') # 示例參數
        # parser.add_argument('--data_config_path', type=str, required=True, help='Path to data configuration folder (containing data.yaml)')
        parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
        parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
        parser.add_argument('--model_output_path', type=str, default='./outputs', help='Path to save trained model') # Azure ML 作業會將此路徑下的內容保存為輸出
    
        args = parser.parse_args()
        main(args)
    ```
    

**步驟 4：定義環境 (Define Environment)**

- **Azure ML 組件：** 環境 (Environments)
- **說明：**
    - 建立一個 YAML 檔案（例如 `environments/yolov8_env.yml`）來定義訓練和推斷所需的 Python 套件、CUDA 版本等。
    - 可以使用 Azure ML 的策劃環境 (curated environments) 作為基礎，或提供自訂的 Dockerfile。
- **示例 (`environments/yolov8_env.yml`):**
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/environment.schema.json
    name: yolov8-pytorch-gpu
    version: 1
    image: mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 # 使用一個帶 GPU 驅動的基礎映像
    conda_file: conda_dependencies.yml # 指向 conda 依賴文件
    description: Environment for training YOLOv8 object detection models with PyTorch and GPU.
    ```
    
    `conda_dependencies.yml`:
    
    YAML
    
    ```
    name: yolov8-env
    channels:
      - pytorch
      - nvidia # For CUDA toolkit if needed directly, often handled by base image
      - defaults
    dependencies:
      - python=3.9
      - pip
      - pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 # 確保與基礎映像的 CUDA 版本匹配
      - pip:
        - ultralytics==8.0.190 # 或更新版本
        - mlflow
        - azureml-mlflow
        - Pillow
        - matplotlib
        - opencv-python-headless
        - pyyaml
        - tqdm
    ```
    
    使用 CLI 創建環境：
    
    Bash
    
    ```
    az ml environment create -f environments/yolov8_env.yml --resource-group <your-rg> --workspace-name <your-ws>
    ```
    

**步驟 5：執行和追蹤訓練作業 (Run and Track Training Jobs)**

- **Azure ML 組件：** 作業 (Jobs), 計算叢集 (Compute Clusters), 實驗 (Experiments - 自動創建)
- **說明：**
    - 使用 Azure ML CLI 或 SDK 提交一個**命令作業 (Command Job)** 來執行您的 `train.py` 腳本。
    - 在作業定義中，您需要指定：
        - 要執行的程式碼。
        - 輸入數據（指向已註冊的資料資產）。
        - 使用的環境。
        - 計算目標（例如一個 GPU 計算叢集）。
    - Azure ML 會自動開始一個**實驗執行 (Experiment Run)**，並追蹤所有由 MLflow 或 Azure ML SDK 記錄的指標、參數、圖表和輸出的模型文件。
- **示例 (作業定義 YAML `train_job.yml`):**
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/commandJob.schema.json
    code: ../src  # 相對路徑，指向包含 train.py 的資料夾
    command: >-
      python train.py
      --epochs ${{inputs.epochs}}
      --batch_size ${{inputs.batch_size}}
      --learning_rate ${{inputs.learning_rate}}
      --model_output_path ${{outputs.model_output_folder}}
    #   --image_data_path ${{inputs.fridge_images}} # 如果需要傳遞數據路徑
    #   --coco_annotations_path ${{inputs.fridge_annotations_coco}} # 如果需要傳遞數據路徑
    inputs:
      epochs: 50 # 可以覆寫
      batch_size: 8
      learning_rate: 0.001
    #   fridge_images: # 示例數據輸入
    #     type: uri_folder
    #     path: azureml:fridge-item-images:1 # 指向註冊的資料資產
    #   fridge_annotations_coco: # 示例數據輸入
    #     type: uri_file
    #     path: azureml:fridge-item-annotations-coco:1
    outputs:
      model_output_folder: # 輸出模型將保存在這裡
        type: uri_folder
    environment: azureml:yolov8-pytorch-gpu:1 # 指向註冊的環境
    compute: azureml:gpu-cluster-nc6 # 指向您的 GPU 計算叢集名稱
    display_name: yolov8_fridge_items_training_run
    experiment_name: fridge_object_detection_training
    distribution: # 如果需要分散式訓練 (例如，YOLOv8 可能透過 torch.distributed.run)
      type: pytorch # 或 mpi
      process_count_per_instance: 1 # 每個節點的 GPU 數量
    ```
    
    使用 CLI 提交作業：
    
    Bash
    
    ```
    az ml job create -f train_job.yml --resource-group <your-rg> --workspace-name <your-ws>
    ```
    
    您可以在 Azure ML Studio 中監控作業進度並查看結果。

**步驟 6：模型評估與註冊 (Model Evaluation & Registration)**

- **Azure ML 組件：** 模型登錄 (Model Registry)
- **說明：**
    1. **評估：**
        - 訓練腳本 (`train.py`) 通常會包含驗證步驟，並計算 mAP 等指標。這些指標會被記錄到 Azure ML。
        - 您可以編寫一個單獨的評估腳本 (`evaluate.py`)，在獨立的作業中對測試數據集進行評估。
    2. **模型比較：** 在 Azure ML Studio 的“作業”或“模型”部分比較不同訓練執行的指標，選擇表現最佳的模型。
    3. **註冊模型：**
        - 將表現最佳的模型從作業的輸出中註冊到 Azure ML 模型登錄中。註冊時可以指定模型名稱、版本、描述、標籤以及相關指標。
        - 模型註冊是 MLOps 的關鍵步驟，它為模型提供了版本控制和治理。
- **示例 (在 `train.py` 結尾或單獨的腳本中註冊模型，使用 MLflow)：**
    
    Python
    
    ```
    # 在 train.py 的結尾 (概念性)
    # mlflow.register_model(
    #     model_uri=f"runs:/{mlflow.active_run().info.run_id}/outputs/model_output_folder", # 指向作業輸出中的模型路徑
    #     name="fridge-object-detector" # 模型登錄中的名稱
    # )
    ```
    
    或者使用 Azure ML CLI v2 (`model_register.yml`):
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/model.schema.json
    name: fridge-object-detector
    version: 1 # Azure ML 會自動遞增版本，或手動指定
    description: YOLOv8 model for detecting items in a fridge.
    path: azureml://jobs/<your_job_name>/outputs/artifacts/paths/model_output_folder/ # 指向特定作業的輸出路徑
    # 或者如果您下載了模型: path: ./downloaded_model_path
    type: mlflow_model # 或 custom_model, onnx_model 等
    properties:
      mAP: 0.75 # 示例指標
      framework: YOLOv8
    ```
    
    註冊模型：
    
    Bash
    
    ```
    # 假設 <your_job_name> 是訓練作業的名稱，如: lucid_glove_j3xl9jqx99
    # 你需要從 Azure ML Studio 或 az ml job show 命令中獲取它
    # 更新 YAML 中的 path
    az ml model create -f model_register.yml --resource-group <your-rg> --workspace-name <your-ws>
    ```
    

**步驟 7：模型部署 (Model Deployment)**

- **Azure ML 組件：** 端點 (Endpoints) - 受控線上端點 (Managed Online Endpoints) 或批次端點 (Batch Endpoints)
- **說明：**
    1. **編寫推斷腳本 (`score.py`)：**
        - 此腳本定義模型如何加載以及如何處理傳入的推斷請求。
        - 包含 `init()` 函數：在服務啟動時執行，通常用於加載模型。模型路徑通常通過環境變數 `AZUREML_MODEL_DIR` 獲得。
        - 包含 `run()` 函數：處理每個推斷請求，接收輸入數據（例如圖像），進行預處理，執行模型預測，然後對預測結果進行後處理，返回 JSON 格式的結果（例如邊界框座標、類別、信賴度）。
    2. **定義線上端點 (Online Endpoint)：**
        - 端點是模型部署的穩定 HTTPS 入口。
    3. **定義線上部署 (Online Deployment)：**
        - 一個端點下可以有多個部署（例如用於藍/綠部署或 A/B 測試）。
        - 部署定義中指定要部署的已註冊模型、`score.py` 腳本、推斷環境、以及計算資源（VM 類型和實例數量）。
- **示例 (`src/score.py` 概念 - 適用於 YOLOv8)：**
    
    Python
    
    ```
    # src/score.py
    import os
    import json
    import base64
    from io import BytesIO
    from PIL import Image
    from ultralytics import YOLO
    import logging
    
    # Called when the service is loaded
    def init():
        global model
        # Get the path to the deployed model file
        # AZUREML_MODEL_DIR is an environment variable created during deployment.
        # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
        # This assumes your model file (e.g., best.pt) is in the root of what was registered.
        model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR", default="."), "best.pt")
        logging.info(f"Loading model from path: {model_path}")
        try:
            model = YOLO(model_path)
            logging.info("Model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise
    
    # Called when a request is received
    def run(raw_data):
        logging.info(f"Request received: {raw_data[:100]}") # Log first 100 chars of raw_data
    
        try:
            # Expects JSON input with an "image" field containing base64 encoded image string
            data = json.loads(raw_data)
            image_bytes = base64.b64decode(data["image"])
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            logging.error(f"Error processing input data: {e}")
            return json.dumps({"error": "Invalid input. Expecting JSON with base64 encoded image."}), 400
    
        try:
            # Perform inference
            results = model(image) # YOLOv8 model call
    
            detections = []
            for result in results: # Results is a list of Results objects
                boxes = result.boxes.cpu().numpy() # get boxes on CPU in numpy
                for box in boxes: # Log each box
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    coords = box.xyxyn[0].tolist() # Normalized xyxy
                    detections.append({
                        "class_id": class_id,
                        "class_name": model.names[class_id],
                        "confidence": confidence,
                        "box_xyxyn": coords # Normalized coordinates [xmin, ymin, xmax, ymax]
                    })
            logging.info(f"Detections: {detections}")
            return json.dumps(detections)
        except Exception as e:
            logging.error(f"Error during inference: {e}")
            return json.dumps({"error": f"Inference error: {e}"}), 500
    
    if __name__ == '__main__':
        # Test init()
        # Set AZUREML_MODEL_DIR to the path of your model for local testing
        # For example, if 'best.pt' is in a 'my_model' subfolder of where you registered the model:
        # os.environ["AZUREML_MODEL_DIR"] = "./outputs/my_model/" # or path to your actual model file's directory
        # init()
        #
        # # Test run()
        # # Create a sample base64 image string
        # try:
        #     with open("path_to_a_test_image.jpg", "rb") as img_file:
        #         b64_string = base64.b64encode(img_file.read()).decode('utf-8')
        #     test_data = json.dumps({"image": b64_string})
        #     predictions = run(test_data)
        #     print(predictions)
        # except FileNotFoundError:
        #     print("Test image not found. Skipping local run test.")
        # except Exception as e:
        #     print(f"Error in local test: {e}")
        pass
    ```
    
    **線上端點 YAML (`endpoints/online_endpoint.yml`):**
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/managedOnlineEndpoint.schema.json
    name: fridge-detector-endpoint # 端點名稱，需在區域內唯一
    auth_mode: key # 或 aml_token
    ```
    
    **線上部署 YAML (`endpoints/online_deployment.yml`):**
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/managedOnlineDeployment.schema.json
    name: blue # 部署的名稱
    endpoint_name: fridge-detector-endpoint # 關聯的端點名稱
    model: azureml:fridge-object-detector:1 # 指向已註冊的模型
    environment: azureml:yolov8-pytorch-gpu:1 # 推斷環境，可能與訓練環境相同或更輕量
    code_configuration:
      code: ../src # 包含 score.py 的路徑
      scoring_script: score.py
    instance_type: Standard_DS3_v2 # VM SKU
    instance_count: 1
    # request_settings: # (可選)
    #   request_timeout_ms: 5000
    #   max_concurrent_requests_per_instance: 1
    #   max_queue_wait_ms: 500
    ```
    
    部署流程 (CLI)：
    
    Bash
    
    ```
    az ml online-endpoint create -f endpoints/online_endpoint.yml -g <your-rg> -w <your-ws>
    az ml online-deployment create -f endpoints/online_deployment.yml --all-traffic -g <your-rg> -w <your-ws>
    ```
    

**步驟 8：模型監控 (Model Monitoring)**

- **Azure ML 組件：** 數據漂移監控 (Data Drift Monitors), Azure Application Insights (用於線上端點)
- **說明：**
    1. **數據漂移：**
        - 設定數據漂移監控器來比較生產推斷數據的特徵分佈與訓練數據集的分佈。
        - 如果檢測到顯著漂移，可能需要重新訓練模型。對於物件偵測，漂移可能體現在圖像的亮度、對比度、新物件的出現等。
    2. **模型性能：**
        - 對於線上端點，Azure ML 會自動將日誌（包括 `score.py` 中的 `print` 和 `logging` 輸出）發送到關聯的 Azure Application Insights。您可以設置警報和儀表板來監控請求延遲、錯誤率等。
        - 定期收集生產環境中的預測結果和真實標籤（如果可能），計算 mAP 等指標，監控模型性能是否下降。
    3. **服務健康：** 監控端點的可用性和回應時間。
- **示例 (啟用 App Insights)：** 在 Azure ML Studio 中為線上端點啟用 Application Insights。

**步驟 9：自動化管線與再訓練 (Automation Pipelines & Retraining)**

- **Azure ML 組件：** 管線 (Pipelines), 元件 (Components)
- **說明：**
    - 使用 Azure ML 管線將上述步驟（數據準備、訓練、評估、註冊、部署）串聯成一個自動化的工作流程。
    - 管線可以使用 YAML 或 Python SDK 定義。
    - **觸發器：**
        - **排程觸發 (Schedule-based):** 定期（例如每週或每月）自動執行管線以使用最新數據重新訓練模型。
        - **數據變動觸發 (Data-driven):** 當 Blob Storage 中的新數據達到一定量時觸發管線。
        - **監控警報觸發：** 當數據漂移或模型性能下降到一定閾值時觸發再訓練管線。
- **示例 (管線 YAML `pipelines/training_pipeline.yml` 概念)：**
    
    YAML
    
    ```
    $schema: https://azuremlschemas.azureedge.net/latest/pipelineJob.schema.json
    type: pipeline
    display_name: Fridge_Object_Detection_Training_Pipeline
    description: Pipeline to train, evaluate, and register an object detection model.
    
    settings:
      default_compute: azureml:gpu-cluster-nc6 # 管線的預設計算目標
    
    inputs: # 管線級別的輸入
      pipeline_epochs: 50
      pipeline_batch_size: 8
      pipeline_registered_model_name: "fridge-object-detector"
      # pipeline_image_data:
      #   type: uri_folder
      #   path: azureml:fridge-item-images:1
      # pipeline_annotation_data:
      #   type: uri_file
      #   path: azureml:fridge-item-annotations-coco:1
    
    jobs:
      # 步驟 1: 訓練模型 (使用前面定義的作業作為元件，或直接定義命令作業)
      train_model_job:
        type: command # 直接使用命令作業
        name: train_yolov8_model
        display_name: Train YOLOv8 Fridge Detector
        command: >-
          python train.py
          --epochs ${{parent.inputs.pipeline_epochs}}
          --batch_size ${{parent.inputs.pipeline_batch_size}}
          --model_output_path ${{outputs.model_output_folder_train}}
        #   --image_data_path ${{parent.inputs.pipeline_image_data}}
        #   --coco_annotations_path ${{parent.inputs.pipeline_annotation_data}}
        code: ../src
        environment: azureml:yolov8-pytorch-gpu:1
        # inputs: # 如果作業有自己的輸入定義
        outputs:
          model_output_folder_train: # 此輸出名需與 train.py 中的 --model_output_path 對應的輸出名匹配
            type: uri_folder
        distribution:
          type: pytorch
          process_count_per_instance: 1
    
      # 步驟 2: (可選) 評估模型 (假設有一個 evaluate.py)
      # evaluate_model_job:
      #   type: command
      #   name: evaluate_trained_model
      #   # ... (類似 train_model_job 的定義，輸入為 train_model_job 的輸出模型)
      #   inputs:
      #     trained_model: ${{parent.jobs.train_model_job.outputs.model_output_folder_train}}
      #     test_data: ...
      #   outputs:
      #     evaluation_results: ...
    
      # 步驟 3: 註冊模型
      register_model_job:
        type: command # 可以是一個簡單的腳本或使用 Azure ML CLI 的元件
        name: register_best_model
        display_name: Register Best Fridge Detector Model
        command: >-
          python register_model_script.py # 一個簡單的 Python 腳本來處理註冊邏輯
          --model_input_path ${{parent.jobs.train_model_job.outputs.model_output_folder_train}}
          --model_name ${{parent.inputs.pipeline_registered_model_name}}
          --mAP_metric "0.78" # 假設這個值來自評估步驟
        code: ../src # 假設 register_model_script.py 在 src 中
        environment: azureml:yolov8-pytorch-gpu:1 # 可以是一個更輕量的環境
    ```
    

inputs: (來自評估作業的指標等) ``*`register_model_script.py` (概念):*``python # src/register_model_script.py import argparse import os from azure.ai.ml import MLClient from azure.ai.ml.entities import Model from azure.ai.ml.constants import AssetTypes from azure.identity import DefaultAzureCredential

````
parser = argparse.ArgumentParser()
parser.add_argument("--model_input_path", type=str, required=True)
parser.add_argument("--model_name", type=str, required=True)
parser.add_argument("--mAP_metric", type=float, required=True) # 示例指標
args = parser.parse_args()

ml_client = MLClient.from_config(DefaultAzureCredential())

model_to_register = Model(
    name=args.model_name,
    path=args.model_input_path, # 指向訓練作業輸出的模型文件夾
    type=AssetTypes.MLFLOW_MODEL, # 或 CUSTOM_MODEL，取決於模型保存格式
    description=f"Fridge object detection model with mAP: {args.mAP_metric}",
    properties={"mAP": args.mAP_metric, "framework": "YOLOv8"}
)
ml_client.models.create_or_update(model_to_register)
print(f"Model {args.model_name} registered.")
```
提交管線：
```bash
az ml job create -f pipelines/training_pipeline.yml -g <your-rg> -w <your-ws>
```
````

**步驟 10：CI/CD (持續整合/持續交付)**

- **工具：** Azure DevOps Pipelines, GitHub Actions
- **說明：**
    - 整合您的 Git 儲存庫與 CI/CD 工具。
    - **CI (持續整合)：** 當程式碼（例如訓練腳本）被推送到特定分支（如 `main` 或 `dev`）時，自動觸發：
        - 程式碼檢查、單元測試。
        - （可選）觸發 Azure ML 訓練管線的驗證版本。
    - **CD (持續交付/部署)：**
        - 當 Azure ML 訓練管線成功執行並註冊了新版本的模型後，如果模型指標達到預期，CD 管線可以自動觸發：
            - 將模型部署到預備環境 (staging deployment)。
            - 執行整合測試和煙霧測試。
            - 如果測試通過，逐步將流量切換到新模型或將模型推廣到生產部署。

**總結**

透過上述步驟，您可以利用 Azure Machine Learning 為物件偵測模型建立一個端到端的 MLOps 工作流程。這個流程涵蓋了從數據準備、模型訓練、版本控制、部署到監控和自動化再訓練的完整生命週期，有助於提高模型開發的效率、可靠性和可維護性。請注意，以上提供的程式碼和 YAML 都是概念性的，您需要根據您的具體框架、數據和需求進行調整。



###### MLOPs using AWS
## AWS S3 與 AWS SageMaker 詳解：AI 模型開發與 MLOps 的基石

在 AWS 上開發 AI 模型、進行部署以及實踐完整的 MLOps 流程，**Amazon S3 (Simple Storage Service)** 和 **Amazon SageMaker** 無疑是兩大核心服務。它們各自承擔著不同的關鍵職責，但又緊密協作，共同構建起一個強大、可擴展且託管的機器學習生態系統。

簡單來說，**是的，若您選擇在 AWS 上進行 AI 模型開發與部署，並實踐完整的 MLOps 流程，S3 和 SageMaker 是您最主要且必不可少的工具集。**S3 扮演著「數據湖」的角色，儲存著模型訓練所需的所有數據和模型本身；而 SageMaker 則是「智能工廠」，負責模型訓練、部署、管理與監控的方方面面。

### 一、 AWS S3 (Simple Storage Service) 介紹

AWS S3 是一個高度可擴展、安全、耐用且具成本效益的**物件儲存服務**。它被廣泛應用於各種場景，包括大數據湖、網站託管、備份和歸檔等。在 AI/ML 領域，S3 更是扮演著數據中心樞紐的角色。

**主要特性：**

- **物件儲存：** S3 以物件的形式儲存數據，每個物件包含數據本身、一個唯一識別符和元數據。
    
- **高擴展性：** 幾乎無限的儲存空間，您無需擔心容量限制。
    
- **高耐久性與可用性：** 數據會自動複製到多個可用區，即使一個可用區發生故障，數據也不會丟失。S3 標準儲存的年耐久性達到 99.999999999%。
    
- **安全性：** 提供多種加密選項（靜態加密和傳輸中加密）、IAM 策略、儲存桶策略、ACLs 等，確保數據安全。
    
- **成本效益：** 提供多種儲存類別，您可以根據數據的存取頻率選擇最經濟的儲存方式。
    
- **事件通知：** S3 可以配置為在特定事件發生時（如物件上傳、刪除）發送通知，觸發其他 AWS 服務（如 Lambda）。
    

**在 MLOps 中的作用：**

S3 是 MLOps 流程中所有數據和模型 artifacts 的**中央儲存庫**：

- **原始數據湖：** 儲存未處理的原始手術視訊、影像數據。
    
- **處理後的數據：** 儲存經過預處理、增強、標註的影像幀數據。
    
- **模型 artifacts：** 儲存訓練好的模型權重、模型定義文件、推理腳本等。
    
- **日誌與監控數據：** 儲存訓練日誌、部署日誌、模型監控數據。
    

---

### 二、 AWS SageMaker 介紹

Amazon SageMaker 是一個**全託管的機器學習服務**，它涵蓋了機器學習的整個生命週期，從數據準備、模型構建、訓練、調優到部署和管理。SageMaker 旨在簡化 ML 流程，讓數據科學家和開發者能更快地將模型投入生產。

**主要組件與功能：**

- **SageMaker Studio：** 統一的基於 Web 的 IDE，用於整個 ML 開發流程。
    
- **SageMaker Data Wrangler：** 用於數據聚合和準備的視覺化工具。
    
- **SageMaker Processing：** 用於大規模數據預處理、特徵工程和模型評估的託管計算服務。
    
- **SageMaker Training：** 託管的訓練環境，支持主流 ML 框架、內置演算法、分佈式訓練和超參數優化。
    
- **SageMaker Model Registry：** 集中管理和版本控制訓練好的模型。
    
- **SageMaker Endpoints：** 用於部署模型為實時或批次推理服務，支持自動擴展和 A/B 測試。
    
- **SageMaker Model Monitor：** 持續監控生產環境中模型的性能和數據/概念漂移。
    
- **SageMaker Pipelines：** 用於構建、管理和自動化 MLOps 工作流。
    
- **SageMaker Ground Truth：** 用於大規模數據標註的服務。
    

**在 MLOps 中的作用：**

SageMaker 提供了 MLOps 流程中所有**核心計算和管理能力**，是實現自動化、可重複和可擴展 ML 工作流的關鍵：

- **實驗與訓練執行：** 在託管環境中執行模型訓練。
    
- **模型管理：** 儲存和管理模型版本。
    
- **部署與服務：** 提供生產級的推理服務。
    
- **監控與自動化：** 監控模型性能並自動觸發再訓練。
    

---

### 三、 AWS MLOps 的 7 個核心功能描述與示例概念

以下將詳細闡述如何利用 AWS 的服務（主要為 S3 和 SageMaker）來實現 MLOps 的 7 個核心功能，並提供概念性的操作示例。

#### 1. 數據管理與版本控制 (Data Management & Versioning)

- **描述：** 確保訓練和測試數據的質量、可追溯性與可重複性。包括數據的攝取、儲存、預處理、標註、版本化和血緣追溯。
    
- **AWS 實現：** 主要依賴 **Amazon S3** 進行數據儲存和版本控制，**SageMaker Ground Truth** 進行標註。
    
- **示例概念：**
    
    - **S3 作為數據湖：**
        
        Bash
        
        ```
        # 創建 S3 桶用於儲存原始手術視訊
        aws s3 mb s3://surgical-video-raw-data-bucket-unique-name
        
        # 上傳視訊文件
        aws s3 cp my-surgery.mp4 s3://surgical-video-raw-data-bucket-unique-name/raw/my-surgery.mp4
        
        # 啟用 S3 桶的版本控制
        aws s3api put-bucket-versioning --bucket surgical-video-raw-data-bucket-unique-name --versioning-configuration Status=Enabled
        ```
        
    - **數據預處理與幀提取 (使用 SageMaker Processing Job)：**
        
        - **Python 腳本 (`video_frame_extractor.py`)：**
            
            Python
            
            ```
            import os
            import sagemaker
            from sagemaker.processing import Processor, ProcessingInput, ProcessingOutput
            import boto3
            
            # ... 載入視訊，用 OpenCV 提取幀，儲存到 /opt/ml/processing/output/frames
            
            # 在 SageMaker Processing 中運行
            processor = Processor(
                role=sagemaker.get_execution_role(),
                instance_count=1,
                instance_type='ml.m5.xlarge',
                image_uri='your_custom_image_with_ffmpeg_opencv', # 或使用 SageMaker 提供的框架映像
                max_runtime_in_seconds=3600
            )
            
            processor.run(
                inputs=[
                    ProcessingInput(
                        source='s3://surgical-video-raw-data-bucket-unique-name/raw/my-surgery.mp4',
                        destination='/opt/ml/processing/input',
                        s3_data_type='S3Prefix',
                        s3_input_mode='File'
                    )
                ],
                outputs=[
                    ProcessingOutput(
                        source='/opt/ml/processing/output/frames',
                        destination='s3://surgical-video-processed-frames-bucket-unique-name/processed-frames/',
                        s3_data_type='S3Prefix'
                    )
                ],
                code='video_frame_extractor.py'
            )
            ```
            
    - **數據標註 (SageMaker Ground Truth)：**
        
        - 在 AWS 控制台或使用 SDK 配置 Ground Truth 標註作業，將 `s3://surgical-video-processed-frames-bucket-unique-name/processed-frames/` 作為輸入，定義物件偵測標籤（如 `Scalpel`, `Tumor`）。
            

#### 2. 實驗管理與追蹤 (Experiment Management & Tracking)

- **描述：** 記錄所有 ML 實驗的詳細資訊，包括程式碼版本、超參數、環境配置、訓練指標、模型權重、日誌等。
    
- **AWS 實現：** **SageMaker Experiments** 是核心，搭配 **CloudWatch** 監控日誌和指標。
    
- **示例概念：**
    
    - 在 SageMaker Studio Notebook 或訓練腳本中：
        
        Python
        
        ```
        import sagemaker
        from sagemaker.experiments import Run
        
        # 創建或恢復一個實驗運行
        with Run(experiment_name='surgical-object-detection-experiment',
                 run_name='yolo-v5-run-001') as run:
            # 訓練程式碼
            # ...
        
            # 記錄超參數
            run.log_parameters({
                "learning_rate": 0.001,
                "epochs": 50,
                "batch_size": 16
            })
        
            # 記錄指標
            for epoch in range(50):
                # ... 訓練邏輯
                mAP = calculate_mAP() # 假設計算出 mAP
                loss = calculate_loss() # 假設計算出 loss
                run.log_metrics({
                    "mAP": mAP,
                    "loss": loss
                }, step=epoch)
        
            # 記錄模型 artifacts 路徑
            run.log_file(name='model_artifacts', s3_uri='s3://your-model-bucket/yolo-v5-run-001/model.tar.gz')
            # 記錄訓練腳本
            run.log_file(name='training_script', local_path='train.py')
        ```
        
    - 在 SageMaker Studio 的 **Experiments** 界面中可視化和比較不同實驗的指標和超參數。
        

#### 3. 模型訓練與自動化 (Model Training & Automation)

- **描述：** 提供可擴展的計算資源和工具，用於訓練 ML 模型。支持自動化訓練流程、超參數優化、分佈式訓練和定時/事件觸發的再訓練。
    
- **AWS 實現：** **SageMaker Training** 承擔核心訓練任務，**SageMaker HyperParameter Tuning** 進行自動優化。
    
- **示例概念：**
    
    - **定義訓練器 (Estimator)：**
        
        Python
        
        ```
        from sagemaker.pytorch import PyTorch
        
        # 使用 PyTorch Estimator 定義訓練任務
        estimator = PyTorch(
            entry_point='train_script.py',        # 您的訓練腳本
            role=sagemaker.get_execution_role(),
            framework_version='1.13.1',           # PyTorch 版本
            py_version='py39',
            instance_count=1,
            instance_type='ml.g4dn.xlarge',       # 帶 GPU 的實例
            hyperparameters={                     # 傳遞給訓練腳本的超參數
                'epochs': 50,
                'batch-size': 16,
                'learning-rate': 0.001
            },
            output_path='s3://your-model-bucket/models/' # 模型輸出路徑
        )
        
        # 啟動訓練
        estimator.fit({'training': 's3://surgical-video-processed-frames-bucket-unique-name/processed-frames/'})
        ```
        
    - **超參數優化 (Hyperparameter Tuning)：**
        
        Python
        
        ```
        from sagemaker.tuner import HyperparameterTuner, IntegerParameter, CategoricalParameter, ContinuousParameter
        
        # 定義超參數範圍
        hyperparameter_ranges = {
            'learning-rate': ContinuousParameter(0.0001, 0.01),
            'epochs': IntegerParameter(20, 100),
            'optimizer': CategoricalParameter(['adam', 'sgd'])
        }
        
        # 定義超參數調優器
        tuner = HyperparameterTuner(
            estimator,
            objective_metric_name='mAP',     # 優化目標指標
            objective_type='Maximize',       # 最大化 mAP
            hyperparameter_ranges=hyperparameter_ranges,
            max_jobs=10,                     # 最多運行 10 個訓練作業
            max_parallel_jobs=2              # 最多同時運行 2 個作業
        )
        
        # 啟動超參數調優
        tuner.fit({'training': 's3://surgical-video-processed-frames-bucket-unique-name/processed-frames/'})
        ```
        

#### 4. 模型管理與版本控制 (Model Management & Versioning)

- **描述：** 集中儲存、管理和版本化訓練好的模型。包括模型註冊、元數據儲存、模型評估結果記錄和模型生命週期管理。
    
- **AWS 實現：** **SageMaker Model Registry** 是核心。
    
- **示例概念：**
    
    - **註冊模型到 Model Registry：**
        
        Python
        
        ```
        from sagemaker.model import Model
        from sagemaker.model_metrics import ModelMetrics, MetricsSource
        from sagemaker.workflow.model_step import RegisterModel
        
        # 假設這是訓練好的模型路徑和推理映像
        model_data_s3_path = estimator.model_data
        inference_image_uri = 'your_custom_inference_image_uri' # 或 SageMaker 提供的框架映像
        
        # 創建 SageMaker Model 物件
        model = Model(
            image_uri=inference_image_uri,
            model_data=model_data_s3_path,
            role=sagemaker.get_execution_role()
        )
        
        # 定義模型指標 (評估結果，假設已儲存到 S3)
        model_metrics = ModelMetrics(
            model_statistics=MetricsSource(
                s3_uri='s3://your-model-bucket/evaluation_results/statistics.json',
                content_type='application/json'
            )
        )
        
        # 將模型註冊到 Model Registry
        register_model_step = RegisterModel(
            name='SurgicalObjectDetectionModel',
            estimator=estimator, # 從哪個訓練器產生的
            model_data=model_data_s3_path,
            content_types=["image/jpeg"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge", "ml.g4dn.xlarge"],
            transform_instances=["ml.m5.xlarge"],
            description="YOLOv5 model for surgical tool and tumor detection.",
            model_metrics=model_metrics,
            # model_package_group_name="SurgicalModelPackageGroup" # 如果有 Model Package Group
        )
        
        # 在 SageMaker Pipelines 中執行時，這會被自動註冊
        # 如果是獨立運行，需要使用 model.register()
        # model.register(
        #     model_package_group_name="SurgicalObjectDetectionGroup",
        #     content_types=["image/jpeg"],
        #     response_types=["application/json"],
        #     inference_instances=["ml.m5.xlarge", "ml.g4dn.xlarge"],
        #     transform_instances=["ml.m5.xlarge"]
        # )
        ```
        
    - 在 SageMaker Studio 的 **Model Registry** 界面中管理不同版本的模型、查看其詳細資訊和狀態（已核准、待審核等）。
        

#### 5. 模型部署與服務 (Model Deployment & Serving)

- **描述：** 將訓練好的模型打包並部署為可供應用程式調用的 API 服務。支持實時推理、批次推理、A/B 測試和自動擴展。
    
- **AWS 實現：** **SageMaker Endpoints** 是核心，搭配 **Lambda** 和 **API Gateway** 提供 API 接口。
    
- **示例概念：**
    
    - **部署到 SageMaker Endpoint (實時推理)：**
        
        Python
        
        ```
        # 從 Model Registry 獲取模型版本或直接從訓練器獲取模型
        # model = Model(image_uri=inference_image_uri, model_data=model_data_s3_path, role=sagemaker.get_execution_role())
        # 或直接使用 estimator 部署
        predictor = estimator.deploy(
            initial_instance_count=1,
            instance_type='ml.g4dn.xlarge', # 或更便宜的 CPU 實例 ml.m5.xlarge
            endpoint_name='surgical-object-detection-endpoint'
        )
        
        # 進行推理
        # response = predictor.predict(image_data)
        ```
        
    - **整合 Lambda + API Gateway：**
        
        - 創建一個 **Lambda 函數**，其程式碼負責接收來自 API Gateway 的請求，將請求內容（例如 Base64 編碼的影像）發送到 SageMaker Endpoint 進行推理，然後格式化推理結果並返回。
            
        - 配置 **API Gateway**，創建一個 REST API，將其方法（如 POST）與上述 Lambda 函數整合。
            
        - 這允許您透過標準 HTTP 請求從任何應用程式調用模型。
            

#### 6. 模型監控與再訓練 (Model Monitoring & Retraining)

- **描述：** 持續監控生產環境中模型的性能表現、輸入數據分佈和模型輸出分佈。當性能下降或數據變化時，自動觸發警報或再訓練流程。
    
- **AWS 實現：** **SageMaker Model Monitor** 用於監控，**CloudWatch** 發送警報，**Step Functions** 或 **SageMaker Pipelines** 編排再訓練工作流。
    
- **示例概念：**
    
    - **配置 Model Monitor：**
        
        Python
        
        ```
        from sagemaker.model_monitor import DataCaptureConfig, ModelMonitor
        
        # 配置數據捕獲：將推理請求和響應數據儲存到 S3
        data_capture_config = DataCaptureConfig(
            enable_capture=True,
            sampling_percentage=100, # 捕獲所有數據
            destination_s3_uri='s3://your-monitor-bucket/data-capture/',
            kms_key_id=None
        )
        
        # 啟動 Model Monitor
        monitor = ModelMonitor(
            role=sagemaker.get_execution_role(),
            instance_count=1,
            instance_type='ml.m5.xlarge',
            max_runtime_in_seconds=3600
        )
        
        # 將數據捕獲配置應用到現有端點
        predictor.update_endpoint(data_capture_config=data_capture_config)
        
        # 創建監控排程 (每小時運行一次分析)
        monitor.create_monitoring_schedule(
            endpoint_input=predictor.endpoint_name,
            record_preprocessor_script='preprocessor.py', # 可選：預處理捕獲數據
            post_processor_script='postprocessor.py',     # 可選：後處理分析結果
            output_s3_uri='s3://your-monitor-bucket/analysis-output/',
            schedule_cron_expression='cron(0 * ? * * *)' # 每小時運行一次
        )
        
        # 設定監控指標和警報規則 (在 CloudWatch 中配置)
        # 例如，當輸入數據的某一特徵分佈與基準數據集顯著不同時發送警報。
        # 警報可觸發 Lambda，進而啟動再訓練管道。
        ```
        
    - **自動再訓練工作流 (使用 AWS Step Functions 或 SageMaker Pipelines)：**
        
        - 當 CloudWatch 警報觸發時，啟動一個 Step Functions 工作流。
            
        - 工作流可以包含步驟：`數據收集` -> `數據標註 (Ground Truth)` -> `模型訓練 (SageMaker Training)` -> `模型評估` -> `模型註冊` -> `條件判斷 (性能是否提升)` -> `模型部署 (SageMaker Endpoint Update)`。
            

#### 7. ML 管道與工作流自動化 (ML Pipeline & Workflow Automation)

- **描述：** 將所有功能模塊化並串聯起來，形成自動化的端到端 ML 工作流。
    
- **AWS 實現：** **SageMaker Pipelines** 提供託管的 ML 工作流服務，與 **AWS CodeCommit/CodeBuild/CodePipeline** 實現 CI/CD。
    
- **示例概念：**
    
    - **使用 SageMaker Pipelines 定義端到端工作流：**
        
        Python
        
        ```
        from sagemaker.workflow.pipeline import Pipeline
        from sagemaker.workflow.steps import ProcessingStep, TrainingStep
        from sagemaker.workflow.parameters import ParameterString, ParameterInteger
        
        # 定義管道參數
        input_data = ParameterString(name="InputData", default_value="s3://surgical-video-raw-data-bucket-unique-name/raw/")
        instance_type = ParameterString(name="InstanceType", default_value="ml.m5.xlarge")
        epochs = ParameterInteger(name="Epochs", default_value=50)
        
        # 步驟 1: 數據處理 (例如，視訊幀提取)
        process_step = ProcessingStep(
            name="VideoFrameExtraction",
            processor=processor, # 上面定義的 Processor
            inputs=[
                ProcessingInput(source=input_data, destination='/opt/ml/processing/input')
            ],
            outputs=[
                ProcessingOutput(source='/opt/ml/processing/output/frames', destination='s3://surgical-video-processed-frames-bucket-unique-name/processed-frames/')
            ],
            code='video_frame_extractor.py'
        )
        
        # 步驟 2: 模型訓練
        train_step = TrainingStep(
            name="ModelTraining",
            estimator=estimator, # 上面定義的 Estimator
            inputs={
                'training': sagemaker.inputs.TrainingInput(s3_data=process_step.properties.ProcessingOutputConfig.Outputs['frames'].S3Output.S3Uri)
            }
        )
        
        # 步驟 3: 模型註冊
        register_step = RegisterModel(
            name="RegisterModel",
            estimator=estimator,
            model_data=train_step.properties.ModelArtifacts.S3ModelDataUrl,
            content_types=["image/jpeg"],
            response_types=["application/json"],
            inference_instances=["ml.m5.xlarge"],
            description="Registered model from pipeline."
        )
        
        # 創建管道
        pipeline = Pipeline(
            name="SurgicalObjectDetectionPipeline",
            parameters=[input_data, instance_type, epochs],
            steps=[process_step, train_step, register_step]
        )
        
        # 提交並啟動管道
        # pipeline.upsert(role_arn=sagemaker.get_execution_role())
        # pipeline.start()
        ```
        
    - **CI/CD 整合 (AWS CodePipeline)：**
        
        - 配置 CodePipeline，監聽 **CodeCommit** 中的程式碼變更。
            
        - 當程式碼提交時，觸發 **CodeBuild** 執行單元測試和構建 Docker 映像。
            
        - 然後，CodePipeline 可以啟動上述定義的 **SageMaker Pipeline**，自動執行數據處理、訓練、評估和模型註冊。
            
        - 如果模型通過評估，CodePipeline 可以自動觸發 **SageMaker Endpoint** 的更新，實現無縫部署。
            

---

### 總結

**AWS S3 和 AWS SageMaker 確實是您在 AWS 上開發 AI 模型、進行部署和實施完整 MLOps 流程的核心。**S3 提供高可靠的數據儲存，而 SageMaker 則提供了一套全面的託管服務，覆蓋了 MLOps 的所有關鍵環節：從數據準備、實驗追蹤、模型訓練、管理、部署到監控和管道自動化。透過這些服務的緊密協作，AWS 使得即使是複雜的手術影像分析 MLOps 也能夠高效、可擴展且可靠地實現。