
|                              |     |
| ---------------------------- | --- |
| [[### MLOPs using AzureML ]] |     |
|                              |     |


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