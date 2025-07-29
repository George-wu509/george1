

Data preparation
 - SegeMaker Ground Truth: 資料標註

Model training
 - SegeMaker NoteBook: 編寫訓練腳本
 - SageMaker Training Job: 進行模型訓練
 - SageMaker Deployment: 部署Inference Endpoint

Automatic service inference
 - Amazon S3: 上傳手術影片到Amazon S3
 - S3 ObjectCreated(): 觸發Lambda function
 - Lambda function: 解析上傳影片名稱和檔案路徑
 - AWS Step Functions: 工作流程協調服務
 - SageMaker: 啟動SageMaker 推論端點


---

## AWS SageMaker 與 AWS Lambda 深度解析及應用

在現代雲端運算和人工智慧領域，AWS SageMaker 和 AWS Lambda 是兩項功能強大且相輔相成的服務。SageMaker 簡化了機器學習模型的整個生命週期，而 Lambda 則提供了事件驅動的無伺服器運算能力。兩者結合可以打造出高度自動化、可擴展且符合成本效益的智慧應用。

### 1. AWS SageMaker：一站式機器學習平台

**AWS SageMaker** 是一個全託管的服務 (Fully Managed Service)，旨在幫助開發人員和資料科學家快速、輕鬆地大規模建置、訓練和部署機器學習 (ML) 模型。它移除了傳統 ML 開發中許多複雜且耗時的環節，讓您可以專注於模型的核心演算法與業務邏輯。

#### SageMaker 的核心功能組件

SageMaker 將機器學習工作流程劃分為四個主要階段，並為每個階段提供專門的工具：

1. **準備 (Prepare)**
    
    - **SageMaker Studio Lab & Notebooks**: 提供基於 Jupyter Notebook 的整合式開發環境 (IDE)，用於資料探索、前處理和模型開發。
        
    - **SageMaker Data Wrangler**: 一個視覺化的資料準備工具，可以快速地透過點擊操作來清理、轉換和理解資料，並自動生成處理程式碼。
        
    - **SageMaker Ground Truth**: 協助您建立高品質、經過精確標註的訓練資料集。它支援使用人力標註員、機器學習輔助標註等多種方式。
        
2. **建置 (Build)**
    
    - 提供針對 TensorFlow, PyTorch, Scikit-learn, MXNet 等主流框架最佳化的環境。
        
    - 內建超過 15 種高效能演算法，例如 XGBoost、影像分類、物件偵測等，可直接取用。
        
    - 允許您攜帶自己的演算法或模型，並在 Docker 容器中執行。
        
3. **訓練與調校 (Train & Tune)**
    
    - **一鍵式訓練 (One-click Training)**: 只需幾行程式碼或在控制台中點擊幾下，即可啟動一個分佈式的模型訓練任務。您無需管理底層的計算資源，SageMaker 會自動佈建、擴展和釋放。
        
    - **自動模型調校 (Automatic Model Tuning)**: 透過智慧化的超參數搜尋（Hyperparameter Tuning），自動為您的模型找到最佳的參數組合，大幅提升模型準確率。
        
4. **部署與管理 (Deploy & Manage)**
    
    - **一鍵式部署 (One-click Deployment)**: 將訓練好的模型輕鬆部署為一個高可用性、自動擴展的 HTTPS 推論端點 (Inference Endpoint)。
        
    - **SageMaker MLOps**: 提供 SageMaker Pipelines 等工具，實現完整的機器學習維運 (MLOps)，自動化從資料準備到模型部署的整個流程。
        
    - **模型監控 (Model Monitor)**: 自動監控已部署模型的效能，偵測概念漂移 (Concept Drift)，確保模型在真實世界中的準確性。
        

### 2. AWS Lambda：事件驅動的無伺服器運算

**AWS Lambda** 是一種無伺服器運算服務 (Serverless Compute Service)，它讓您可以在不需佈建或管理任何伺服器的情況下執行程式碼。您只需上傳您的程式碼，Lambda 會在需要時自動且精確地擴展運算能力來滿足請求。

#### Lambda 的核心概念

- **無伺服器 (Serverless)**: 您無需關心底層的作業系統、硬體或伺服器擴展。AWS 會處理所有基礎設施的管理。
    
- **事件驅動 (Event-Driven)**: Lambda 函數通常由一個「事件」觸發。這個事件可以來自 200 多種 AWS 服務或您自己的應用程式。
    
    - **常見觸發器範例**:
        
        - 在 S3 儲存桶中上傳、修改或刪除檔案。
            
        - 透過 Amazon API Gateway 接收到 HTTP 請求。
            
        - Amazon DynamoDB 資料表中有新的資料寫入。
            
        - 按預定時間 (Cron job) 觸發。
            
- **按需付費 (Pay-per-use)**: Lambda 的計費單位是「執行時間」（以毫秒計）和「請求次數」。當您的程式碼沒有執行時，完全不產生任何費用，極具成本效益。
    
- **限制**: Lambda 為了維持其輕量和快速的特性，存在一些限制，例如最長執行時間為 15 分鐘、暫存空間 `/tmp` 有 512MB 到 10GB 的限制等。
    

---

### 3. 應用：結合 SageMaker 與 Lambda 實現手術影片智慧分析

手術影片的自動偵測與分割是一個典型的電腦視覺應用場景，目標是即時識別手術器械、定位人體器官或分割病灶區域。這對於術後分析、醫生培訓和開發手術輔助系統至關重要。

SageMaker 和 Lambda 的組合為此類應用提供了一個理想的雲端架構。

#### 整體工作流程

**階段一：模型訓練 (使用 SageMaker)**

1. **資料準備**: 收集大量手術影片，並將其分解為數萬甚至數百萬張影像畫格 (Frames)。
    
2. **資料標註**: 使用 **SageMaker Ground Truth**，由專業醫生或標註團隊在這些畫格上標註出感興趣的對象。
    
    - **物件偵測 (Detection)**: 用邊界框 (Bounding Box) 框出手術剪、鑷子、縫合針等器械。
        
    - **語意分割 (Segmentation)**: 用像素級的遮罩 (Mask) 描繪出肝臟、血管、腫瘤等區域的輪廓。
        
3. **模型訓練**:
    
    - 在 **SageMaker Notebook** 中編寫訓練腳本，使用 TensorFlow 或 PyTorch 等框架，並選擇一個適合的模型架構（例如用於偵測的 YOLOv8 或 Faster R-CNN，用於分割的 U-Net 或 Mask R-CNN）。
        
    - 啟動 **SageMaker Training Job**，利用其強大的 GPU 運算能力進行模型訓練。SageMaker 會自動處理資源分配和管理。
        
4. **模型部署**:
    
    - 訓練完成後，將效果最好的模型透過 **SageMaker Deployment** 功能部署成一個「即時推論端點 (Real-Time Inference Endpoint)」。這個 Endpoint 是一個受保護的 API，可以接收影像資料並即時回傳分析結果。
        

**階段二：自動化推論 (使用 Lambda + SageMaker)**

當需要分析一段新的手術影片時，以下自動化流程將被觸發：

1. **影片上傳**: 操作人員將新的手術影片檔案（如 `surgery_01.mp4`）上傳到指定的 Amazon S3 儲存桶 (`s3://surgical-videos/uploads/`)。
    
2. **Lambda 觸發**: S3 的 `ObjectCreated` 事件會自動觸發一個預先設定好的 Lambda 函數。
    
3. **Lambda 執行任務**: Lambda 函數收到觸發事件，從中解析出剛上傳影片的儲存桶名稱和檔案路徑。
    
4. **呼叫 SageMaker**:
    
    - **重要考量**: 由於 Lambda 有 15 分鐘的執行時間限制，直接在一個 Lambda 函數中處理數小時長的影片是不可行的。因此，需要一個更穩健的架構。
        
    - **生產級架構**: Lambda 的主要職責是啟動一個 **AWS Step Functions** 狀態機。Step Functions 是一個工作流程協調服務，它可以執行以下步驟： a. 使用 **AWS Batch** 或 **AWS Fargate**（更適合長時間運行的任務）將影片分割成每秒一幀的圖片。 b. 將所有圖片存儲到另一個 S3 位置（如 `s3://surgical-videos/frames/`）。 c. Step Functions 會並行地（Map State）為每一張圖片（或每批圖片）呼叫 SageMaker 推論端點。 d. SageMaker 端點接收圖片，執行偵測或分割，並回傳結果（如 JSON 格式的座標或遮罩資料）。 e. 將所有畫格的分析結果匯總，並儲存到 **Amazon DynamoDB** 資料庫或 S3 儲存桶 (`s3://surgical-videos/results/`) 中。
        
5. **結果儲存與後續處理**: 分析結果（例如，帶有邊界框的影片、每幀的 JSON 數據）被儲存起來，可供儀表板視覺化、進一步的臨床研究或警報系統使用。
    

#### 具體範例與程式碼

以下是一個**簡化版**的範例，旨在展示核心的互動邏輯。在這個範例中，我們假設 Lambda 被**單張圖片**的上傳觸發，然後呼叫 SageMaker 進行分析。這也是測試和驗證整個流程的常用方法。

**情境**: 當一張手術場景的圖片被上傳到 `s3://my-surgical-images/input/` 時，一個 Lambda 函數會被觸發，呼叫名為 `surgical-instrument-detector` 的 SageMaker 端點，並將結果存為 JSON 檔案到 `s3://my-surgical-images/output/`。

**Lambda 函數程式碼 (Python 3.9)**

您需要先在 Lambda 的執行角色 (IAM Role) 中添加權限，允許它讀取 S3 物件和呼叫 SageMaker 端點 (`sagemaker:InvokeEndpoint`)。

```Python
import json
import boto3
import os
import urllib.parse

# 初始化 AWS 客戶端
s3_client = boto3.client('s3')
# 從環境變數中讀取 SageMaker Endpoint 名稱，這是在 Lambda 設定中配置的
SAGEMAKER_ENDPOINT_NAME = os.environ['SAGEMAKER_ENDPOINT_NAME']
# 建立 SageMaker Runtime 客戶端
sagemaker_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    """
    當有新圖片上傳到 S3 時，此函數會被觸發。
    它會讀取圖片，呼叫 SageMaker 端點進行分析，並將結果存回 S3。
    """
    # 1. 從 S3 觸發事件中解析 Bucket 和 Key
    bucket = event['Records'][0]['s3']['bucket']['name']
    # 處理檔名中的特殊字元，例如空格或 '+'
    key = urllib.parse.unquote_plus(event['Records'][0]['s3']['object']['key'], encoding='utf-8')
    
    print(f"處理檔案: s3://{bucket}/{key}")
    
    # 為了呼叫 SageMaker，我們通常需要傳送圖片的原始位元組資料。
    # SageMaker 的內建演算法通常接受 'image/jpeg' 或 'image/png' 作為 ContentType。
    try:
        # 從 S3 讀取觸發事件的圖片檔案
        response = s3_client.get_object(Bucket=bucket, Key=key)
        image_data = response['Body'].read()
        
        # 取得圖片的 Content-Type
        content_type = response['ContentType']
        print(f"圖片 Content-Type: {content_type}")

    except Exception as e:
        print(f"讀取 S3 檔案時出錯: {e}")
        raise e

    # 2. 呼叫 SageMaker 推論端點
    try:
        # 向 SageMaker 端點發送請求
        # Body 是圖片的位元組資料
        # ContentType 告訴端點我們傳送的是什麼格式的資料
        # Accept 告訴端點我們希望接收什麼格式的回應，通常是 application/json
        response_sagemaker = sagemaker_runtime.invoke_endpoint(
            EndpointName=SAGEMAKER_ENDPOINT_NAME,
            ContentType=content_type,
            Body=image_data,
            Accept='application/json'
        )
        
        # 3. 處理來自 SageMaker 的回應
        # SageMaker 的回應 Body 是一個串流物件，需要讀取並解碼
        result_str = response_sagemaker['Body'].read().decode('utf-8')
        # 將 JSON 字串解析為 Python 字典
        result_json = json.loads(result_str)
        
        print("SageMaker 推論結果:")
        print(json.dumps(result_json, indent=2))

    except Exception as e:
        print(f"呼叫 SageMaker 端點時出錯: {e}")
        raise e

    # 4. 將分析結果儲存回 S3
    try:
        # 建立輸出的檔案名稱
        output_key = key.replace('input/', 'output/').rsplit('.', 1)[0] + '.json'
        
        s3_client.put_object(
            Bucket=bucket,
            Key=output_key,
            Body=json.dumps(result_json, indent=2), # 將結果格式化後寫入
            ContentType='application/json'
        )
        print(f"結果已成功儲存到: s3://{bucket}/{output_key}")

    except Exception as e:
        print(f"儲存結果到 S3 時出錯: {e}")
        raise e

    return {
        'statusCode': 200,
        'body': json.dumps(f'成功處理檔案 {key}，結果已儲存至 {output_key}')
    }
```

#### 如何部署與測試

1. **建立 Lambda 函數**: 在 AWS 管理控制台中，建立一個新的 Lambda 函數，選擇 Python 執行環境。
    
2. **配置環境變數**: 在 Lambda 的「設定」>「環境變數」中，新增一個鍵值對：
    
    - **Key**: `SAGEMAKER_ENDPOINT_NAME`
        
    - **Value**: `surgical-instrument-detector` (您部署的 SageMaker 端點名稱)
        
3. **設定觸發器**: 在 Lambda 的「函式概觀」中，點擊「新增觸發器」，選擇 S3。
    
    - **儲存桶**: 選擇 `my-surgical-images`。
        
    - **事件類型**: 選擇 `所有物件建立事件` (All object create events)。
        
    - **前置詞 (Prefix)**: 輸入 `input/`。這能確保只有上傳到 `input/` 資料夾的圖片才會觸發此函數。
        
4. **上傳程式碼與權限**: 將上面的 Python 程式碼複製到 Lambda 的程式碼編輯器中，並確保其 IAM 角色有足夠的權限。
    
5. **測試**: 將一張 JPG 或 PNG 格式的手術圖片上傳到 `s3://my-surgical-images/input/` 資料夾。上傳成功後，Lambda 函數會自動執行。您可以檢查 CloudWatch Logs 來查看 `print` 語句的輸出，並在 `s3://my-surgical-images/output/` 中找到生成的 JSON 結果檔案。
    

這個範例展示了 AWS 服務之間無縫整合的強大能力，透過事件驅動的架構，可以建構出反應迅速、高度自動化且無需管理伺服器的智慧分析流程。