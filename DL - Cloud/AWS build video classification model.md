
好的，這是一個非常棒的學習專案！從本地訓練轉移到雲端是 AI/ML 從業人員的關鍵技能。我們將以一個最簡單、最直接的方式，帶您走完在 AWS 上使用 VideoMAE-v2 進行影片理解任務的全過程。

由於影片**分割 (Segmentation)** 的資料集（需要像素級標籤）通常很龐大且處理複雜，對於初學者來說，一個更理想的起點是影片**分類 (Classification)**。分類任務可以讓您完全熟悉 AWS SageMaker 的工作流程（資料處理、腳本打包、遠端訓練、模型存儲），而無需立即陷入複雜的資料標註和損失函數中。

一旦您掌握了分類的工作流程，過渡到分割會容易得多，因為變動的部分主要在於模型頭部 (Model Head)、資料集和損失函數，而 AWS 的整體架構是完全一樣的。

因此，我們將以一個**影片分類**的專案作為「最簡單的例子」，並在最後說明如何將這個流程擴展到影片分割。

### 專案目標：訓練一個能辨識「舉手」動作的影片分類模型

- **模型骨幹 (Backbone)**: VideoMAE-v2
    
- **任務**: 影片分類 (Video Classification)
    
- **平台**: Amazon Web Services (AWS) SageMaker
    
- **資料集**: `rajans/raising-hands` - 一個非常小且適合教學的影片資料集，可以直接從 Hugging Face Hub 下載。
    

---

### 第一步：前置作業 - 設定您的 AWS 環境

這是您進入雲端世界的第一步，我們需要設定好「工作台」。

1. **登入 AWS 管理控制台**: 使用您的 AWS 帳號登入。
    
2. **選擇區域 (Region)**: 在右上角，選擇一個您要工作的區域，例如 `us-east-1` (N. Virginia) 或 `ap-northeast-1` (Tokyo)。建議選擇一個離您較近且 SageMaker 服務完整的區域。
    
3. **建立 IAM 角色 (IAM Role)**: 這是**最重要**的一步。您需要建立一個「身份」，讓 SageMaker 有權限存取您帳戶中的其他資源（主要是 S3 儲存桶）。
    
    - 前往 **IAM** 服務。
    - https://us-east-1.console.aws.amazon.com/iam/home?region=us-east-2#/home
        
    - 在左側選擇「角色 (Roles)」，點擊「建立角色 (Create role)」。
        
    - 在「選擇受信任的實體 (Select trusted entity)」中，選擇 **AWS service**。(Allow AWS services like EC2, Lambda, or others to perform actions in this account)
        
    - 在「使用案例 (Use case)」中，選擇 **SageMaker**。
        
    - 點擊「下一步 (Next)」。
        
    - 在權限頁面，它會預設選中 `AmazonSageMakerFullAccess`，這就夠了。點擊「下一步 (Next)」。
        
    - 給您的角色取一個名字，例如 `SageMaker-Video-ExecutionRole`，然後點擊「建立角色 (Create role)」。
        
    - **記下這個角色的 ARN** (Amazon Resource Name)，它看起來像 `arn:aws:iam::123456789012:role/SageMaker-Video-ExecutionRole`。雖然 SageMaker 通常能自動找到它，但知道在哪裡找總是好的。
        
4. **啟動 SageMaker Studio**: SageMaker Studio 是一個網頁版的整合開發環境 (IDE)，您可以在這裡寫程式碼、管理專案。
    
    - 前往 **Amazon SageMaker** 服務。
    - https://us-east-2.console.aws.amazon.com/sagemaker/home?region=us-east-2#/studio-landing
        
    - 在左側選擇 **Studio**。
        
    - 如果這是您第一次使用，系統會引導您設定一個使用者設定檔。選擇「快速設定 (Quick start)」，並在「執行角色 (Execution role)」處，選擇您剛剛建立的 `SageMaker-Video-ExecutionRole`。
        
    - 設定完成後，點擊「啟動 Studio (Launch Studio)」。這可能需要幾分鐘的時間來準備環境。
        

---

### 第二步：在 SageMaker Studio 中建立並執行 Notebook

當 Studio 啟動後，您會看到一個類似 JupyterLab 的介面。我們將在這裡完成所有工作。

1. **建立新的 Notebook**:
    
    - 在左側檔案瀏覽器中，點擊 `+` 按鈕或在 `File` -> `New` 中選擇 `Notebook`。
        
    - 當跳出「選擇 Kernel」的視窗時，選擇 **PyTorch 2.0.0 Python 3.10 GPU Optimized** 或類似的 PyTorch GPU Kernel。這表示您的 Notebook 執行環境已經預裝了 PyTorch 和 CUDA。
        
2. **在 Notebook 中撰寫程式碼（開發與測試階段）**: 我們將直接在 Notebook 中安裝套件、下載資料、測試模型，確保所有程式碼都正確無誤，然後再將其打包成訓練任務。
    
    **Cell 1: 安裝必要的套件**
    
    Python
    
    ```
    # SageMaker Studio 預裝了很多套件，但我們需要確保有最新版的
    # transformers、datasets 和 evaluate (用於 Hugging Face 生態系)
    !pip install -q "transformers>=4.38.0" "datasets[video]" "evaluate" "accelerate"
    ```
    
    **Cell 2: 載入資料集** 這一步展示了 Hugging Face `datasets` 函式庫的強大之處，我們無需手動下載和上傳。
    
    Python
    
    ```
    from datasets import load_dataset
    
    # 這個資料集非常小，包含 "raising_hand" 和 "not_raising_hand" 兩類影片
    dataset = load_dataset("rajans/raising-hands")
    
    # 查看一下資料集結構
    print(dataset)
    
    # 將標籤從數字轉換成可讀的文字
    id2label = {id: label for id, label in enumerate(dataset["train"].features["label"].names)}
    label2id = {label: id for id, label in id2label.items()}
    print(f"ID to Label mapping: {id2label}")
    ```
    
    **Cell 3: 資料預處理** 影片資料需要經過特殊處理，才能輸入到 VideoMAE 模型中。這包括幀取樣、尺寸調整和正規化。`AutoImageProcessor` 會幫我們處理好這一切。
    
    Python
    
    ```
    from transformers import AutoImageProcessor
    import torch
    
    # 從模型 checkpoint 載入對應的圖片/影片處理器
    model_checkpoint = "MCG-NJU/videomae-v2-base-finetuned-kinetics-400"
    image_processor = AutoImageProcessor.from_pretrained(model_checkpoint)
    
    # 建立一個整理函式 (collate_fn)，將每個影片樣本處理成一個批次 (batch)
    def collate_fn(examples):
        # 移除影片檔案路徑，因為它們無法轉換成 Tensor
        video_paths = [example.pop("video") for example in examples]
    
        # 從每個影片中提取 16 幀
        # 這是 VideoMAE 的標準輸入格式
        pixel_values = [image_processor(list(video.iter_frames()), return_tensors="pt").pixel_values for video in video_paths]
    
        # 將多個影片的 pixel_values 堆疊成一個批次
        pixel_values = torch.stack(pixel_values)
    
        # 處理標籤
        labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    
        return {"pixel_values": pixel_values, "labels": labels}
    ```
    
    **Cell 4: 定義模型、訓練參數與評估指標**
    
    Python
    
    ```
    from transformers import AutoModelForVideoClassification, TrainingArguments, Trainer
    import evaluate
    import numpy as np
    
    # 載入預訓練的 VideoMAE-v2 模型，並將其頭部修改為我們的分類任務 (2個類別)
    model = AutoModelForVideoClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,  # 忽略預訓練頭部和我們新頭部的尺寸不匹配問題
    )
    
    # 載入準確率評估指標
    accuracy = evaluate.load("accuracy")
    
    def compute_metrics(eval_pred):
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    
    # 定義訓練參數
    training_args = TrainingArguments(
        output_dir="./videomae-v2-raising-hands", # 暫時的輸出目錄
        per_device_train_batch_size=2, # 根據您的 Notebook 執行個體記憶體調整
        per_device_eval_batch_size=2,
        num_train_epochs=3, # 為了快速測試，只訓練3個 epoch
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        learning_rate=5e-5,
    )
    
    # 建立 Hugging Face Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=image_processor, # 雖然是圖片處理器，但傳給 tokenizer 參數
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )
    
    # 在 Notebook 執行個體上進行一個小規模的測試訓練，確保一切正常
    print("--- Starting a small test training run on the notebook instance ---")
    trainer.train()
    ```
    
    執行到這裡，如果訓練迴圈能夠正常啟動並運行，就代表您的程式碼邏輯是正確的！接下來，我們將把它部署到更強大的專用訓練伺服器上。
    

---

### 第三步：打包程式碼並啟動 SageMaker 遠端訓練任務

在 Notebook 上直接訓練只適合小規模測試。對於正式訓練，我們應該使用 SageMaker Training Job，它會在背景啟動一台專門的、更強大的 GPU 伺服器來執行任務，結束後自動關閉，非常經濟高效。

1. **將訓練邏輯打包成 `train.py` 腳本**: 我們將剛才在 Notebook 中驗證過的程式碼，整理成一個獨立的 Python 檔案。在 Studio Notebook 的同一個 Cell 中，使用 `%%writefile` 這個“魔法命令”可以直接建立檔案。
    
    **Cell 5: 建立 `train.py` 檔案**
    
    Python
    
    ```
    %%writefile train.py
    
    import argparse
    import os
    from datasets import load_dataset
    from transformers import (
        AutoImageProcessor,
        AutoModelForVideoClassification,
        TrainingArguments,
        Trainer,
    )
    import torch
    import evaluate
    import numpy as np
    
    def main(args):
        # 1. 載入資料集
        dataset = load_dataset("rajans/raising-hands")
        id2label = {id: label for id, label in enumerate(dataset["train"].features["label"].names)}
        label2id = {label: id for id, label in id2label.items()}
    
        # 2. 載入處理器和模型
        image_processor = AutoImageProcessor.from_pretrained(args.model_name)
        model = AutoModelForVideoClassification.from_pretrained(
            args.model_name,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    
        # 3. 定義 Collate Function 和評估指標
        def collate_fn(examples):
            video_paths = [example.pop("video") for example in examples]
            pixel_values = [image_processor(list(video.iter_frames()), return_tensors="pt").pixel_values for video in video_paths]
            pixel_values = torch.stack(pixel_values)
            labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
            return {"pixel_values": pixel_values, "labels": labels}
    
        accuracy = evaluate.load("accuracy")
        def compute_metrics(eval_pred):
            predictions = np.argmax(eval_pred.predictions, axis=1)
            return accuracy.compute(predictions=predictions, references=eval_pred.label_ids)
    
        # 4. 定義訓練參數
        # SageMaker 會將模型輸出儲存在 /opt/ml/model
        training_args = TrainingArguments(
            output_dir=args.output_dir,
            per_device_train_batch_size=args.train_batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            num_train_epochs=args.epochs,
            logging_steps=10,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            learning_rate=args.learning_rate,
        )
    
        # 5. 建立並啟動 Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=image_processor,
            compute_metrics=compute_metrics,
            data_collator=collate_fn,
        )
        trainer.train()
    
        # 將訓練好的最佳模型儲存到指定路徑
        trainer.save_model(args.model_dir)
    
    if __name__ == "__main__":
        parser = argparse.ArgumentParser()
        # 從 SageMaker 傳入的參數
        parser.add_argument("--model_name", type=str, default="MCG-NJU/videomae-v2-base-finetuned-kinetics-400")
        parser.add_argument("--epochs", type=int, default=10) # 在遠端機器上可以訓練更久
        parser.add_argument("--train_batch_size", type=int, default=4)
        parser.add_argument("--eval_batch_size", type=int, default=4)
        parser.add_argument("--learning_rate", type=float, default=5e-5)
        # SageMaker 環境提供的路徑
        parser.add_argument("--output_dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])
        parser.add_argument("--model_dir", type=str, default=os.environ["SM_MODEL_DIR"])
    
        args = parser.parse_args()
        main(args)
    ```
    
2. **使用 SageMaker SDK 啟動訓練任務**: 現在，我們回到 Notebook，用幾行程式碼來設定並啟動這個遠端訓練任務。
    
    **Cell 6: 啟動遠端訓練**
    
    Python
    
    ```
    import sagemaker
    from sagemaker.huggingface import HuggingFace
    import sagemaker
    
    # 獲取 SageMaker Session 和 IAM Role
    sess = sagemaker.Session()
    sagemaker_role = sagemaker.get_execution_role()
    
    # 這是您的 S3 儲存桶，如果沒有會自動建立一個預設的
    s3_bucket = sess.default_bucket()
    print(f"S3 Bucket: {s3_bucket}")
    print(f"IAM Role ARN: {sagemaker_role}")
    
    # 定義要傳遞給 train.py 的超參數
    hyperparameters = {
        'model_name': 'MCG-NJU/videomae-v2-base-finetuned-kinetics-400',
        'epochs': 15,
        'train_batch_size': 4,
        'learning_rate': 5e-5
    }
    
    # 建立一個 HuggingFace Estimator
    huggingface_estimator = HuggingFace(
        entry_point          = 'train.py',       # 我們的訓練腳本
        source_dir           = './',             # 腳本所在的目錄
        instance_type        = 'ml.g4dn.xlarge', # **選擇 GPU 執行個體**，這是性價比不錯的選擇
        instance_count       = 1,                # 使用 1 台機器
        role                 = sagemaker_role,   # IAM 角色
        framework_version    = '2.1',            # Hugging Face 建議的 PyTorch 版本
        py_version           = 'py310',          # Python 版本
        hyperparameters      = hyperparameters,
    )
    
    # 啟動訓練！
    # 因為我們的資料集是從 Hugging Face Hub 即時下載的，所以 fit() 中不需要傳入 data 參數
    print("--- Launching remote SageMaker Training Job ---")
    huggingface_estimator.fit()
    print("--- Training Job launched successfully! You can monitor it in the AWS SageMaker console. ---")
    ```
    

### 第四步：監控與結果

- **監控**: 執行 `.fit()` 後，您的 Notebook 會即時輸出遠端伺服器上的 Log。您也可以前往 AWS 管理控制台 -> SageMaker -> 訓練 (Training) -> 訓練任務 (Training jobs) 中看到您的任務正在運行，並可以查看 CPU/GPU 使用率、日誌等詳細資訊。
    
- **結果**: 訓練完成後，訓練好的模型（`pytorch_model.bin`, `config.json` 等檔案）會被 SageMaker 自動打包成 `model.tar.gz` 並儲存到您的 S3 儲存桶中，路徑類似 `s3://sagemaker-us-east-1-123456789012/huggingface-pytorch-training-YYYY-MM-DD-HH-MM-SS-SSS/output/model.tar.gz`。您可以隨時下載它用於推論或進一步的開發。
    

---

### 下一步：如何走向影片分割 (Video Segmentation)？

恭喜您！完成以上步驟後，您已經掌握了在 AWS 上訓練一個複雜的 Transformer 影片模型的核心流程。現在要走向影片分割，您需要調整以下幾個部分，但 AWS 的架構（Estimator, Training Job, S3 I/O）是完全不變的：

1. **資料集**: 您需要一個帶有像素級標註的影片資料集，例如 [DAVIS](https://davischallenge.org/) 或 [VSPW (Video Semantic Parsing in the Wild)](https://www.google.com/search?q=https://github.com/VSPW-Dataset/VSPW_480p)。資料預處理的 `collate_fn` 會變得更複雜，因為您不僅要處理影片幀，還要同步處理對應的 Mask 標籤，並對它們進行相同的資料增強（如翻轉、裁切）。
    
2. **模型架構**: 您不能再直接使用 `AutoModelForVideoClassification`。您需要一個自訂的模型類別。這個類別會：
    
    - 載入 VideoMAE-v2 作為編碼器 (Encoder) 來提取特徵。
        
    - 在編碼器的基礎上，加上一個解碼器 (Decoder)，例如像 U-Net 那樣的結構，來將特徵上採樣回原始解析度，並預測每個像素的類別。
        
3. **損失函數**: 分類任務使用交叉熵損失 (Cross-Entropy Loss)。對於分割，您通常會使用 **Dice Loss**, **Focal Loss**，或者它們與交叉熵的組合，以更好地處理類別不平衡（背景像素遠多於前景像素）的問題。
    
4. **訓練腳本 `train.py`**: 您需要修改 `train.py` 來引入新的模型類別、新的資料集處理邏輯和新的損失函數。但啟動這個腳本的 `sagemaker_launcher.ipynb` 幾乎不需要改變！