

|                                            |     |
| ------------------------------------------ | --- |
| [[#### 1. 初步架構分析]]                         |     |
| [[#### 2. 具體架構分析]]                         |     |
| [[#### 3. Deployment跟multi environment設定]] |     |
| [[#### 4. 使用Docker]]                       |     |
| [[#### 5. 所有方案(Native, Cloud跟其他)比較]]       |     |
| [[#### 6. 現在開發核心模組未來再決定部署方式]]              |     |
| [[#### 7. 實作cli跟server模式]]                 |     |
|                                            |     |
|                                            |     |



#### 1. 初步架構分析

```
我想開發一套python based imaging processing system在windows platform並支援GPU. 這是一套對一個watch由自動camra system拍攝錶面不同部位約90張images (2048 x 2048)傳入, 並針對每張image有不同的imaging processing task分析上面的features並儲存結果可能包括segmentation masks及這種values並傳出某個matrics, 其中1/3有些可能需要segmentation AI models or OCR辨認上面文字. 最後整合90幾個這些matrics並和某些reference matrics做比對, 最後確認這watch是真的或假的. 這90幾個imaging processing task有的可能需要不同或獨立的的python environment因為可能有的library會有版本衝突. 

一次work可能需要處理500個watch, 所以會有個images folder裡面有500個sub folders各代表一個watch, 每個subfolder裡面有90張images. 另外還有一個folder儲存500個mat files, 每個mat file儲存一個watch裡面每個不同部位的image, 對應到subfolder裡面的image path. 這樣的系統要如何設計, 是否需要用到multi-thread or GPU並行處理這些imaging processing tasks, 如何維持系統穩定高效不會有memory或各種問題. 並詳細分析可能需要注意的地方及如何擴充

90個imaging processing task目前寫成是獨立python py file, 譬如run_task1_process.py, run_task2_process.py, run_task3_process.py....等等負責watch不同部位image的處理並會輸出某個matrics以及有些會輸出segmentation masks, 這些segmentation masks或其他的images可能會儲存起來到某個folders供後續其他tasks分析. 所以要設計這樣的系統是不是要有個main.py當入口, 而之後有個機制可以handle多個或專屬的不同python environment可以給對應的imaging processing tasks, 又要支援multi-thread or GPU並行處理. 請提供example code 並具體舉例說明
```

這是一個典型的高吞吐量 (High-Throughput)、多環境 (Multi-Environment) 且異質計算 (Heterogeneous Computing, CPU+GPU) 的系統架構問題。針對您的需求：**Windows 平台、500 支手錶/批、每支 90 個 Tasks、混合不同 Python 環境、GPU/CPU 並行處理**，我建議採用 **「Orchestrator (指揮官) + Subprocess Worker」** 的架構。

這樣的設計可以確保：
1. **環境隔離**：透過 `subprocess` 調用不同路徑的 Python Interpreter，完美解決版本衝突。
2. **記憶體安全**：每個 Task 跑完後 Process 結束，記憶體 (VRAM/RAM) 強制釋放，避免 Memory Leak。
3. **GPU 排程**：透過 Semaphore 機制控制同時存取 GPU 的 Task 數量，避免 OOM (Out Of Memory)。

---

### 一、 系統架構設計

#### 1. 核心模組

- **Main Controller (Orchestrator):** 負責讀取 Watch List，管理 Thread Pool，並根據 Config 分派任務。
- **Task/Env Manager:** 一個設定檔 (YAML/JSON)，定義 Task 1 到 Task 90 需要用哪個 Python 環境、是否需要 GPU、以及依賴關係 (Dependency)。
- **Worker Wrapper:** 負責組裝 command line arguments，啟動子進程 (`subprocess`) 執行具體的 `run_task_X.py`。
- **Data Aggregator:** 當一支手錶的所有 Tasks 完成後，收集所有產出的 metrics 並進行真偽比對。

#### 2. 並行策略 (Concurrency Strategy)

由於 Windows 的 Process 建立開銷較大，且 GPU 資源有限，建議採用 **混合式並行**：

- **Watch Level (手錶層級):** 循序處理或小批量並行 (例如一次處理 2-4 支錶)，取決於磁碟 I/O 速度。
- **Task Level (任務層級):** 這是並行的主力。
    
    - **CPU Tasks:** 使用 `ThreadPoolExecutor` 管理，可以開較多 Threads (例如 CPU 核心數 * 1.5)，每個 Thread 去喚起一個 `subprocess`。
    - **GPU Tasks:** 使用 `threading.Semaphore` 限制併發數 (例如限制同時只能有 1-2 個 GPU Task 執行)，以免顯卡記憶體爆掉。

---

### 二、 系統實作 Example Code

這個範例包含三個部分：設定檔結構、Task 範例、以及最重要的 Main Orchestrator。

#### 1. 設定檔 (`config.py` 或 YAML)

定義每個 Task 的屬性，這是系統靈活擴充的關鍵。
```
# config.py

# 定義不同環境的 Python 執行檔路徑
ENV_PATHS = {
    "base": r"C:\Envs\base_env\python.exe",
    "ocr_env": r"C:\Envs\ocr_env\python.exe",  # 專門跑 paddleOCR/Tesseract
    "seg_env": r"C:\Envs\seg_env\python.exe",  # 專門跑 PyTorch/Segmentation
}

# 定義 Task 列表
# order: 執行順序 (若 tasks 有相依性，譬如 mask 產生後才能做分析，需分層級)
# use_gpu: 是否佔用 GPU 資源
TASKS_CONFIG = {
    "task1": {
        "script": "tasks/run_task1_process.py",
        "env": "base",
        "use_gpu": False,
        "description": "Basic Image Stats"
    },
    "task2": {
        "script": "tasks/run_task2_segmentation.py",
        "env": "seg_env",
        "use_gpu": True,
        "description": "Dial Segmentation"
    },
    "task3": {
        "script": "tasks/run_task3_ocr.py",
        "env": "ocr_env",
        "use_gpu": True, # OCR 也可以用 GPU
        "description": "Serial Number OCR"
    },
    # ... 擴充至 Task 90
}

# GPU 資源鎖 (假設只有一張顯卡，且 VRAM 只能同時跑 2 個模型)
GPU_CONCURRENCY_LIMIT = 2 
# CPU 最大並行 Process 數
CPU_WORKER_LIMIT = 8
```
#### 2. 單一 Task 範例 (`tasks/run_task1_process.py`)

所有的 Task 腳本都應設計為「接收參數 -> 處理 -> 輸出 JSON/File」。
```
# tasks/run_task1_process.py
import argparse
import json
import os
import sys

# 模擬處理
def process(image_path, output_dir):
    # 實際邏輯寫在這裡
    # print(f"Processing {image_path}...") 
    
    # 模擬計算出的 Metrics
    result_metrics = {
        "brightness": 128.5,
        "sharpness": 0.95,
        "task_name": "task1"
    }
    
    # 將結果存成 JSON 讓 Main Process 讀取
    out_file = os.path.join(output_dir, "task1_result.json")
    with open(out_file, 'w') as f:
        json.dump(result_metrics, f)
        
    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    
    try:
        process(args.image, args.out_dir)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
```
#### 3. Main Orchestrator (`main.py`)
這是系統的核心入口。
```
import os
import subprocess
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import TASKS_CONFIG, ENV_PATHS, GPU_CONCURRENCY_LIMIT, CPU_WORKER_LIMIT
import scipy.io

# GPU Semaphore: 限制同時存取 GPU 的任務數量
gpu_semaphore = threading.Semaphore(GPU_CONCURRENCY_LIMIT)

def run_single_task(watch_id, task_name, task_info, image_path, output_dir):
    """
    執行單個任務的 Wrapper
    """
    python_exe = ENV_PATHS[task_info["env"]]
    script_path = task_info["script"]
    
    use_gpu = task_info["use_gpu"]
    
    # 建構指令
    cmd = [
        python_exe, 
        script_path, 
        "--image", image_path, 
        "--out_dir", output_dir
    ]
    
    # 如果需要 GPU，先獲取鎖
    if use_gpu:
        gpu_semaphore.acquire()
    
    try:
        # 執行外部 Python Process
        # capture_output=True 可以攔截 stdout/stderr 寫入 log
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # print(f"[{watch_id}] {task_name} Finished.")
        return {"status": "success", "task": task_name, "watch_id": watch_id}
        
    except subprocess.CalledProcessError as e:
        print(f"[{watch_id}] {task_name} FAILED: {e.stderr}")
        return {"status": "failed", "task": task_name, "error": e.stderr}
        
    finally:
        # 釋放 GPU 鎖
        if use_gpu:
            gpu_semaphore.release()

def process_watch(watch_folder, mat_file_path, output_root):
    """
    處理單支手錶的所有 Tasks
    """
    watch_id = os.path.basename(watch_folder)
    watch_out_dir = os.path.join(output_root, watch_id)
    os.makedirs(watch_out_dir, exist_ok=True)
    
    # 1. 讀取 .mat 檔取得 metadata (假設 mat 檔裡有 image mapping)
    # mat_data = scipy.io.loadmat(mat_file_path)
    # 簡化範例：假設我們知道 image 路徑規則
    
    tasks_to_run = []
    
    # 2. 準備任務清單
    for task_name, info in TASKS_CONFIG.items():
        # 根據 task 邏輯找到對應的圖片 (這裡假設 image 名稱與 task 對應，需根據實際 mat 檔邏輯修改)
        # image_path = mat_data['images'][task_name] 
        target_image = os.path.join(watch_folder, f"image_for_{task_name}.jpg") # 範例路徑
        
        if not os.path.exists(target_image):
            # 如果圖片不存在，可能用預設圖或跳過
            target_image = os.path.join(watch_folder, "default.jpg")

        tasks_to_run.append((task_name, info, target_image))

    results = []
    
    # 3. 使用 ThreadPool 並行執行 Subprocess
    # 注意：這裡的 Thread 只是在「等待」Subprocess 結束，不會佔用 Python GIL 導致運算瓶頸
    with ThreadPoolExecutor(max_workers=CPU_WORKER_LIMIT) as executor:
        future_to_task = {
            executor.submit(
                run_single_task, 
                watch_id, 
                t_name, 
                t_info, 
                img_path, 
                watch_out_dir
            ): t_name for t_name, t_info, img_path in tasks_to_run
        }
        
        for future in as_completed(future_to_task):
            results.append(future.result())

    # 4. 聚合結果 (Aggregation)
    final_metrics = {}
    for task_name in TASKS_CONFIG.keys():
        json_path = os.path.join(watch_out_dir, f"{task_name}_result.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
                final_metrics.update(data)
    
    return final_metrics

def main():
    root_images_dir = r"C:\Data\Watches_Images"
    root_mat_dir = r"C:\Data\Watches_Mats"
    output_root = r"C:\Data\Results"
    
    # 取得所有手錶清單
    watch_folders = [f.path for f in os.scandir(root_images_dir) if f.is_dir()]
    
    print(f"Found {len(watch_folders)} watches to process.")

    # 針對每一支手錶進行處理
    # 如果硬體夠強，這裡也可以再套一層 ThreadPoolExecutor 並行處理多支手錶
    for watch_folder in watch_folders:
        watch_id = os.path.basename(watch_folder)
        mat_path = os.path.join(root_mat_dir, f"{watch_id}.mat")
        
        print(f"=== Processing Watch: {watch_id} ===")
        metrics = process_watch(watch_folder, mat_path, output_root)
        
        # 5. 真偽判定邏輯 (Authentication)
        # is_real = compare_with_reference(metrics)
        # print(f"Result for {watch_id}: {is_real}")
        print(f"Finished {watch_id}. Metrics keys: {list(metrics.keys())}")

if __name__ == "__main__":
    main()
```

### 三、 詳細分析與注意事項

#### 1. 穩定性與記憶體管理 (Stability & Memory)

- **Process Isolation (進程隔離):** 這是此方案最大的優點。因為 Image Processing (特別是 OpenCV 和 PyTorch) 經常會有 C++ 層級的 Memory Leak。如果全部跑在一個 Python Process 裡，處理到第 200 支手錶時 RAM 可能會爆滿導致 Crash。使用 `subprocess.run`，每次 Task 結束，OS 會回收所有資源，**保證系統可以長時間穩定運行**。
- **Timeout 機制:** 在 `subprocess.run` 中務必加入 `timeout` 參數（例如 60 秒）。如果某個 Task 卡死（Deadlock），主程式可以直接 Kill 掉它並記錄 Error，而不是整個系統停擺。

#### 2. GPU 資源調度 (GPU Scheduling)

- **顯存碎片化:** PyTorch 預設會佔用大量顯存。如果 Task 1 跑完沒釋放，Task 2 就會 OOM。Process 架構天然解決了釋放問題。
- **Semaphore:** 程式碼中的 `gpu_semaphore` 至關重要。你必須根據你的 GPU VRAM 大小 (例如 24GB RTX 3090 vs 80GB A100) 來調整 `GPU_CONCURRENCY_LIMIT`。如果使用 Segmentation 模型 (如 DINOv3, SAM)，通常一個 Process 會吃掉 2-4GB VRAM。

#### 3. 擴充性 (Scalability)

- **新增 Task:** 只需寫好新的 `.py` 檔，並在 `config.py` 中加入一行設定即可，完全不影響主程式邏輯。
- **多機擴充:** 如果單機速度不夠，這個架構很容易改成「分布式」。只需將 `config.py` 改成從 Redis Queue 讀取任務，多台電腦執行 Worker 即可。

#### 4. 資料依賴與 DAG (Directed Acyclic Graph)

目前的範例假設 Task 之間相對獨立。如果 **Task B 需要 Task A 的 Segmentation Mask**：

- **簡單解法:** 在 `process_watch` 內部，將 Tasks 分為 `Stage 1 (Independent)`, `Stage 2 (Dependent)`。先跑完 Stage 1 的所有 Thread，再跑 Stage 2。
- **複雜解法:** 引入 DAG 排程庫 (如 Airflow, Prefect)，但對於單機應用來說可能過度設計。建議用「分階段 (Stages)」的方式處理即可。

#### 5. Windows 平台的特殊限制

- **路徑長度:** Windows 有 260 字元路徑限制。建議在 Registry 開啟 Long Paths Enabled，或者將 Output Folder 設在磁碟根目錄附近 (如 `D:\Data\`).
- **File Locking:** Windows 對檔案鎖定很嚴格。確保 Task 寫入檔案後確實 `close()`，否則下一個 Task 讀取時會報錯 `Permission Denied`。

### 四、 總結

要建立這套系統，您的下一步應該是：

1. **建立環境清單**：確認 90 個 Tasks 歸類為哪幾種環境 (例如：`legacy_cv`, `modern_pytorch`, `ocr_paddle`)。
2. **標準化 I/O**：規定所有 Task 的 Python Script 必須接受 `--image` 和 `--out_dir` 參數，並輸出標準 JSON。
3. **Benchmark**：先拿 5 支手錶跑跑看，觀察 GPU VRAM 佔用量，據此調整 `GPU_CONCURRENCY_LIMIT`。




#### 2. 具體架構分析

```
請按照之前你建議的「Orchestrator (指揮官) + Subprocess Worker」 的架構請針對目前我的repo提供建議或修改, 以達到最穩定及優化的project架構. 目前我的Project repo的架構如下, 在root folder有90個imaging processing task目前寫成是獨立python py file, 譬如run_task1_process.py, run_task2_process.py, run_task3_process.py....等等. 在root folder有幾個sub folders包括App, config, env, images, output. 

App folder目前放我完成的ui包括有ProveWatchApp.py這是ui的入口, 除此之外還有main.ui, resources.qrc. 在App folder之下的sub folder modules有ui.main.py, ui_functions.py等等. 另外執行ProveWatchApp.py可以打開ui並有camera extraction以及image extraction可以直接從camera傳入90張不同部位的images, 或可以從以前已經儲存的watch images選取要分析的images. 所以接下來應該就是接上類似Orchestrator開始分配執行不同的imaging processing tasks. 但也希望Orchestrator也可以不經由roveWatchApp.py而直接可以用cmd執行並分配執行不同的imaging processing tasks. 所以可以理解有兩個不同的入口, 經由ui或者沒有ui.

config folder統一放置這些imaging processing tasks所需要的config yaml files. env folder也統一放要建立python environment 的yaml files(conda+pip). images則是之前已經extract的images. output則是輸出的output.  

請提供example code 並具體舉例說明 及中文詳細說明並分析優缺點
```


這是一個非常好的架構規劃問題。您現在面臨的是如何將 **「現有的分散式腳本」** 與 **「GUI 應用程式」** 以及 **「批次處理需求」** 完美縫合。

為了達到高穩定性與優化，關鍵在於 **「解耦 (Decoupling)」**。核心邏輯（Orchestrator）必須獨立於 UI 之外，這樣它才能同時服務 CLI（命令列）和 GUI（PySide6）。

以下是針對您目前的 Repo 結構進行的**重構建議**與**實作代碼**。

---

### 一、 建議的專案目錄結構 (Refined Project Structure)

目前您的 90 個 tasks 散落在 root folder 會顯得雜亂且難以維護。建議建立一個 `core` 資料夾放核心邏輯，並將 tasks 移入專屬資料夾。
```
Project_Root/
├── App/                     # UI 相關代碼
│   ├── modules/
│   ├── resources.qrc
│   ├── ProveWatchApp.py     # UI 入口 (Entry Point 1)
│   └── worker_thread.py     # [新增] 負責連接 UI 與 Orchestrator 的橋樑
├── config/                  # 設定檔
│   ├── system_config.yaml   # [新增] 定義 Python 環境路徑、GPU 設定
│   ├── tasks_map.yaml       # [新增] 定義 90 個 Tasks 的執行順序、環境與參數
│   └── task_configs/        # 各個 Task 獨有的參數 (如閾值等)
├── core/                    # [新增] 核心邏輯層
│   ├── __init__.py
│   └── orchestrator.py      # [核心] 指揮官，負責調度 Subprocess
├── env/                     # Conda environments YAML
├── images/                  # 輸入圖片庫
├── output/                  # 輸出結果
├── tasks/                   # [移動至此] 將 root 的 90 個 py 檔移進來
│   ├── run_task1_process.py
│   ├── ...
│   └── utils.py             # 共用的 image processing helper functions
├── main_cli.py              # [新增] 純命令列入口 (Entry Point 2)
└── requirements.txt         # Root 環境依賴 (只負責 UI 和 Orchestrator)
```

### 二、 核心模組實作 (Core Implementation)

這個部分是系統的心臟，不依賴 UI，也不依賴特定的 Watch，只負責「執行邏輯」。
#### 1. 設定檔範例 (`config/tasks_map.yaml`)
將任務的依賴關係與環境配置外部化。
```
# config/tasks_map.yaml
global:
  max_cpu_workers: 8
  max_gpu_workers: 2

envs:
  base: "C:/Envs/base_env/python.exe"
  ocr: "C:/Envs/ocr_env/python.exe"
  segmentation: "C:/Envs/seg_env/python.exe"

tasks:
  - name: "dial_analysis"
    script: "tasks/run_task1_process.py"
    env: "base"
    use_gpu: false
    input_image_pattern: "dial_*.jpg" # 告訴系統這個 task 吃哪張圖

  - name: "logo_ocr"
    script: "tasks/run_task3_ocr.py"
    env: "ocr"
    use_gpu: true
    input_image_pattern: "crown_*.jpg"
```

#### 2. 指揮官 (`core/orchestrator.py`)
這是最重要的 Class。它封裝了 `subprocess` 的複雜度，並提供 callbacks 讓 UI 可以顯示進度。
```
# core/orchestrator.py
import os
import subprocess
import yaml
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class Orchestrator:
    def __init__(self, config_path="config/tasks_map.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.gpu_semaphore = threading.Semaphore(self.config['global']['max_gpu_workers'])
        self.env_paths = self.config['envs']

    def _run_single_task(self, task_conf, watch_dir, output_dir):
        """
        執行單一 Task (會在子線程中執行)
        """
        script_path = os.path.abspath(task_conf['script'])
        python_exe = self.env_paths[task_conf['env']]
        
        # 尋找對應圖片 (簡單範例)
        # 實際應用可加入更複雜的 mapping 邏輯
        img_name = task_conf.get('input_image_pattern', 'default.jpg').replace('*', '') 
        # 這裡只是簡化邏輯，實務上可能要做 glob search
        image_path = os.path.join(watch_dir, img_name)

        cmd = [
            python_exe, script_path,
            "--image", image_path,
            "--out_dir", output_dir,
            "--config", f"config/task_configs/{task_conf['name']}.yaml" # 傳入細部參數
        ]

        use_gpu = task_conf.get('use_gpu', False)
        
        start_time = datetime.now()
        result = {
            "task": task_conf['name'],
            "status": "pending",
            "logs": ""
        }

        if use_gpu:
            self.gpu_semaphore.acquire()
        
        try:
            # 啟動子進程
            proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
            result["status"] = "success"
            result["logs"] = proc.stdout
        except subprocess.CalledProcessError as e:
            result["status"] = "failed"
            result["logs"] = e.stderr
        except Exception as e:
            result["status"] = "error"
            result["logs"] = str(e)
        finally:
            if use_gpu:
                self.gpu_semaphore.release()
            
            result["duration"] = (datetime.now() - start_time).total_seconds()
            
            # 檢查是否有 result json 產出
            json_out = os.path.join(output_dir, f"{task_conf['name']}_result.json")
            if os.path.exists(json_out):
                with open(json_out, 'r') as f:
                    result["data"] = json.load(f)
            
            return result

    def process_watch(self, watch_dir, output_root, progress_callback=None):
        """
        處理單支手錶 (Entry Point Logic)
        progress_callback: function(current, total, message)
        """
        watch_id = os.path.basename(watch_dir)
        watch_out = os.path.join(output_root, watch_id)
        os.makedirs(watch_out, exist_ok=True)

        tasks = self.config['tasks']
        total_tasks = len(tasks)
        completed_count = 0
        results_summary = {}

        # 使用 ThreadPool 平行處理 tasks
        with ThreadPoolExecutor(max_workers=self.config['global']['max_cpu_workers']) as executor:
            future_map = {
                executor.submit(self._run_single_task, task, watch_dir, watch_out): task['name']
                for task in tasks
            }

            for future in as_completed(future_map):
                task_name = future_map[future]
                res = future.result()
                results_summary[task_name] = res
                
                completed_count += 1
                if progress_callback:
                    progress_callback(completed_count, total_tasks, f"Task {task_name} finished ({res['status']})")

        # 最終聚合
        summary_path = os.path.join(watch_out, "final_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
            
        return results_summary
```

### 三、 雙入口設計 (Dual Entry Points)

這展示了如何重用上面的核心邏輯。
#### 入口 1: 命令列批次處理 (`main_cli.py`)

不需要啟動 UI，適合放在 Server 上跑三天三夜。
```
# main_cli.py
import os
import argparse
from core.orchestrator import Orchestrator

def cli_progress(current, total, msg):
    print(f"[{current}/{total}] {msg}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch_folder", help="Folder containing subfolders of watches")
    parser.add_argument("--output_dir", default="output")
    args = parser.parse_args()

    engine = Orchestrator()
    
    # 假設 watch_folder 裡面有 500 個手錶資料夾
    watches = [f.path for f in os.scandir(args.watch_folder) if f.is_dir()]

    for watch_path in watches:
        print(f"=== Processing {os.path.basename(watch_path)} ===")
        engine.process_watch(watch_path, args.output_dir, progress_callback=cli_progress)

if __name__ == "__main__":
    main()
```

#### 入口 2: UI 整合 (`App/ProveWatchApp.py` & `worker_thread.py`)

在 PySide6 中，絕對不能在主線程 (Main Thread) 跑耗時運算，否則介面會凍結。我們必須使用 `QThread`。

**1. 建立 Worker Thread (`App/worker_thread.py`):**
```
# App/worker_thread.py
from PySide6.QtCore import QThread, Signal
from core.orchestrator import Orchestrator

class ProcessingWorker(QThread):
    # 定義信號：進度更新(int, int, str)、完成(dict)、錯誤(str)
    progress_updated = Signal(int, int, str)
    finished = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, watch_dir, output_dir):
        super().__init__()
        self.watch_dir = watch_dir
        self.output_dir = output_dir
        self.orchestrator = Orchestrator() # 載入核心

    def run(self):
        try:
            # 定義回調函數，用來發送 Signal
            def callback(curr, total, msg):
                self.progress_updated.emit(curr, total, msg)

            # 開始執行
            results = self.orchestrator.process_watch(
                self.watch_dir, 
                self.output_dir, 
                progress_callback=callback
            )
            self.finished.emit(results)
            
        except Exception as e:
            self.error_occurred.emit(str(e))
        
```

2. 在 App 中調用 (`App/ProveWatchApp.py`):
```
# App/ProveWatchApp.py (部分代碼)
from PySide6.QtWidgets import QMainWindow, QFileDialog
from .worker_thread import ProcessingWorker

class ProveWatchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # ... ui setup ...
        self.worker = None

    def start_analysis(self):
        # 假設使用者已經選好了資料夾
        watch_dir = self.current_watch_folder
        output_dir = "output"

        # 實例化 Worker
        self.worker = ProcessingWorker(watch_dir, output_dir)
        
        # 連接信號
        self.worker.progress_updated.connect(self.update_progress_bar)
        self.worker.finished.connect(self.on_analysis_finished)
        self.worker.error_occurred.connect(self.on_analysis_error)
        
        # 啟動線程
        self.worker.start()
        
        self.ui.statusLabel.setText("Analysis Started...")
        self.ui.runButton.setEnabled(False) # 防止重複點擊

    def update_progress_bar(self, current, total, msg):
        percent = int((current / total) * 100)
        self.ui.progressBar.setValue(percent)
        self.ui.logTextEdit.append(msg)

    def on_analysis_finished(self, results):
        self.ui.statusLabel.setText("Done!")
        self.ui.runButton.setEnabled(True)
        # 顯示結果到 UI...
```

### 四、 Task 腳本標準化建議

為了讓 Orchestrator 能順利工作，所有的 `tasks/run_taskX_process.py` 必須遵守「合約」：

1. **Arguments:** 必須接收 `--image`, `--out_dir`。
    
2. **Output:** 運算結果必須存為 JSON，Mask 必須存為圖片。
    
3. **Error Handling:** 盡量自己 catch exception 並印出錯誤，不要讓 process crash 導致沒有 return code。

```
# tasks/run_taskX_template.py
import argparse
import json
import os
import sys

# 標準化參數
parser = argparse.ArgumentParser()
parser.add_argument("--image", required=True)
parser.add_argument("--out_dir", required=True)
parser.add_argument("--config", help="Optional path to specific task config")
args = parser.parse_args()

try:
    # --- 你的演算法邏輯 ---
    # metric_val = my_algorithm(args.image)
    
    result = {"val": 100, "passed": True}
    
    # 輸出結果
    out_path = os.path.join(args.out_dir, "taskX_result.json")
    with open(out_path, 'w') as f:
        json.dump(result, f)
        
except Exception as e:
    # 讓 stderr 被 orchestrator 抓取
    print(f"Critical Error: {e}", file=sys.stderr)
    sys.exit(1)
```

### 五、 優缺點分析

#### 優點 (Pros)

1. **極高的穩定性 (Stability):** * 即使 Task 5 的 segmentation model 因為記憶體不足崩潰 (Crash)，`subprocess` 會捕捉到 error，Orchestrator 會記錄失敗但**不會影響主程式**，更不會讓 UI 閃退。
    
    - 每個 Task 跑完記憶體完全釋放，解決 Python 常見的記憶體洩漏問題。
        
2. **環境解耦 (Environment Decoupling):**
    
    - 你可以同時跑 PaddleOCR (需要特定版本) 和 DINOv3 (需要最新 PyTorch)，完全不衝突。
        
3. **可擴充性 (Scalability):**
    
    - 未來如果要在 Linux Server 上跑，只需更改 `tasks_map.yaml` 裡的 python 路徑，代碼幾乎不用改。
        
    - UI 與邏輯分離，未來可以把 UI 換成 Web (Flask/FastAPI)，後端的 `core/orchestrator.py` 可以直接重用。
        
4. **UI 響應 (Responsiveness):**
    
    - 使用 `QThread` 確保處理 90 張圖時，UI 依然流暢，可以拖動視窗或取消任務。
        

#### 缺點 (Cons)

1. **啟動開銷 (Overhead):**
    
    - 每次啟動 `subprocess` (Python interpreter) 在 Windows 上大約需要 0.5 ~ 1 秒的 overhead。90 個 tasks 就是額外 1.5 分鐘的時間。
        
    - _解法：_ 如果某些 tasks 很小且環境相同，可以合併成一個 script (e.g., `run_tasks_group1.py`) 一次跑完。
        
2. **除錯複雜度 (Debugging):**
    
    - 因為是 subprocess，你沒辦法直接在 IDE 裡對著 `main.py` 下中斷點跳進去 `run_task1.py`。
        
    - _解法：_ 開發階段直接單獨執行 `python tasks/run_task1.py ...` 進行除錯，確認沒問題再接上 Orchestrator。
        

### 六、 下一步具體建議

您現在可以做的是：

1. 在 Root 建立 `core/` 資料夾，並把上面的 `orchestrator.py` 放進去。
    
2. 建立 `config/tasks_map.yaml`，先填入 2-3 個現有的 tasks 做測試。
    
3. 修改 `App/ProveWatchApp.py`，加入 `ProcessingWorker` 類別並嘗試呼叫。





#### 3. Deployment跟multi environment設定

```
那關於這個project, 如果是需要直接和camera system連結直接擷取images是否可以用main_cli.py或者要其他的python code? 如果有需要deploy到其他customer的主機(windows)執行要如何做, 而且又需要做到不能code跟parameter leak. 是否最好的方法就是打包成exe file? 那這樣要怎麼handle 這個project 的多python environment設定? 以及這個project需要有AI model但checkpoint files可能就會到500mb或1gb, 那要全部包在一起嗎? 而且要如何處理在 customer的主機(windows)的images跟output? 
```
這是一個非常務實且關鍵的 Deployment (部署) 問題。當專案從「開發實驗室」走向「客戶端商業應用」時，架構必須針對 **安全性 (Security)**、**可攜性 (Portability)** 和 **版本管理** 進行重大調整。

以下針對您的疑問逐一分析並提出解決方案：

---

### 一、 Camera System 的整合方式

**Q: 是否可以用 `main_cli.py` 直接截圖，還是需要其他 code？**

建議**不要**把 Camera 邏輯寫死在 `main_cli.py`。為了保持彈性（有些客戶有 Camera，有些客戶只有圖檔），應該採用 **模組化設計**。

1. **建立 `CameraDriver` 模組**： 在 `core/` 下建立一個 `camera_controller.py`。這個模組負責呼叫相機 SDK (Basler, Hikvision, etc.)，拍攝 90 張圖並存入暫存區。
    
2. **修改 `main_cli.py` 增加模式**： 增加一個 `--mode` 參數。

```
# main_cli.py 概念代碼
import argparse
from core.orchestrator import Orchestrator
from core.camera_controller import CameraController # 新增

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["file", "camera"], default="file")
    parser.add_argument("--output_dir", default="./output")
    # ... 其他參數
    args = parser.parse_args()

    watch_images_path = args.watch_folder

    # 模式 1: 連接相機拍攝
    if args.mode == "camera":
        print("Initializing Camera System...")
        cam = CameraController()
        # 拍攝並儲存到臨時資料夾，例如 temp/watch_TIMESTAMP
        watch_images_path = cam.capture_sequence_process() 
        print(f"Images captured at: {watch_images_path}")

    # 模式 2: (或接續模式 1) 開始分析
    orchestrator = Orchestrator()
    orchestrator.process_watch(watch_images_path, args.output_dir)

if __name__ == "__main__":
    main()
```

這樣一支程式就能同時滿足「自動化產線 (Camera Mode)」和「離線分析 (File Mode)」。

### 二、 部署與防護策略 (Deployment & Security)

這是最棘手的部分。Python 本質是直譯語言，源碼很容易被反編譯。**單純打包成一個 EXE (用 PyInstaller) 無法解決多環境問題，也無法完美保護代碼。**

#### 1. 核心代碼保護：使用 Cython 編譯 (最重要的步驟)

不要直接發布 `.py` 檔。使用 **Cython** 將您的核心邏輯 (`core/`, `tasks/`) 編譯成 C 語言，再編譯成 `.pyd` (Windows DLL)。

- **優點**：`.pyd` 是二進位檔，極難反編譯 (Reverse Engineering)，幾乎等同於 C++ 的保護等級。
- **做法**：寫一個 `setup.py` 把所有商業邏輯 compile 起來。

#### 2. 多環境打包：Conda-Pack + 相對路徑

由於您需要多個 Python 環境 (OCR環境, Segmentation環境)，不能把所有東西塞進一個 EXE。

- **策略**：將環境作為「資料夾」隨附在軟體包中。
- **工具**：使用 `conda-pack` 將開發好的 Conda環境打包成獨立的可執行資料夾。

#### 3. 部署架構圖

最終交給客戶的資料夾結構應該長這樣：
```
WatchAuthSystem_Deploy/
├── Launcher.exe          # (PyInstaller) 只有最薄的一層 UI 和呼叫邏輯
├── core.pyd              # (Cython) 加密編譯後的核心指揮官
├── tasks/                # (Cython) 加密編譯後的各個任務邏輯 (.pyd files)
├── config/               # 加密的設定檔
├── envs/                 # [關鍵] 獨立的 Python 環境包
│   ├── base_env/         # 解壓縮後的完整 python 環境
│   ├── ocr_env/
│   └── seg_env/
├── models/               # AI Checkpoints (另外下載或加密)
└── Data/                 # 預設存放 Images 和 Output 的地方
```

### 三、 具體實作細節

#### 1. 解決多環境路徑問題 (Relative Paths)

在客戶電腦上，我們不知道安裝路徑在哪，所以 `config.yaml` 不能寫死 `C:\Envs\...`。必須使用**相對路徑**。

**修改 `core/orchestrator.py` (或 Cython 原始碼) 自動偵測路徑：**

```
import os
import sys

class Orchestrator:
    def __init__(self):
        # 取得目前執行檔 (Launcher.exe) 的所在目錄
        if getattr(sys, 'frozen', False):
            self.base_dir = os.path.dirname(sys.executable)
        else:
            self.base_dir = os.path.dirname(os.path.abspath(__file__))

        # 動態組裝環境路徑
        self.env_paths = {
            "base": os.path.join(self.base_dir, "envs", "base_env", "python.exe"),
            "ocr":  os.path.join(self.base_dir, "envs", "ocr_env", "python.exe"),
            "seg":  os.path.join(self.base_dir, "envs", "seg_env", "python.exe"),
        }
```

#### 2. AI 模型 (Checkpoints) 的處理

模型很大 (500MB+)，不要打包進 `.exe`，也不要放在 Git 裡。

- **部署策略**：
    
    - **方案 A (USB 安裝)**：將模型檔放在隨身碟或光碟，安裝時 copy 到 `models/` 資料夾。
        
    - **方案 B (雲端下載)**：`Launcher.exe` 第一次執行時，檢查 `models/` 是否為空，若是則從您的 S3/Google Drive 下載。
        
- **防護策略**：
    
    - 如果模型是您微調 (Fine-tuned) 過的資產，怕被客戶拿去用，可以對模型權重檔進行**簡單 XOR 加密**。
        
    - 在 Task Script 載入模型前，先在記憶體中解密，再用 `torch.load(io.BytesIO(decrypted_bytes))` 讀取。這樣硬碟上永遠是加密的亂碼。
        

#### 3. 參數 (Parameters) 防護

對於 `config.yaml` 裡的敏感閾值 (Thresholds) 或 API Keys：

- **加密**：將 YAML 轉為二進位加密檔 (e.g., using `fernet` key)。
    
- **硬編碼 (Hardcode)**：如果參數不多，直接寫死在 Cython 的 `.pyx` 檔裡編譯成 `.pyd`，這是最安全的，因為沒有外部文字檔可讀。
    

#### 4. 客戶端資料 (Input/Output) 管理

Windows 有標準的權限管理。

- **建議路徑**：不要存放在 `Program Files` (需要管理員權限)。
    
- **最佳實務**：存放在 `%USERPROFILE%\Documents\WatchAuthSystem\`。
    
- **Log 管理**：系統應該實作「自動清理」，例如保留最近 30 天的 Images/Logs，舊的自動刪除，避免客戶硬碟爆滿。
    

---

### 四、 總結：開發到部署的流程 (Pipeline)

如果要做到您要求的「商業級交付」，流程如下：

1. **環境準備 (Environment Packing)**：
    
    - 使用 `conda-pack -n base_env -o base_env.tar.gz` 打包所有環境。
        
    - 解壓縮到 `deploy_folder/envs/`。
        
2. **代碼混淆與編譯 (Code Protection)**：
    
    - 編寫 `setup.py`，使用 Cython 將 `core/*.py` 和 `tasks/*.py` 編譯成 `.pyd` 檔。
        
    - 刪除原始 `.py` 檔，只保留 `.pyd`。
        
3. **主程式打包 (Launcher)**：
    
    - 使用 PyInstaller 打包 `ProveWatchApp.py` (UI) 和 `main_cli.py`。
        
    - **注意**：PyInstaller 只需要包 "Launcher" 需要的庫 (PySide6, PyYAML)，不需要包 pytorch/ocr 那些重型庫（因為它們在外部 `envs/` 裡）。
        
4. **模型加密 (Optional)**：
    
    - 執行加密腳本，將 `.pth` 轉為加密格式。
        
5. **安裝程式製作 (Inno Setup)**：
    
    - 使用 "Inno Setup" (Windows 免費軟體) 將上述所有檔案 (Exe, Envs, Pyd, Models) 打包成一個 `Setup.exe` 安裝檔。
        
    - 這個安裝檔負責把檔案解壓到客戶的 `C:\Users\Name\AppData\Local\WatchApp`。
        

### 五、 優缺點分析

- **優點**：
    
    - **高安全性**：Cython + 模型加密，極大墊高了破解門檻。
        
    - **無版本衝突**：每個 Task 用自己獨立的 Python 環境執行，完全隔離。
        
    - **穩定**：UI (Launcher) 與運算 (Tasks) 分離，主程式不閃退。
        
- **缺點**：
    
    - **檔案體積大**：因為攜帶了多個 Python 環境 (每個可能 100-500MB)，整個軟體包可能超過 2-3GB。
        
    - **開發複雜度增加**：需要維護 `setup.py` 和編譯流程。



#### 4. 使用Docker

```
那關於考量到python environment跟deployment問題, 如果用Docker是否會推薦呢? 請就使用Docker中文詳細解釋要如何用在這個project, 怎麼處理python environment, deployment, AI model等問題. 並詳細比較優缺點
```

這是一個非常好的問題。引入 Docker 會將專案架構從「本地腳本執行 (Local Script Execution)」轉變為**「微服務架構 (Microservices Architecture)」**。

針對您的需求（Windows、GPU、多環境、部署、IP 保護），使用 Docker **是一個強大的技術選擇，但也伴隨著特定的挑戰**。

以下我將詳細分析如何在 Windows 上使用 Docker 來實現此專案，以及它與直接使用 Conda/Exe 打包方式的比較。

---

### 一、 基於 Docker 的系統架構設計

如果使用 Docker，核心概念是：**不要為每個 Task 啟動一個 Container，而是為每一種「環境」啟動一個常駐服務 (Service)。**

#### 1. 架構概念

- **Host (Windows):** 運行 GUI App (PySide6) 和核心指揮官 (Orchestrator)。
    
- **Docker Container A (OCR Service):** 預裝 PaddleOCR/Tesseract 環境，封裝成一個 FastAPI Server。
    
- **Docker Container B (Seg Service):** 預裝 PyTorch/DINOv3 環境，封裝成另一個 FastAPI Server。
    
- **Shared Volume (資料共用):** Windows 的 `D:\WatchData` 掛載到容器內的 `/data`，讓容器可以直接讀取圖片並寫入結果，不需要透過網路傳輸大量圖片數據。
    

#### 2. 如何處理 Python Environment？

您不再需要 Conda。所有的依賴包都寫在 `Dockerfile` 裡。

**範例：Segmentation 環境的 Dockerfile (`docker/seg_env/Dockerfile`)**

```
# 使用支援 NVIDIA GPU 的 Base Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 安裝依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt fastapi uvicorn

# 複製您的 Task 代碼 (建議是編譯過的 .pyd 或加密過的代碼)
COPY tasks/ ./tasks/
COPY server.py .

# 啟動 API Server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "5001"]
```

#### 3. 如何處理 Deployment (部署)？

在客戶端，您不需要安裝 Python，也不需要設定環境變數。您只需要：

1. **安裝 Docker Desktop (Windows):** 這是最大的前置要求。
    
2. **Docker Compose:** 使用一個 `docker-compose.yml` 檔來定義整個系統。
    

**`docker-compose.yml` 範例：**
```
version: '3.8'

services:
  # OCR 服務
  ocr-service:
    image: myrepo/watch-ocr:v1.0
    ports:
      - "5000:5000"
    volumes:
      - C:\WatchData:/data  # 將 Windows 資料夾掛載進去
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # Segmentation 服務
  seg-service:
    image: myrepo/watch-seg:v1.0
    ports:
      - "5001:5001"
    volumes:
      - C:\WatchData:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```
#### 4. 如何處理 AI Model (Checkpoints)？

- **不建議打包進 Image:** 如果 Image 包含 1GB 模型，更新程式碼時就要重新傳輸 1GB。
    
- **建議方法 - Volume Mapping:** 把模型放在 Windows 的 `C:\WatchModels`，然後掛載進容器。

```
volumes:
  - C:\WatchModels:/app/models  # 容器內程式碼從 /app/models 讀取
```

- 這樣您只需要給客戶一個新的 `.pth` 檔放入資料夾，不需要更新 Docker Image。
    

#### 5. 程式碼與參數保護 (Code & Parameter Leak)

**這是 Docker 的弱點。** Docker Image 本質上只是分層的檔案系統 (Tar files)。

- **問題：** 客戶可以用 `docker export` 把 Image 匯出，解壓縮後就能看到裡面的 `.py` 檔和 `config.yaml`。
    
- **解法：** **Docker 內依然需要使用 Cython。** 在構建 Docker Image 之前，先將 Python Code 編譯成 `.so` (Linux 上的 pyd 對應檔)，然後只 COPY `.so` 檔進入 Docker。這樣即使客戶進入容器，也只能看到二進位檔。
    

---

### 二、 實際運作流程 Example

**App/Orchestrator (Host Windows) 的呼叫方式變了：** 以前是 `subprocess.run(...)`，現在變成 `requests.post(...)`。

```
# Host 端 (Windows GUI) 的 Orchestrator 邏輯
import requests
import os

class DockerOrchestrator:
    def process_watch(self, watch_id):
        # 假設 Windows 路徑是 C:\WatchData\watch_001
        # 容器內路徑對應為 /data/watch_001
        
        container_path = f"/data/{watch_id}"
        
        # 呼叫 OCR Service 處理 Task 3
        # 這裡不需要傳送圖片 Binary，只傳路徑即可 (因為有 Volume Mapping)
        try:
            response = requests.post("http://localhost:5000/process_task", json={
                "task_name": "task3",
                "image_path": f"{container_path}/image_crown.jpg",
                "output_dir": container_path
            })
            print(response.json())
        except Exception as e:
            print(f"Docker service error: {e}")
```

### 三、 Docker 方案 vs. Native (Exe/Conda) 方案比較

這是決策的關鍵對比表：

|**特性**|**Docker 方案 (Microservices)**|**Native 方案 (Conda-pack + Exe)**|
|---|---|---|
|**環境隔離**|**完美**。絕對乾淨，不會受客戶電腦其他軟體影響。|**優秀**。Conda 環境也是隔離的，但依賴 Windows DLL。|
|**部署難度**|**中/高**。客戶必須安裝並設定 Docker Desktop (可能涉及 WSL2 設定)。|**低**。解壓縮 + 執行 Setup.exe 即可，客戶無感。|
|**啟動速度**|**極快**。容器是常駐服務，API 呼叫毫秒級回應。無 Interpreter 啟動開銷。|**慢**。每個 Task 都要重新啟動 Python Interpreter (約 0.5-1秒開銷)。|
|**GPU 支援**|**複雜**。需安裝 NVIDIA Container Toolkit，且 WSL2 的 GPU 效能約為原生 90-95%。|**原生**。直接調用顯卡驅動，效能 100%，相容性最好。|
|**代碼保護**|**弱 (需額外處理)**。Image 易被檢視，必須配合 Cython (.so)。|**中/強**。配合 Cython (.pyd) + PyInstaller，一般人難以破解。|
|**資料傳輸**|透過 Volume Mapping，但需注意 Windows/Linux 路徑轉換。|直接檔案讀寫，無路徑轉換問題。|
|**記憶體佔用**|**高**。每個容器都要預先佔用 RAM/VRAM (除非動態釋放)，Docker 本身也吃資源。|**低/動態**。Task 跑完 Process 殺掉，記憶體立刻釋放。|

---

### 四、 結論建議

針對您的專案特性 (Windows, High Throughput, Commercial Deployment)，我的建議如下：

#### **情況 A：如果您對「部署簡易度」和「原生效能」要求最高**

**👉 推薦使用之前的 `Conda-pack` + `Orchestrator Subprocess` + `Cython` 方案。**

- **理由：** 要求客戶安裝 Docker Desktop 在商業軟體中是個很高的門檻 (特別是企業客戶可能有 IT 權限限制)。原生的 EXE 檔案最容易被客戶接受。
    
- **解決啟動速度問題：** 您可以把 90 個 Task 合併成 3 個大的 Script (對應 3 個環境)，一次傳入多張圖處理，減少 Process 啟動次數。
    

#### **情況 B：如果您希望「開發維護容易」且「系統需要長時間穩定運行」**

**👉 推薦使用 `Docker` 方案。**

- **理由：** 您不需要寫複雜的 Subprocess 管理代碼。API Server (FastAPI) 非常穩定。且 Docker 保證了「在我的電腦能跑，在客戶電腦就能跑」。
    
- **前提：** 必須確認客戶電腦允許安裝 Docker Desktop 並且開啟 WSL2 功能。
    

#### **折衷方案 (最佳化架構)**

如果是我來架構這個系統，我會選擇 **"Native Service" (方案 A 的改良版)**：

1. 不使用 Docker (為了部署方便)。
    
2. 但在打包的 `envs` 裡，針對那 3 個不同的 Python 環境，各寫一個 `server.py` (FastAPI)。
    
3. 當 `Launcher.exe` 啟動時，它會在背景使用 `subprocess` 啟動這 3 個 Python API Server (Localhost:5000, 5001, 5002)。
    
4. 主程式透過 HTTP Request 跟這 3 個本地 Server 溝通。
    

**這樣做的優點：**

- **速度快：** 像 Docker 一樣是常駐服務，沒有啟動開銷。
    
- **部署簡單：** 不需要客戶裝 Docker，只需要解壓縮檔案。
    
- **記憶體安全：** 可以寫 API 讓它定時釋放 VRAM。
    
- **保護性：** 依然可以用 Cython 編譯 Server 代碼。





#### 5. 所有方案(Native, Cloud跟其他)比較
```
那如果換成主要計算的部分方在cloud, 譬如AWS. 在客戶端只要安裝最基本的client及從camera 擷取images, 可以直接上傳到cloud完成所有計算再把結果傳回給客戶. 也請基於這方案詳細分析及提供實作的方法. 除此之外還有甚麼其他推薦的系統設計方案? 請就這些方案統整比較優缺點
```

這是一個將商業模式從「軟體銷售 (License)」轉向「軟體即服務 (SaaS)」的重大轉變。

將主要計算移至 **Cloud (AWS)** 確實能解決許多地端部署的痛點（如硬體規格不一、代碼保護、環境衝突），但同時也會引入新的挑戰（如頻寬、雲端成本）。

以下針對 **Cloud 架構** 進行詳細設計，並將其與之前的 **Local Native (原生)** 及 **On-Prem Server (專用主機)** 方案進行總結比較。

---

### 一、 Cloud Architecture (AWS 雲端方案) 詳細設計

這個架構的核心思想是：**客戶端極輕量化 (Light Client)**，只負責拍照和上傳；**雲端負責重運算 (Heavy Compute)**。

#### 1. 系統架構圖 (AWS 範例)

1. **Client (Windows):** Python GUI + Camera SDK。負責拍照 -> 壓縮 -> 上傳。
    
2. **Upload (S3):** 接收原始圖片。
    
3. **Queue (SQS):** 當圖片上傳完成，發送訊息到佇列，觸發後端處理（削峰填谷，避免瞬間流量沖垮 Server）。
    
4. **Compute (ECS/EKS on GPU Instances):**
    
    - 使用 **Amazon ECS (Elastic Container Service)** 管理 Docker 容器。
        
    - 配置 **Auto Scaling Group**，使用 `g4dn.xlarge` (T4 GPU) 或 `g5` 系列。平時開 0 台，有任務時自動開啟，省錢。
        
    - 這裡部署您原本的 3 種環境 (Base, OCR, Seg) 為不同的 Task Definition。
        
5. **Storage (S3 + RDS):**
    
    - Masks/Output Images -> 存回 S3。
        
    - Metrics/Auth Result -> 存入 **Amazon RDS (PostgreSQL)**。
        
6. **Response:** Client 輪詢 (Polling) API 或透過 WebSocket 接收完成通知。
    

#### 2. 實作細節與 Example Code

**A. 客戶端 (Client Side - Upload Logic)** 客戶端不再需要 PyTorch 或 CUDA，只需要 `boto3` (AWS SDK)。
```
# client/uploader.py
import boto3
import os
import time

# 初始化 S3 Client (在正式版中應透過 API Gateway 取得暫時的 Presigned URL，而非直接把 Key 放在這)
s3 = boto3.client('s3', region_name='us-east-1')
bucket_name = 'my-watch-analysis-bucket'

def upload_watch_images(watch_id, image_folder):
    """
    將 90 張圖片打包上傳
    """
    print(f"Uploading watch {watch_id}...")
    
    # 建議先壓縮成 zip 以減少 HTTP request 數量
    zip_path = f"{watch_id}.zip"
    # ... (壓縮邏輯) ...
    
    # 上傳
    s3.upload_file(zip_path, bucket_name, f"incoming/{watch_id}.zip")
    print("Upload complete. Waiting for results...")

def check_status(watch_id):
    """
    輪詢 API 查詢進度
    """
    # 假設有一個 API Gateway endpoint
    import requests
    url = f"https://api.mywatchservice.com/status/{watch_id}"
    
    while True:
        resp = requests.get(url).json()
        if resp['status'] == 'completed':
            return resp['result']
        elif resp['status'] == 'failed':
            raise Exception("Processing failed")
        
        print(f"Current status: {resp['progress']}%")
        time.sleep(5)
```
**B. 雲端後端 (Cloud Side - Worker Logic)** 使用 Python 腳本監聽 SQS 佇列或被 AWS Batch 觸發。


```
# cloud/worker.py (跑在 ECS 容器內)
import boto3
import os
from core.orchestrator import Orchestrator # 重用之前的核心邏輯

def process_from_s3(bucket, key):
    # 1. 下載 Zip
    local_zip = "/tmp/watch.zip"
    s3.download_file(bucket, key, local_zip)
    
    # 2. 解壓縮
    extract_path = "/tmp/watch_images"
    # ... (解壓縮) ...
    
    # 3. 執行核心分析 (重用既有的 Orchestrator)
    # 這裡的 Orchestrator 會呼叫本地的其他 Docker 容器或直接執行
    orchestrator = Orchestrator()
    results = orchestrator.process_watch(extract_path, "/tmp/output")
    
    # 4. 上傳結果
    s3.upload_file("/tmp/output/final_summary.json", bucket, f"results/{key}.json")
    
    # 5. 寫入資料庫 (RDS)
    save_to_db(results)
```

### 二、 其他推薦的系統設計方案

除了 **Local Native** 和 **Cloud SaaS**，還有一個非常適合工業/高階檢測的「中間路線」：

#### **方案 C：Edge Box (邊緣運算專用機/黑盒子)**

這是目前高端檢測設備（如康耐視 Cognex、Keyence）的主流做法。

- **概念**：您不出售軟體安裝檔，而是出售一台**「預先灌好軟體和環境的硬體主機 (IPC - Industrial PC)」**。
    
- **硬體**：一台配備 NVIDIA RTX 4060/A2000 的小型工控機 (如 Advantech, Dell Precision Compact)。
    
- **部署**：
    
    - 您在公司內部把環境 (Docker/Native)、模型、加密都做完。
        
    - 將這台電腦直接寄給客戶，客戶只需要把 Camera USB 插上去，接上螢幕即可使用。
        
- **優點**：硬體規格統一（您說了算），完全沒有環境相容性問題，IP 保護最好（可以封死 USB port 或 BIOS 加密）。
    

---

### 三、 三種方案綜合比較 (優缺點統整)

這是決策的關鍵對比表，請根據您的商業模式選擇：

|**比較項目**|**1. Local Native / Microservices (軟體安裝包)**|**2. Cloud SaaS (AWS/Azure)**|**3. Edge Box (專用硬體主機)**|
|---|---|---|---|
|**IP 保護 (防破解)**|**中** (需依賴 Cython/混淆，駭客仍有機會逆向)|**最高** (代碼和模型都在雲端，客戶完全拿不到)|**高** (物理隔離，硬碟加密，客戶難以取出檔案)|
|**部署難度**|**高** (需處理客戶各種雜亂的 Windows 環境、驅動衝突)|**極低** (客戶只需安裝一個 10MB 的上傳程式)|**低** (插電即用，但涉及硬體物流配送)|
|**運算效能/延遲**|**取決於客戶硬體** (快慢不一)|**受限於頻寬** (90張 2K 圖約 300MB+，上傳需要時間)|**極高且穩定** (本地 GPU 直連，無網路延遲)|
|**成本結構 (Cost)**|**開發成本** (一次性賣斷或訂閱)|**營運成本 (OpEx)** (每月付 AWS GPU 費用，需轉嫁給客戶)|**硬體成本 (CapEx)** (需先墊付硬體費，售價較高)|
|**維護與更新**|**困難** (需請客戶下載 Patch 檔安裝)|**極易** (Server 端更新代碼，全球客戶即刻生效)|**中等** (需遠端連線 TeamViewer 或 OTA 更新)|
|**適用場景**|客戶已有高性能電腦，且預算有限|客戶網路環境好，不想買硬體，按次計費|高單價手錶鑑定，追求極致速度與穩定性|

---

### 四、 深度分析：Cloud 方案的致命傷與解法

雖然 Cloud 看起來最現代化，但在您的應用場景 (Imaging Processing) 中有一個巨大的隱憂：**頻寬 (Bandwidth)**。

- **數據量分析**：
    
    - 1 張 2048x2048 jpg (高品質) $\approx$ 2MB ~ 5MB。
        
    - 90 張 $\approx$ 180MB ~ 450MB。
        
    - **500 支手錶/批 $\approx$ 90GB ~ 225GB 的數據量。**
        
- **問題**：
    
    - 一般辦公室網速上傳 500支手錶可能需要 **數小時** 甚至 **數天**。這會讓「高吞吐量」的系統變得卡在網路上。
        
    - AWS 的 Data Transfer Out (下載結果) 和 Storage 費用會隨量暴增。
        

#### **針對 Cloud 頻寬問題的解法：**

如果不放棄 Cloud 方案，您必須做 **Edge-Cloud Hybrid (混合雲)**：

1. **Client 端預處理**：在客戶端 Python 做 Resize (例如縮到 1024x1024) 或 ROI Crop (只切出錶盤、Logo 部分)，將上傳量減少 80%。
    
2. **分流計算**：
    
    - **輕量 OCR**：直接在客戶端跑 (Tesseract/PaddleOCR 輕量版)。
        
    - **重型 Segmentation/真偽判定**：只上傳關鍵特徵圖到 Cloud 跑 DINOv3。
        

### 五、 最終建議

根據您提到的需求：「一次處理 500 支手錶」、「支援 GPU」、「穩定高效」：

1. **首選推薦：Edge Box (方案 C)**
    
    - **理由**：這是 B2B 高階檢測設備的標準做法。賣整套設備 (Camera Rig + Computer + Software)。您可以完全控制硬體環境，穩定性最高，且沒有上傳 200GB 圖片的網路瓶頸。這最能展現「專業鑑定」的價值。
        
2. **次選推薦：Local Native Service (方案 A - 改良版)**
    
    - **理由**：如果必須讓客戶用自己的電腦，請採用我上一則回答提到的「本地 API Service (Native Microservices)」。打包好的環境 + Cython 加密。這省去了雲端昂貴的 GPU 租賃費和巨大的傳輸時間。
        
3. **不推薦：純 Cloud 方案**
    
    - **理由**：除非您的客戶只是一次鑑定 1 支錶，否則「500支 x 90張高解析圖」的傳輸延遲會毀了使用體驗，且 AWS GPU 實例長期運行的帳單會非常驚人。





#### 6. 現在開發核心模組未來再決定部署方式
```
請中文詳細解釋比較: App/Orchestrator (Host Windows) 的呼叫方式變了：

以前是 subprocess.run(...)，現在變成 requests.post(...)這兩者差別在哪裡? , 也中文詳細解釋: 在打包的 envs 裡，針對那 3 個不同的 Python 環境，各寫一個 server.py (FastAPI)。

當 Launcher.exe 啟動時，它會在背景使用 subprocess 啟動這 3 個 Python API Server (Localhost:5000, 5001, 5002)跟要怎麼實作? 最後目前還處於系統開發階段. 所以目前先開發最重要的python files, 但希望到時候有足夠彈性可以選擇Local Native / Microservices (軟體安裝包用cli或者有ui), Cloud SaaS或Edge Box (專用硬體主機). 所以我現在先著重在main_cli.py, 資料夾core的orchestrator.py, camera_controler.py, 資料夾tasks的所有run_tasks.py, 以及env資料夾跟config資料夾, 以及App裡面的ui相關檔案. 到時候要選擇不同方案都很容易使用. 這是否正確?
```

這是一個非常核心且具備高度戰略性的問題。您目前的思維完全正確：**「現在開發核心模組，保持架構彈性，未來再決定部署方式」** 是最穩健的開發策略。

我將分三個部分詳細回答：比較呼叫方式、實作 Local Native Microservices、以及確認您目前的開發路徑。

---

### 一、 `subprocess.run(...)` vs `requests.post(...)` 的詳細比較

這兩者的差別在於 **「啟動成本 (Startup Cost)」** 與 **「狀態保留 (State Persistence)」**。想像你要開一家餐廳（您的系統），裡面有 90 個不同的廚師（Tasks）：

#### 1. `subprocess.run(...)` (CLI 模式)

這就像是 **「臨時工廚師」**。

- **流程**：每當有一張圖片需要處理（例如 Task 1），Orchestrator 就打電話叫廚師來上班 (`python.exe run_task1.py`)。廚師走進廚房，穿上圍裙，把刀具拿出來，**載入 AI 模型 (Loading Model)**，切好菜，把結果寫在紙上 (JSON)，然後**脫掉圍裙下班回家 (Process Exit)**。
    
- **記憶體**：用完即丟。Task 結束後，VRAM/RAM 全部清空。
    
- **缺點 (致命傷)**：如果載入 AI 模型需要 3 秒，處理圖片只要 0.1 秒。那你每處理一張圖都要浪費 3 秒載入。處理 500 支錶 x 90 張圖，時間都浪費在載入模型上。
    
- **呼叫方式**：
```
- # 透過作業系統指令呼叫，傳遞參數透過字串
subprocess.run(["python", "task1.py", "--image", "img.jpg"])
```

#### 2. `requests.post(...)` (API Server 模式)

這就像是 **「正職駐點廚師」**。

- **流程**：餐廳一開門 (`Launcher` 啟動)，所有廚師都已經在廚房待命了。他們已經穿好圍裙，刀具拿好了，**AI 模型已經載入在 VRAM 裡了**。Orchestrator 只需要喊一聲（發送 HTTP Request），廚師立刻切菜並回報結果，然後**繼續站在原地等下一個指令**。
    
- **記憶體**：常駐佔用。如果模型很大，VRAM 會一直被佔用，直到整個軟體關閉。
    
- **優點**：沒有載入時間。處理速度極快，吞吐量最高。
    
- **呼叫方式**：
```
# 透過網路協定呼叫，傳遞參數透過 JSON
requests.post("http://localhost:5000/analyze", json={"image": "img.jpg"})
```

**總結差異表：**

|**特性**|**subprocess.run**|**requests.post**|
|---|---|---|
|**模型載入**|每次執行都要重新載入 (慢)|程式啟動時載入一次，之後常駐 (快)|
|**記憶體**|峰值高，但用完即放|持續佔用，需計算總 VRAM 是否足夠|
|**溝通介面**|命令行參數 (CLI Args)|JSON payload|
|**適用場景**|簡單運算、不需要 GPU 模型的 Task|需要 GPU 模型、高頻率呼叫的 Task|

### 二、 如何實作「Local Native Microservices」(本地 API 服務)

這是結合了 Conda 的便利性與 Microservices 效能的「黃金方案」。

#### 1. 概念圖

當使用者點擊 `Launcher.exe` (或執行 `main.py`) 時：

1. **Main Process** 啟動。
    
2. **Main Process** 立刻在背景用 `subprocess.Popen` (注意不是 `run`) 去啟動 3 個隱藏的 Python Process。
    
    - Process A: 啟動 PaddleOCR Server (Port 5000)
        
    - Process B: 啟動 PyTorch Segmentation Server (Port 5001)
        
    - Process C: 啟動 Base Processing Server (Port 5002)
        
3. **Main Process** 等待 5 秒讓它們初始化完畢。
    
4. **Main Process** 開始透過 `requests` 發送任務給這 3 個 Port。
    
5. 軟體關閉時，Main Process 殺掉這 3 個子 Process。
    

#### 2. 實作步驟 Example

**Step A: 撰寫 Server 端 (`tasks/server_seg.py`)** 這是在 `seg_env` 環境下跑的程式。

```
# tasks/server_seg.py
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
# 假設您的核心邏輯已經封裝成 function
from my_segmentation_algo import load_model, process_image

app = FastAPI()

# 全域變數，啟動時載入模型 (Warm Start)
print("Loading Segmentation Model into VRAM...")
model = load_model("checkpoint.pth") 
print("Model Loaded!")

class TaskRequest(BaseModel):
    image_path: str
    output_dir: str

@app.post("/process_task2")
def run_task2(req: TaskRequest):
    # 直接記憶體內執行，不用重載模型
    result = process_image(model, req.image_path)
    # 儲存結果...
    return {"status": "success", "data": result}

if __name__ == "__main__":
    # 啟動 Server
    uvicorn.run(app, host="127.0.0.1", port=5001)
```

**Step B: 撰寫啟動器 (`core/service_manager.py`)** 這是 Main Process 用來管理背景服務的模組。
```
# core/service_manager.py
import subprocess
import time
import os
import requests
import sys

class ServiceManager:
    def __init__(self, config):
        self.processes = []
        self.config = config # 包含環境路徑的設定

    def start_services(self):
        """啟動所有背景 API Servers"""
        services = [
            {"env": "seg_env", "script": "tasks/server_seg.py", "port": 5001},
            {"env": "ocr_env", "script": "tasks/server_ocr.py", "port": 5000},
        ]

        for svc in services:
            python_exe = self.config['envs'][svc['env']] # 取得該環境的 python.exe 路徑
            cmd = [python_exe, svc['script']]
            
            print(f"Starting {svc['env']} service on port {svc['port']}...")
            # Popen 不會卡住主程式，它會在背景跑
            proc = subprocess.Popen(
                cmd, 
                cwd=os.getcwd(),
                # Windows 下隱藏視窗的參數 (Optional)
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0
            )
            self.processes.append(proc)

        # 等待服務啟動 (可用簡單的 sleep，正規作法是輪詢 /health endpoint)
        print("Waiting for services to warm up...")
        time.sleep(10) 

    def stop_services(self):
        """關閉所有背景服務"""
        for proc in self.processes:
            proc.terminate()
            proc.wait()
        print("All services stopped.")
```

Step C: 修改 Orchestrator 呼叫方式

```
# core/orchestrator.py
def _run_task_via_api(self, port, image_path, out_dir):
    url = f"http://127.0.0.1:{port}/process_task2"
    resp = requests.post(url, json={"image_path": image_path, "output_dir": out_dir})
    return resp.json()
```

### 三、 對於您目前開發策略的驗證與建議

您說：

> _「目前先開發最重要的python files (main_cli, core, tasks, env, config, ui)，但希望到時候有足夠彈性可以選擇... 這是否正確?」_

**答案是：完全正確，但有一個「關鍵前提」。**

為了要能同時支援 `CLI (subprocess)`, `Local API`, `Cloud`, `Edge`，您的代碼必須嚴格遵守 **「邏輯與介面分離 (Decoupling Logic from Interface)」** 的原則。

#### 1. 什麼是「邏輯與介面分離」？ (The Golden Rule)

很多初學者會把程式寫死在 `if __name__ == "__main__":` 裡面。這樣做的話，FastAPI Server 無法 import 你的演算法，Cloud Worker 也無法重用。

**錯誤的寫法 (Tasks/run_task1.py):**
```
import sys
# ... imports

if __name__ == "__main__":
    img = sys.argv[1]
    # ... 一大堆運算邏輯寫在這裡 ...
    # ... 載入模型 ...
    print(result)
```

_為什麼錯？_ 因為如果我要改用 FastAPI，我無法 `import` 這段邏輯，我只能用 `subprocess` 跑它，這樣就無法享受 FastAPI 的 "Warm Start" 優勢。

**正確的寫法 (重構建議):**

建議將 Task 拆成兩層：

**第一層：純演算法 (`tasks/algo/task1_algo.py`)**
```
# 純 Python function，不涉及 sys.argv，不涉及 print
def analyze_dial_image(image_path, model=None):
    # 如果傳入 model 就用，沒傳入就載入 (支援兩種模式)
    if model is None:
         model = load_model()
    
    # ... 運算邏輯 ...
    return {"score": 98.5}
```

**第二層：介面層 (Interfaces)**

- **介面 A (CLI):** `tasks/run_task1_cli.py`
```
from algo.task1_algo import analyze_dial_image
# 這裡處理 argparse, sys.argv
# 這裡呼叫 analyze_dial_image()
```
**介面 B (API Server):** `tasks/server_seg.py`
```
from algo.task1_algo import analyze_dial_image
# 這裡處理 FastAPI request
# 這裡預先載入 model，並重複呼叫 analyze_dial_image(img, model=my_loaded_model)
```
#### 2. 您目前的資料夾結構建議微調

為了達成上述彈性，建議您的結構微調如下：

```
Project_Root/
├── core/
│   ├── orchestrator.py      # 這裡要有 "Strategy Pattern"，可以切換 call subprocess 或 call requests
│   └── service_manager.py   # 新增：負責啟動 API Server
├── tasks/
│   ├── algorithms/          # [關鍵] 這裡放純演算法，無 CLI 依賴
│   │   ├── dial_algo.py
│   │   └── ocr_algo.py
│   ├── cli_wrappers/        # 這裡放原本的 run_taskX.py (for CLI/Subprocess mode)
│   └── api_servers/         # 這裡放 server_seg.py (for Local Native/Docker mode)
├── config/
│   └── config.yaml          # 在這裡增加一個開關: mode: "cli" 或 "api"
...
```

#### 3. 結論

您目前的路線圖非常清晰。現在只需要專注於：

1. **Environment Setup:** 把 env yaml 寫好。
    
2. **Core Algorithms:** 寫好 `tasks/algorithms/` 下的圖像處理邏輯。
    
3. **CLI Wrappers:** 先把 `main_cli.py` 和 `subprocess` 跑通（這是最基礎的）。
    

等到基礎跑通後，要升級成 **Local Native Microservices** 或 **Cloud** 時，只需要：

1. 寫 `api_servers/*.py` (Import 剛剛寫好的 Algorithms)。
    
2. 在 `orchestrator.py` 裡加一個 `if mode == "api": requests.post(...)`。
    

這樣您就立於不敗之地，可以隨時根據客戶需求切換架構。






#### 7. 實作cli跟server模式
```
請一步步解釋如果這些code都經ready, 要怎麼實作, 譬如已經有image folder要怎麼跑這個程式, 以及要怎麼跟之前在App folder下有ui介面的code結合? 已知有個介面上的按鍵按下去就會triger開始影像分析的tasks, 是不是直接按鍵按下去的event就直接接main_cli.py? 然後所以目前執行main_cli.py 如果system_config.yaml是"api"等於是開啟背景服務, 執行main_cli.py如果system_config.yaml是"cli"等於是主動run這些task file?
```

### 第三部分：實作步驟教學 (How to run)

假設您已經把檔案都放好了（依照之前的目錄結構）。

#### 1. 準備工作

- **Images**: 確保 `C:/Data/Watch_Test/` 裡面有幾張測試圖片。
    
- **Config**: 確保 `config/system_config.yaml` 和 `config/task_configs/ocr_task.yaml` 都設定正確（路徑指向您的 Conda 環境 Python exe）。
    

#### 2. 測試 CLI 模式 (最簡單的測試)

我們先不要跑背景服務，直接測試 `tasks/cli_wrappers/run_ocr_cli.py` 是否能動。

打開 CMD (Terminal):
```
# 啟動您的 ocr conda 環境
conda activate ocr-analysis-env

# 執行 CLI Wrapper
python tasks/cli_wrappers/run_ocr_cli.py \
  --image "C:/Data/Watch_Test/dial.jpg" \
  --out_dir "C:/Data/Output" \
  --config "config/task_configs/ocr_task.yaml"
```
如果成功，您應該會在 Output 資料夾看到結果圖片和 JSON。

#### 3. 測試 API 模式 (背景服務)

這是未來 UI 要用的模式。

**步驟 A: 啟動 Server** 在 CMD 中（或透過 `main_cli.py` 的程式碼）：
```
# 啟動 ocr 環境
conda activate ocr-analysis-env

# 啟動 Server
python tasks/api_servers/server_ocr.py \
  --config "config/task_configs/ocr_task.yaml" \
  --port 5000
```
_這時候您會看到 "EasyOCR Reader Initialized" 的 Log，並且程式會卡住等待連線 (Uvicorn running on ...)。_

**步驟 B: 發送請求** 開**另一個** CMD (模擬 UI 端)：
```
# 可以用 curl 測試
curl -X POST "http://127.0.0.1:5000/process" \
     -H "Content-Type: application/json" \
     -d "{\"image_path\": \"C:/Data/Watch_Test/dial.jpg\", \"output_dir\": \"C:/Data/Output\"}"
```
### 第四部分：與 App UI 的結合與邏輯釐清

這裡回答您關於 `main_cli.py` 與 `App` 的關鍵問題。

#### Q1: 按下 UI 按鍵後，是直接接 `main_cli.py` 嗎？

**不建議直接接 `main_cli.py`。**

- `main_cli.py` 通常是給**工程師**在沒有 UI 的情況下跑批次作業用的腳本。
    
- UI (`ProveWatchApp.py`) 應該直接呼叫 `core.orchestrator.Orchestrator`。
    

#### Q2: 關於 `system_config.yaml` 的 "api" 與 "cli" 模式解釋

您的理解基本正確，但實作細節如下：

- **如果 Config 是 "cli"**:
    
    - Orchestrator 每次處理一張圖，都會執行 `subprocess.run(...)`。
        
    - 這意味著：開啟 Python -> 載入 EasyOCR (3秒) -> 運算 -> 關閉。
        
    - **優點**: 不用管理背景服務。**缺點**: 很慢。
        
- **如果 Config 是 "api"**:
    
    - Orchestrator 會假設 **「已經有一個 Server 在 localhost:5000 等我了」**，所以它直接發 `requests.post`。
        
    - **關鍵問題：誰去啟動那個 Server？**
        
        - 在 **App** 的架構下，應該是 **App 啟動時 (`__init__`)** 就呼叫 `ServiceManager` 去把背景黑視窗 (Server) 打開。
            
        - 當 **App 關閉時 (`closeEvent`)**，呼叫 `ServiceManager` 把背景 Server 殺掉。
            

#### Q3: UI 程式碼實作範例

這是如何在 PySide6 中整合的關鍵代碼。
```
# App/ProveWatchApp.py
import sys
from PySide6.QtWidgets import QMainWindow, QApplication, QPushButton
from PySide6.QtCore import QThread, Signal

# 引入核心模組
from core.service_manager import ServiceManager
from core.orchestrator import Orchestrator

# --- Worker Thread (負責執行耗時的分析任務) ---
class AnalysisWorker(QThread):
    finished = Signal(dict) # 完成信號
    progress = Signal(str)  # 進度信號

    def __init__(self, orchestrator, image_list, output_dir):
        super().__init__()
        self.orchestrator = orchestrator
        self.image_list = image_list
        self.output_dir = output_dir

    def run(self):
        self.progress.emit("Starting analysis...")
        results = {}
        for img_path in self.image_list:
            # 這裡 Orchestrator 會自動根據 config 決定是用 API 還是 CLI
            self.progress.emit(f"Processing {img_path}...")
            
            # 假設這個圖是要跑 OCR Task
            res = self.orchestrator.run_task("task_ocr_analysis", img_path, self.output_dir)
            results[img_path] = res
            
        self.finished.emit(results)

# --- Main Window ---
class ProveWatchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Watch Auth System")
        self.resize(800, 600)

        # 1. 啟動背景服務 (這是 API 模式的關鍵!)
        # 建議做一個 Loading Splash Screen 來掩蓋這 10 秒的啟動時間
        self.service_mgr = ServiceManager("config/system_config.yaml")
        print("Starting background services...")
        self.service_mgr.start_all_services() 

        # 2. 初始化指揮官
        self.orchestrator = Orchestrator("config/system_config.yaml")

        # UI 元件
        self.btn_run = QPushButton("Start Analysis", self)
        self.btn_run.clicked.connect(self.on_start_click)
        self.setCentralWidget(self.btn_run)

    def on_start_click(self):
        # 模擬取得圖片列表
        images = ["C:/Data/test/dial.jpg", "C:/Data/test/logo.jpg"]
        output = "C:/Data/output"

        # 3. 啟動 Worker Thread
        self.worker = AnalysisWorker(self.orchestrator, images, output)
        self.worker.progress.connect(lambda msg: print(f"[UI Log] {msg}"))
        self.worker.finished.connect(self.on_analysis_done)
        self.worker.start()
        
        self.btn_run.setEnabled(False) # 防止重複點擊

    def on_analysis_done(self, results):
        print("Analysis Finished!", results)
        self.btn_run.setEnabled(True)

    def closeEvent(self, event):
        # 4. 關閉視窗時，務必殺掉背景服務
        print("Stopping services...")
        self.service_mgr.stop_all_services()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ProveWatchApp()
    window.show()
    sys.exit(app.exec())
```
### 總結您的開發路徑

1. **現在**: 將提供的 code 填入 `ocr_algo.py`，並確認 `core/` 下的檔案都建立了。
    
2. **Config**: 檢查 `system_config.yaml` 把模式設為 `"api"`。
    
3. **Run**: 不要直接跑 `ocr_algo.py`。
    
    - 如果您想測試單一任務：跑 `tasks/cli_wrappers/run_ocr_cli.py`。
        
    - 如果您想測試完整流程：跑 `App/ProveWatchApp.py` (它會自動幫您開啟 server)。
        

這樣設計，您的 UI 永遠保持流暢，而背後的 Python 環境與 GPU 模型管理則由 `ServiceManager` 完美處理。







```
ImagingLibWatch/
├── App/                     # UI 相關代碼
│   ├── modules/
│   ├── resources.qrc
│   ├── ProveWatchApp.py     # UI 入口 (Entry Point 1)
│   └── worker_thread.py     # [新增] 負責連接 UI 與 Orchestrator 的橋樑
├── checkpoints/   
├── config/                  # 設定檔
│   ├── system_config.yaml   # [新增] 定義 Python 環境路徑、GPU 設定
│   ├── tasks_map.yaml       # [新增] 定義 90 個 Tasks 的執行順序、環境與參數
│   └── task_configs/        # 各個 Task 獨有的參數 (如閾值等)
├── core/                    # [新增] 核心邏輯層
│   ├── __init__.py
│   ├── orchestrator.py      # [核心] 指揮官，負責調度 Subprocess
│   └── service_manager.py   # 新增：負責啟動 API Server
├── DB/                      # UI 相關代碼
│   ├── WatchDB/             # watch mat DB files
│   ├── hb_profile/          # texture fingerprint files
│   └── Experiment.parquet   # Experiment index table file
├── env/                     # Conda environments YAML
├── Experiments/             # Image pipeline輸出結果 (Experiment result folder)
├── images/                  # 輸入圖片庫
├── output/                  # 輸出結果
├── tasks/
│   ├── algorithms/          # [關鍵] 這裡放純演算法，無 CLI 依賴
│   │   ├── dial_algo.py
│   │   └── ocr_algo.py
│   ├── cli_wrappers/        # 這裡放原本的 run_taskX.py (for CLI/Subprocess mode)
│   └── api_servers/         # 這裡放 server_seg.py (for Local Native/Docker mode)
└── main_cli.py              # 純命令列入口 (Entry Point 2)

```


```
ImagingLibWatch/
├── App/                     # App/UI Layer
│   ├── modules/
│   ├── resources.qrc
│   ├── ProveWatchApp.py     # UI entry (Entry Point 1)
│   └── worker_thread.py     # connect UI to Orchestrator
├── checkpoints/   
├── config/                  
│   ├── system_config.yaml   # Define Python env setting, tasks env and setting
│   └── "task X"_config.yaml  # "Task X" config files
├── core/                    # Core logic layer
│   ├── __init__.py
│   ├── orchestrator.py      # Core orchestrator to manage Subprocess
│   └── service_manager.py   # Activate and manage API Server
├── DB/                      # Database related files
│   ├── WatchDB/             # watch mat DB files
│   ├── hb_profile/          # texture fingerprint files
│   └── Experiment.parquet   # Experiment index table file
│   └── Authentication.parquet   # Authentication rule Database file
├── env/                     # Conda environments YAML
├── Experiments/             # Image pipeline results (Experiment result folder)
├── images/                  # Image/Camera extraction folder
├── output/                  # User output/statistics/authentication output folder
├── tasks/
│   ├── algorithms/          # algorithm functions
│   │   └── "task X"_algo.py
│   ├── cli_wrappers/        # run_"task X".py (for CLI/Subprocess mode)
│   └── api_servers/         # server_"task X".py (for Local Native/Docker mode)
└── main_cli.py              # CLI entry (Entry Point 2)

```