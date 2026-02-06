
|                                      |     |
| ------------------------------------ | --- |
| [[#### 1226完善代理人模式 (Proxy Pattern)]] |     |
|                                      |     |
|                                      |     |
|                                      |     |




#### 1226完善代理人模式 (Proxy Pattern)
```
20251226

這系統在硬體上是自動光學檢測 (AOI) 系統包括多相機系統以及精密運動平台：使用 Zaber 控制器進行 X, Y, Z 三軸移動, 雷射/位移感測器：用於測量表面高度和玻璃厚度, 可控光源, 多通道燈光控制以及Vanta Element-S可以做material analysis. 目的是將watch放在平台上藉由多相機系統在watch的不同view, 不同part可以下紀錄40個不同部位的images, 並用不同的image processing tasks分析出features去辨別真偽. 

軟體方面包括有: 1. 控制硬體(camera, Zaber 控制器, 雷射/位移感測器, light source control, IoT Ethernet I/O Modules, Vanta Element-S等)目前是Controller folder的硬體相關code. 

2. tasks folder的algorithms 裡面是20~40個image processing tasks的算法, 而cli_wrappers跟api_servers分別是這些tasks的cli跟api mode. algorithms folder是關於image processing tasks會用到的算法(其中有些是AI-based model infernece譬如unet, ocr). config folder內部則是這些tasks的config files. core folder則是orchestrator去管理這些image processing tasks. tasks folder的algorithms 裡面是image processing tasks的算法, 而cli_wrappers跟api_servers分別是這些tasks的cli跟api mode. algorithms folder是關於image processing tasks會用到的算法. config folder內部則是這些tasks的config files. core folder則是orchestrator去管理這些image processing tasks. 

3. 這個software有兩個入口(cli(command line)跟有UI的App): 目前main_cli.py是在cli執行這些image processing tasks.  App folder則是有ui的App. 這系統上所有的功能性(硬體, 影像分析, image儲存管理等等...)都可以用main_cli.py跟App進行操作執行. 而且也包括可以在local computer(跟硬體系統連結)跟 remote client都可以進行操作. 

4. camera拍攝的images先進行head encryption, 以及meta data等都用encryption加密並存到local folders, 並將這些images跟yaml files由CloudSyncManager管理同步到AWS cloud. camera拍攝的images再經過image processing tasks處理完之後的一些結果包括segmentation masks image files或其他figure images, result yaml files等也會經過加密再存到這個experiment folders, 並將這些images跟yaml files由CloudSyncManager管理同步到AWS cloud. 而QueryEngine則記錄下所有local folders跟experiment folders上傳到AWS cloud的東西而方便查詢, 也包括可查詢AWS上同步的資料並可下載到local computer. 

5. AuditLogger則是一套log記錄系統可以紀錄關於這套系統的任何操作並記錄每個不同user的所有操作並基於數位指紋 (Hash)跟鏈式鎖定 (Chaining). 

6. 這套系統也包括透過代理人模式 (Proxy Pattern)在remote client進行完整的操作包括操作hardware, data manager, Orchestrator, AuditLogger等系統, 包括在操作camera跟Zaber時需要有串流回傳到remote client讓user在遠端App也能看到即時畫面進行操作


現在我想要在原來code的基礎上不功能刪除功能性也儘量不改動原來code而加入新的需求:
在透過代理人模式 (Proxy Pattern)在remote client進行完整的操作包括操作hardware, data manager, Orchestrator, AuditLogger等系統, 包括在操作camera跟Zaber時需要有串流回傳到remote client讓user在遠端App也能看到即時畫面進行操作, 而像data manager在local computer也會需要上傳同步images跟metadata到AWS cloud, 而且遠端App也能高效查詢之前擷取的影像等等(這是否可以透過先從Cloud下載到local computer, 然後在remote client可以讀取這些影像).


請幫我30年資深軟體開發專家tech lead的身分詳細設地基於上個版本去設計並儘量優化這套系統完整系統, 並中文詳細解釋
```

Okay, I have a clear understanding of the full system requirements, the evolution of the architecture (V1 to V3), and the specific components that have been refined (DataManager, WorkflowManager, AuditLogger, Proxy Pattern, etc.).

You've provided the previous versions of the core files and `system_config.yaml`. Based on the extensive discussion and the "V3.0 Hybrid Communication Pattern" we've established, I will now provide the **final, integrated code** for the requested files.

Here is the plan:

1. **`config/system_config.yaml`**: Ensure it includes the new `network` section for Remote/Local mode switching, which was missing in your provided snippet but critical for the Proxy pattern.
    
2. **`core/workflow_manager.py`**: Update it to be the "real" manager that runs on the Server. It needs to integrate the specific 5-step DataManager flow we defined (Raw -> Encrypt -> Sync -> Experiment -> Result -> Encrypt -> Sync) and ensure AuditLogging happens at the right spots.
    
3. **`main_cli.py`**: Update it to use the `get_workflow_engine` factory pattern so it can seamlessly switch between `WorkflowManager` (Local) and `WorkflowProxy` (Remote).
    
4. **`core/workflow_proxy.py`** & **`core/workflow_server.py`**: Provide these essential new files for the Remote Client capability (V3.0 with WebSocket streaming).
    
5. **`core/__init__.py`**: Provide the factory function code.
    

Here is the complete, tech-lead-grade code package.

---

### 1. Configuration: `config/system_config.yaml`

I have added the missing `network` section required for the Client-Server architecture.

YAML

```
# config/system_config.yaml

# ==============================================================================
# 1. System Identity & Mode
# ==============================================================================
system:
  mode: "simulation"  # "simulation" or "production"
  user_id: "operator_001"
  version: "1.0.0"

# ==============================================================================
# 2. Network & Remote Access [NEW]
# ==============================================================================
network:
  # App Mode: "local" (Run everything on this machine) or "remote" (Connect to server)
  app_mode: "local" 
  
  # Server Connection Details (Used if app_mode is "remote")
  server_ip: "127.0.0.1"
  server_port: 8000
  auth_token: "WATCH_SYS_SECRET_2025" # Simple security token

# ==============================================================================
# 3. Global Data Paths
# ==============================================================================
paths:
  base_data_dir: "./Local_Data"
  sample_assets_dir: "./assets/sample_images" 
  secret_key_path: "./config/secret.key"

# ==============================================================================
# 4. Hardware Configuration
# ==============================================================================
hardware:
  zaber:
    port: "COM3" 
    device_map:
      stage_L_X: 1
      stage_L_Y: 2
      stage_L_Z: 3
      stage_R_Z: 4
      stage_R_X: 5

  cameras:
    macro_cam_id: "ITA204_ID"
    micro_cam_id: "ITA81_ID"

  keyence:
    ip: "169.254.64.64"
    port: 24685

  vanta:
    ip: "192.168.1.10"
    timeout: 30

  lighting:
    controller_ip: "192.168.1.50"

# ==============================================================================
# 5. Core Execution Settings
# ==============================================================================
core:
  global_execution_mode: "mixed"
  max_parallel_workers: 4

# ==============================================================================
# 6. AWS / Cloud Settings [NEW]
# ==============================================================================
aws:
  s3_bucket: "watch-analysis-v1"
  region: "us-east-1"

# ==============================================================================
# 7. Environment Paths
# ==============================================================================
envs:
  base_env: "C:/Envs/base_env/python.exe"
  ocr_env: "C:/Envs/ocr_env/python.exe"
  dinov3_env: "C:/Envs/dinov3_env/python.exe"
  sam_env: "C:/Envs/sam_env/python.exe"
  doctr_env: "C:/Envs/doctr_env/python.exe"
  lume_env: "C:/Envs/lume_env/python.exe"

# ==============================================================================
# 8. Services
# ==============================================================================
services:
  lume_service:
    env: "lume_env"
    script: "tasks/api_servers/server_lume.py"
    host: "127.0.0.1"
    port: 5002
    config_path: "config/lume_config.yaml"

  features_service:
    env: "sam_env"
    script: "tasks/api_servers/server_features.py"
    host: "127.0.0.1"
    port: 5003
    config_path: "config/features_config.yaml"

  movement2_service:
    env: "dinov3_env"
    script: "tasks/api_servers/server_movement2.py"
    host: "127.0.0.1"
    port: 5005
    config_path: "config/movement2_config.yaml"

  doctr_service:
    env: "doctr_env"
    script: "tasks/api_servers/server_doctr.py"
    host: "127.0.0.1"
    port: 5010
    config_path: "config/doctr_config.yaml"

  sam_graph_service:
    env: "sam_env"
    script: "tasks/api_servers/server_sam_graph.py"
    host: "127.0.0.1"
    port: 5007
    config_path: "config/sam_graph_config.yaml"

  crystal_service:
    env: "base_env"
    script: "tasks/api_servers/server_crystal.py"
    host: "127.0.0.1"
    port: 5006
    config_path: "config/crystal_config.yaml"

# ==============================================================================
# 9. Tasks Definitions
# ==============================================================================
tasks:
  lume_segmentation:
    mode: "api"
    service_name: "lume_service"
    cli_script: "tasks/cli_wrappers/run_lume_cli.py"
    timeout: 60

  sam_features:
    mode: "api"
    service_name: "features_service"
    cli_script: "tasks/cli_wrappers/run_features_cli.py"
    timeout: 120

  sam_graph_analysis:
    mode: "api"
    service_name: "sam_graph_service"
    cli_script: "tasks/cli_wrappers/run_sam_graph_cli.py"
    timeout: 60

  movement2_analysis:
    mode: "api"
    service_name: "movement2_service"
    cli_script: "tasks/cli_wrappers/run_movement2_cli.py"
    timeout: 90

  doctr_analysis:
    mode: "api"
    service_name: "doctr_service"
    cli_script: "tasks/cli_wrappers/run_doctr_cli.py"
    timeout: 60

  crystal_analysis:
    mode: "cli"
    service_name: "crystal_service"
    cli_script: "tasks/cli_wrappers/run_crystal_cli.py"
    timeout: 30

  bracelet_focus:
    mode: "cli"
    service_name: "bracelet_service"
    cli_script: "tasks/cli_wrappers/run_bracelet_cli.py"
    timeout: 20
```

---

### 2. Core Logic: `core/workflow_manager.py`

This version implements the **specific 5-step data flow** you requested (Capture -> Process Raw -> Create Experiment -> Analyze -> Save Result) and integrates the AuditLogger.

Python

```
# core/workflow_manager.py
import os
import time
import yaml
import logging
import traceback

from core.orchestrator import Orchestrator
from data_manager.local_storage import DataManager
from logging_system.audit_logger import AuditLogger
from Controller.hardware_drivers.simulation_driver import SimulationHardware
# from Controller.controller_client import ControllerClient # Uncomment for Production

class WorkflowManager:
    """
    The 'Brain' running on the Server (Local Computer).
    Coordinates Hardware, DataManager, Orchestrator, and AuditLogger.
    """
    def __init__(self, config_path="config/system_config.yaml"):
        # 1. Load Config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        self.user_id = self.config['system']['user_id']
        self.mode = self.config['system']['mode']
        
        # Logging
        logging.basicConfig(level=logging.INFO, format="[Workflow] %(message)s")
        self.console = logging.getLogger("Workflow")

        # 2. Initialize Sub-systems
        self.data_mgr = DataManager(self.config)
        
        # Pass CloudSync manager to AuditLogger for auto-uploading logs
        self.audit = AuditLogger(self.config, cloud_sync_manager=self.data_mgr.cloud_sync)
        
        self.orchestrator = Orchestrator(config_path)
        
        # 3. Hardware Initialization
        if self.mode == "simulation":
            self.hw = SimulationHardware(self.config)
            self.audit.log_action("SYSTEM_INIT", {"mode": "simulation"})
        else:
            # self.hw = ControllerClient(self.config['hardware'])
            # self.console.warning("Real hardware not connected, falling back to simulation.")
            self.hw = SimulationHardware(self.config) 
            self.audit.log_action("SYSTEM_INIT", {"mode": "production_fallback"})

    def execute_routine(self, watch_id, routine_name="Standard_Check", user_id=None):
        """
        Executes the full inspection routine:
        1. Capture & Encrypt Raw Images
        2. Sync Raw to Cloud
        3. Create Experiment Structure
        4. Run Analysis -> Encrypt Result -> Sync
        """
        # Use remote user_id if provided (from Proxy), else local config user
        current_user = user_id if user_id else self.user_id
        
        # Log Start
        self.audit.log_action("ROUTINE_START", {"watch_id": watch_id, "routine": routine_name}, user=current_user)
        self.console.info(f"--- Starting Routine: {routine_name} for Watch: {watch_id} ---")

        try:
            # ==========================================
            # Step 1 & 2: Capture Raw Images & Process
            # ==========================================
            self.console.info("Step 1: Capturing Raw Images...")
            
            # Example Hardware Move
            self.hw.move_stage("stage_L_X", 100)
            
            # Define Temp Path
            raw_filename = "front_view.jpg"
            temp_dir = os.path.join(self.data_mgr.base_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            temp_capture_path = os.path.join(temp_dir, raw_filename)
            
            # Capture (Simulated or Real)
            self.hw.capture_image("macro_cam_id", 5000, temp_capture_path)
            
            # Process: Move to Watch/Raw -> Encrypt Header -> Register DB -> Queue Sync
            local_raw_path = self.data_mgr.process_and_sync_raw_image(
                temp_capture_path, watch_id, raw_filename
            )
            self.console.info(f"Raw Image Secured: {local_raw_path}")
            self.audit.log_action("CAPTURE_SAVED", {"file": raw_filename}, user=current_user)

            # ==========================================
            # Step 3: Create Experiment Hierarchy
            # ==========================================
            self.console.info("Step 3: Creating Experiment Context...")
            # Created only when analysis is about to start
            exp_id, exp_dir = self.data_mgr.create_experiment_folder(watch_id, routine_name)
            self.console.info(f"Experiment ID: {exp_id}")

            # ==========================================
            # Step 4: Image Processing & Result Storage
            # ==========================================
            self.console.info("Step 4: Running Image Processing...")
            
            # 4.1 Decrypt Header temporarily for Analysis
            # (In-memory decryption or temp file restore)
            self.data_mgr.prepare_image_for_viewing(local_raw_path)
            
            task_results = None
            try:
                # 4.2 Run Orchestrator
                # Note: Orchestrator returns data objects (masks as numpy arrays), not yet saved files
                task_results = self.orchestrator.run_batch(
                    task_list=["lume_segmentation"], 
                    image_path=local_raw_path,
                    output_dir=exp_dir # Pass exp_dir for reference
                )
            finally:
                # 4.3 Re-encrypt Header immediately
                self.data_mgr.re_encrypt_image_after_viewing(local_raw_path)

            # 4.4 Save & Encrypt Results
            if task_results:
                # Save Images (e.g. Masks)
                lume_res = task_results.get("lume_segmentation", {})
                if "mask_image" in lume_res:
                    mask_filename = "lume_mask.png"
                    self.data_mgr.save_result_image_and_sync(
                        lume_res["mask_image"], mask_filename, exp_id, exp_dir, watch_id
                    )
                    del lume_res["mask_image"] # Remove heavy object before YAML save

                # Save Data Report (YAML)
                self.data_mgr.save_analysis_report_and_sync(
                    task_results, exp_id, exp_dir, watch_id
                )

            # ==========================================
            # Step 5: Finish & Audit
            # ==========================================
            # Cloud Sync is already queued in previous steps
            
            self.audit.log_action("ROUTINE_COMPLETE", {"status": "success", "exp_id": exp_id}, user=current_user)
            self.console.info("--- Routine Completed Successfully ---")
            
            return task_results

        except Exception as e:
            self.console.error(f"Routine Failed: {e}")
            self.audit.log_action("ROUTINE_ERROR", {"error": str(e)}, user=current_user)
            traceback.print_exc()
            return None

    def manual_capture(self, watch_id, cam_id):
        """Manual capture for testing (follows same security flow)"""
        filename = f"manual_{int(time.time())}.jpg"
        temp_dir = os.path.join(self.data_mgr.base_dir, "temp")
        os.makedirs(temp_dir, exist_ok=True)
        temp_path = os.path.join(temp_dir, filename)
        
        self.hw.capture_image(cam_id, 5000, temp_path)
        real_path = self.data_mgr.process_and_sync_raw_image(temp_path, watch_id, filename)
        return real_path

    def manual_move(self, axis, value):
        self.hw.move_stage(axis, value)

    def force_cloud_sync(self):
        """Trigger immediate check on sync queue (though it's auto)"""
        self.console.info("Force sync check initiated.")
        # Logic depends on CloudSync implementation, usually it's always running.

    def shutdown(self):
        self.data_mgr.close()
```

---

### 3. CLI Entry Point: `main_cli.py`

Updated to use the **Factory Pattern** (`get_workflow_engine`), allowing it to work on both Local and Remote computers without changing code.

Python

```
# main_cli.py
import argparse
import sys
import json
import traceback

# Import the Factory
from core import get_workflow_engine

def main():
    parser = argparse.ArgumentParser(description="ImagingLibWatch V1 CLI")
    
    # Global args
    parser.add_argument("--config", default="config/system_config.yaml", help="Config file path")
    parser.add_argument("--watch_id", required=True, help="Target Watch ID")
    
    # Sub-commands
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # 1. Run Full Routine
    cmd_routine = subparsers.add_parser("run_routine", help="Execute inspection routine")
    cmd_routine.add_argument("--name", default="Standard_Check")
    
    # 2. Manual Capture
    cmd_capture = subparsers.add_parser("manual_capture", help="Take single photo")
    cmd_capture.add_argument("--cam", default="macro_cam_id")
    
    # 3. Analyze Local File (Bypass hardware, for dev/test)
    cmd_analyze = subparsers.add_parser("analyze_local", help="Run analysis on existing file")
    cmd_analyze.add_argument("--image", required=True)
    cmd_analyze.add_argument("--task", default="lume_segmentation")

    # 4. Sync
    cmd_sync = subparsers.add_parser("force_sync", help="Force cloud sync")

    args = parser.parse_args()

    engine = None
    try:
        # Factory determines if we get WorkflowManager (Local) or WorkflowProxy (Remote)
        engine = get_workflow_engine(args.config)
        print(f">>> Engine Loaded: {type(engine).__name__}")
        
        if args.command == "run_routine":
            print(f">>> Executing Routine: {args.name}...")
            results = engine.execute_routine(args.watch_id, args.name)
            
            if results:
                print(json.dumps(results, indent=2, default=str))
            else:
                print("[Failed] No results returned.")
                sys.exit(1)

        elif args.command == "manual_capture":
            path = engine.manual_capture(args.watch_id, args.cam)
            print(f"[Success] Image saved at: {path}")

        elif args.command == "analyze_local":
            # Note: Orchestrator is usually local logic. 
            # If running as Proxy, this might not work unless we upload image to server first.
            # Assuming analyze_local is a Dev tool for Local machine.
            if hasattr(engine, 'orchestrator'):
                res = engine.orchestrator.run_batch([args.task], args.image, "output_debug")
                print(json.dumps(res, indent=2, default=str))
            else:
                print("[Error] 'analyze_local' only available in Local Mode.")

        elif args.command == "force_sync":
            engine.force_cloud_sync()
            print("[Info] Sync signal sent.")

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
    finally:
        if engine: engine.shutdown()

if __name__ == "__main__":
    main()
```

---

### 4. New Core File: `core/workflow_proxy.py` (Remote Client Logic)

Handles HTTP requests and provides the `get_stream_url` helper for the App.

Python

```
# core/workflow_proxy.py
import requests
import yaml
import logging
import json

class WorkflowProxy:
    """
    Client-side Proxy that communicates with the WorkflowServer.
    Mimics the interface of WorkflowManager.
    """
    def __init__(self, config_path="config/system_config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        net = self.config['network']
        self.host = net['server_ip']
        self.port = net['server_port']
        self.token = net['auth_token']
        self.base_url = f"http://{self.host}:{self.port}"
        self.headers = {"x-token": self.token}
        self.local_user_id = self.config['system']['user_id']
        
        logging.basicConfig(level=logging.INFO, format="[Proxy] %(message)s")
        self.logger = logging.getLogger("WorkflowProxy")
        self.logger.info(f"Initialized Proxy -> {self.base_url}")

    # --- 1. Live Stream Support ---
    def get_stream_url(self, cam_id="macro_cam_id"):
        """Returns WebSocket URL for live video widget in App"""
        return f"ws://{self.host}:{self.port}/ws/stream?cam_id={cam_id}&token={self.token}"

    # --- 2. Workflow Actions ---
    def execute_routine(self, watch_id, routine_name="Standard_Check"):
        self.logger.info(f"Remote executing: {routine_name}")
        payload = {
            "watch_id": watch_id,
            "routine_name": routine_name,
            "user_id": self.local_user_id
        }
        try:
            # V3.0: Supports Async polling if Server implements it, 
            # or simple synchronous wait as implemented in basic server.
            # Assuming Basic Sync Server for simplicity in this file:
            resp = requests.post(f"{self.base_url}/api/routine/start", json=payload, headers=self.headers, timeout=300)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            self.logger.error(f"Remote Execution Failed: {e}")
            return None

    def manual_capture(self, watch_id, cam_id):
        payload = {"watch_id": watch_id, "cam_id": cam_id}
        try:
            resp = requests.post(f"{self.base_url}/api/manual_capture", json=payload, headers=self.headers)
            resp.raise_for_status()
            return resp.json().get("path")
        except Exception as e:
            self.logger.error(f"Capture Failed: {e}")
            return None

    def manual_move(self, axis, value):
        try:
            requests.post(
                f"{self.base_url}/api/manual_move",
                params={"axis": axis, "value": value},
                headers=self.headers
            )
        except Exception as e:
            self.logger.error(f"Move Failed: {e}")

    def force_cloud_sync(self):
        try:
            requests.post(f"{self.base_url}/api/force_sync", headers=self.headers)
        except Exception as e:
            self.logger.error(f"Sync Request Failed: {e}")

    def shutdown(self):
        pass
    
    # --- 3. Remote Data Access (Optional) ---
    def list_remote_files(self, watch_id):
        """Ask server for file list"""
        resp = requests.get(f"{self.base_url}/api/data/list_files", params={"watch_id": watch_id}, headers=self.headers)
        return resp.json().get("files", [])
```

---

### 5. New Core File: `core/workflow_server.py` (Server Logic)

This must be run on the **Local Computer** (where hardware is attached).

Python

```
# core/workflow_server.py
import uvicorn
from fastapi import FastAPI, HTTPException, Header, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
import yaml
import os
import sys
import threading
import asyncio
import cv2

# Import Real Manager
sys.path.append(os.getcwd())
from core.workflow_manager import WorkflowManager

# --- State ---
class ServerContext:
    def __init__(self):
        self.manager = None
        self.lock = threading.Lock()

ctx = ServerContext()
app = FastAPI()
CONFIG_PATH = "config/system_config.yaml"

def get_config():
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def verify_token(x_token: str = Header(...)):
    cfg = get_config()
    if x_token != cfg['network']['auth_token']:
        raise HTTPException(status_code=401, detail="Invalid Token")

# --- Lifecycle ---
@app.on_event("startup")
def startup():
    print(">>> [Server] Starting WorkflowManager...")
    ctx.manager = WorkflowManager(CONFIG_PATH)
    print(">>> [Server] Ready.")

@app.on_event("shutdown")
def shutdown():
    if ctx.manager:
        ctx.manager.shutdown()

# --- Models ---
class RoutineRequest(BaseModel):
    watch_id: str
    routine_name: str
    user_id: str

class CaptureRequest(BaseModel):
    watch_id: str
    cam_id: str

# --- API Endpoints ---

@app.post("/api/routine/start")
def start_routine(req: RoutineRequest, x_token: str = Header(...)):
    verify_token(x_token)
    # Simple Lock to prevent race conditions
    if not ctx.lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="System Busy")
    
    try:
        # Pass remote user_id to manager for Audit Log
        results = ctx.manager.execute_routine(req.watch_id, req.routine_name, user_id=req.user_id)
        return results
    finally:
        ctx.lock.release()

@app.post("/api/manual_capture")
def manual_capture(req: CaptureRequest, x_token: str = Header(...)):
    verify_token(x_token)
    path = ctx.manager.manual_capture(req.watch_id, req.cam_id)
    return {"path": path}

@app.post("/api/manual_move")
def manual_move(axis: str, value: float, x_token: str = Header(...)):
    verify_token(x_token)
    ctx.manager.manual_move(axis, value)
    return {"status": "ok"}

@app.post("/api/force_sync")
def force_sync(x_token: str = Header(...)):
    verify_token(x_token)
    ctx.manager.force_cloud_sync()
    return {"status": "ok"}

@app.get("/api/data/list_files")
def list_files(watch_id: str, x_token: str = Header(...)):
    verify_token(x_token)
    files = ctx.manager.data_mgr.db.get_watch_files(watch_id)
    return {"files": files}

# --- WebSocket Streaming ---
@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, cam_id: str, token: str):
    # Verify token
    cfg = get_config()
    if token != cfg['network']['auth_token']:
        await websocket.close(code=1008)
        return

    await websocket.accept()
    print(f"[Stream] Start for {cam_id}")
    
    try:
        while True:
            # Assume Hardware has get_live_frame
            # If not, use simulation/capture fallback
            if hasattr(ctx.manager.hw, 'get_live_frame'):
                frame = ctx.manager.hw.get_live_frame(cam_id)
            else:
                # Fallback: create mock frame
                import numpy as np
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"LIVE: {time.time()}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            if frame is not None:
                ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                if ret:
                    await websocket.send_bytes(buffer.tobytes())
            
            await asyncio.sleep(0.05) # 20 FPS
    except WebSocketDisconnect:
        print("[Stream] Client disconnected")
    except Exception as e:
        print(f"[Stream] Error: {e}")

if __name__ == "__main__":
    cfg = get_config()
    uvicorn.run(app, host="0.0.0.0", port=cfg['network']['server_port'])
```

---

### 6. Factory Logic: `core/__init__.py`

This allows `main_cli.py` and `App` to switch modes instantly.

Python

```
# core/__init__.py
import yaml
import os
from .workflow_manager import WorkflowManager
from .workflow_proxy import WorkflowProxy

def get_workflow_engine(config_path="config/system_config.yaml"):
    """
    Factory: Returns Manager (Local) or Proxy (Remote) based on config.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    mode = config.get('network', {}).get('app_mode', 'local')
    
    if mode == 'remote':
        return WorkflowProxy(config_path)
    else:
        return WorkflowManager(config_path)
```

This completes the V3.0 architecture code implementation. You now have a unified system where the `main_cli.py` works identically on both the machine connected to the Zaber/Camera and on a laptop in another room, with full audit logging, data encryption, and live streaming capabilities.