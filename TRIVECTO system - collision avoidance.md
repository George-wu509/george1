

|     |     |
| --- | --- |
|     |     |
|     |     |
|     |     |
|     |     |

目前的主要防撞路徑是 `UnifiedHardwareDriver.move_compound()`：先檢查門禁鎖定、預測目標碰撞，再把移動拆成「Z 退避 → XY/旋轉 → Z 接近」的安全段落。

### 程式與設定位置

- 總設定：[hardware_config.yaml](D:\Provenance Project\ImagingLibWatch\config\hardware_config.yaml:78)
- 安全路徑、Strap 高風險規則、E-stop：[unified_driver.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_drivers\unified_driver.py:475)
- Zaber 軸界限、韌體限位、單軸防呆：[hardware_managers.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_managers.py:249)
- MQTT 門禁監聽：[hardware_managers.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_managers.py:4261)
- 幾何數位孿生碰撞預測：[digital_twin_guard.py](D:\Provenance Project\ImagingLibWatch\Controller\kinematics\digital_twin_guard.py:62)
- 舊版 `SafeZoneManager`：[safe_zone_manager.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_drivers\safe_zone_manager.py:11)；目前沒有被正式 driver 建立或呼叫，屬未接線的舊機制。

### 現有保護機制

|層級|定義與作用|
|---|---|
|軸行程限制|X `0–435 mm`、Y `0–292 mm`、Z `0–150 mm`、R_X `0–90°`。程式會 clamp，連線後也會寫入 Zaber 韌體 `limit.min/max`。|
|R_Z 轉動|`rotate_z_limit: []` 表示連續軸；程式會選擇最短等效角度旋轉，避免不必要多圈，但沒有機械纜線扭轉/圈數上限。|
|運動平滑|linear/rotation motion profiles 限制速度、加減速與 S-curve，降低慣性撞擊，但本身不是碰撞偵測。|
|一般安全路徑|Side/Crown 等路徑先移到 `safe_retract_z`（watch 為 20 mm；box 為 10 mm），再平移/旋轉，最後回目標 Z。|
|R_X/R_Z interlock|觸發條件包含大角度 R_X、R_Z 穿越危險角、XY 大位移等；會先退 Z、把 R_X 收至 `0°`、完成 R_Z/XY、再恢復 R_X。|
|Strap 高風險|R_X ≥ 30° 且 R_Z 接近/穿越 90° 或 270° 時，強制至 Y=160、Z=70 的已驗證避讓姿態；先折回 R_X=0°，再轉 R_Z，最後才接近目標。|
|Strap 白名單|`4029–4032` 必須同時匹配完整五軸姿態才允許，並經由 Y=160/Z=70 進出。|
|Strap 牆面包絡|以 200 mm Strap、Y=0 牆面、30 mm 最小淨空估算；在 R_X > 30° 時拒絕端點會進入危險區的目標。|
|讀回確認|Strap 每個涉及旋轉的 segment 前，都會讀回 Y/Z；R_Z 旋轉前還會確認 R_X 已折回安全角度。|
|門禁 E-stop|MQTT 收到 `DI0 = 0` 時，鎖定系統、停止所有 Zaber 軸、關燈、停相機；必須手動 reset 才可恢復。|

Strap 的路徑規劃與高風險判斷集中在 [_plan_motion_segments](D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py)，讀回保護在 [這兩個檢查](D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py)。

### 目前「怎麼確保不撞」

若所有正式拍攝、移動、對焦流程都走 `driver.move_compound(...)`，保護順序是：

```
門禁未鎖定
  → 目標姿態數位孿生預檢
  → 軸行程限制
  → 依情境產生退避/折臂/旋轉/接近的分段路徑
  → Strap 旋轉前讀回 Y/Z/R_X
  → Zaber 韌體限位與運動 profile 執行
```

### 重要缺口／風險

1. 數位孿生目前實質失效。  
    [`safe_zone`](D:\Provenance Project\ImagingLibWatch\config\hardware_config.yaml:110) 的 platform、macro、micro bounding box 全部是 `0.0`；程式會把這些零值覆蓋預設尺寸，因此 AABB 碰撞模型退化，無法代表鏡頭、治具、手錶的實際體積。
    
2. 數位孿生只檢查「最終姿態」，不掃描每個移動軌跡；真正的路徑安全主要仰賴規則式退避與 staging，而非連續幾何碰撞檢查。
    
3. 有低階入口可繞過安全規劃。  
    例如 [controller_server.py](D:\Provenance Project\ImagingLibWatch\Controller\controller_server.py:100) 的 `move_stage`、[workflow_manager.py](D:\Provenance Project\ImagingLibWatch\core\workflow_manager.py:3062) 的 manual move，直接呼叫 `zaber.move_axis()`，只有軸界限 clamp，沒有門禁鎖、數位孿生、Strap staging 或路徑拆段。應限制這些入口只用於維護模式，或改統一委派給 `move_compound()`。
    
4. MQTT 門禁斷線目前只記錄 warning，沒有 fail-safe E-stop；這是軟體/網路型保護，不能視為安全等級的硬體 interlock。對人員防護應有獨立硬體安全迴路直接切斷馬達使能。
    
5. R_Z 無圈數/纜線保護。最短路徑可降低風險，但不等於可無限轉動；應依實際配線加入累積圈數、軟體角度窗或硬體 slip ring 的明確規則。
    

已存在相當多路徑規劃測試，例如 [test_unified_motion_planning.py](D:/Provenance Project/ImagingLibWatch/tests/test_unified_motion_planning.py)。我嘗試執行，但目前 Python 環境缺少 `pytest` 與 `PyYAML`，所以沒有做出可執行的測試驗證。此次沒有修改檔案。