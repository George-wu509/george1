

|                                          |     |
| ---------------------------------------- | --- |
| [[主要防撞路徑]]                               |     |
| [[#### 完整防撞路徑及現有保護機制]]                   |     |
| [[#### move_compound function analysis]] |     |
|                                          |     |

目前的主要防撞路徑是 `UnifiedHardwareDriver.move_compound()`：先檢查門禁鎖定、預測目標碰撞，再把移動拆成「Z 退避 → XY/旋轉 → Z 接近」的安全段落。

### 程式與設定位置

- 總設定：[hardware_config.yaml](D:\Provenance Project\ImagingLibWatch\config\hardware_config.yaml:78)
- 安全路徑、Strap 高風險規則、E-stop：[unified_driver.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_drivers\unified_driver.py:475)
- Zaber 軸界限、韌體限位、單軸防呆：[hardware_managers.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_managers.py:249)
- MQTT 門禁監聽：[hardware_managers.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_managers.py:4261)
- 幾何數位孿生碰撞預測：[digital_twin_guard.py](D:\Provenance Project\ImagingLibWatch\Controller\kinematics\digital_twin_guard.py:62)
- 舊版 `SafeZoneManager`：[safe_zone_manager.py](D:\Provenance Project\ImagingLibWatch\Controller\hardware_drivers\safe_zone_manager.py:11)；目前沒有被正式 driver 建立或呼叫，屬未接線的舊機制。

### 現有保護機制

| 層級                | 定義與作用                                                                                          |
| ----------------- | ---------------------------------------------------------------------------------------------- |
| 軸行程限制             | X `0–435 mm`、Y `0–292 mm`、Z `0–150 mm`、R_X `0–90°`。程式會 clamp，連線後也會寫入 Zaber 韌體 `limit.min/max`。 |
| R_Z 轉動            | `rotate_z_limit: []` 表示連續軸；程式會選擇最短等效角度旋轉，避免不必要多圈，但沒有機械纜線扭轉/圈數上限。                               |
| 運動平滑              | linear/rotation motion profiles 限制速度、加減速與 S-curve，降低慣性撞擊，但本身不是碰撞偵測。                            |
| 一般安全路徑            | Side/Crown 等路徑先移到 `safe_retract_z`（watch 為 20 mm；box 為 10 mm），再平移/旋轉，最後回目標 Z。                  |
| R_X/R_Z interlock | 觸發條件包含大角度 R_X、R_Z 穿越危險角、XY 大位移等；會先退 Z、把 R_X 收至 `0°`、完成 R_Z/XY、再恢復 R_X。                         |
| Strap 高風險         | R_X ≥ 30° 且 R_Z 接近/穿越 90° 或 270° 時，強制至 Y=160、Z=70 的已驗證避讓姿態；先折回 R_X=0°，再轉 R_Z，最後才接近目標。          |
| Strap 白名單         | `4029–4032` 必須同時匹配完整五軸姿態才允許，並經由 Y=160/Z=70 進出。                                                 |
| Strap 牆面包絡        | 以 200 mm Strap、Y=0 牆面、30 mm 最小淨空估算；在 R_X > 30° 時拒絕端點會進入危險區的目標。                                 |
| 讀回確認              | Strap 每個涉及旋轉的 segment 前，都會讀回 Y/Z；R_Z 旋轉前還會確認 R_X 已折回安全角度。                                      |
| 門禁 E-stop         | MQTT 收到 `DI0 = 0` 時，鎖定系統、停止所有 Zaber 軸、關燈、停相機；必須手動 reset 才可恢復。                                  |

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





#### 完整防撞路徑及現有保護機制
```
請幫我整理目前的完整防撞路徑及現有保護機制, 程式與設定位置包括code在哪裡, config在哪裡?, 以及似乎有很多是很類似的請整理
```
以下是依目前工作樹程式碼做的完整防撞盤點；本次只分析，沒有修改檔案。

## 一頁結論

目前有四層保護：

1. Zaber 軸範圍與韌體 soft limit。
2. `UnifiedHardwareDriver` 規則式安全路徑規劃。
3. `DigitalTwinGuard` 幾何碰撞預測。
4. 門禁 MQTT → 急停 → 軟體鎖定。

但現況有三個重要缺口：

- `hardware.zaber.safe_zone` 的碰撞盒尺寸全部是 `0`，Digital Twin 退化成零體積幾何，幾乎沒有有效碰撞保護。
- `strap_macro_cam1_scan.yaml` 又設定 `require_nonzero_collision_envelopes: false`，所以 production preflight 不會因零尺寸而拒絕。
- 手動移動、舊 controller server、部分 autofocus 與 `execute_template_point` 內部路徑會繞過部分保護。

此外目前 [system_config.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.yaml:1) 是 `mode: simulation`，因此從標準 Workflow 啟動時不會控制實體硬體，門禁 MQTT 也不會啟用。

---

## 完整正式移動路徑

### A. 最完整路徑：`move_compound`

````
```mermaid
flowchart TD
    A["App / Keyence / Exit cleanup"] --> B["move_compound"]
    B --> C["check_safety_lock"]
    C --> D["讀取目前 5 軸位置"]
    D --> E["可選運動學補償"]
    E --> F["DigitalTwinGuard 檢查最終目標"]
    F --> G["_plan_motion_segments"]
    G --> H["依優先序產生 staging segments"]
    H --> I["Strap 旋轉前 Y/Z 與 R_X readback 驗證"]
    I --> J["ZaberManager move_axis / move_axes_absolute"]
    J --> K["軟體 clamp + firmware soft limits"]
```
````

主入口在 [unified_driver.py (line 475)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:475)：

- 先檢查門禁鎖。
- 讀取目前五軸座標。
- 可選擇套用旋轉偏心補償。
- Digital Twin 檢查「最終位置」。
- 呼叫路徑規劃器。
- 旋轉 segment 前驗證 Strap staging readback。
- 最後交給 ZaberManager 執行。

注意：Digital Twin 只檢查最終目標，不會沿每個中間 segment 連續取樣檢查碰撞。

### B. 一般 Template capture 路徑

`execute_template_point()` 在 [unified_driver.py (line 8591)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:8591)：

- 有執行 `check_safety_lock()`。
- 有呼叫 `_plan_motion_segments()`。
- 但沒有經過 `move_compound()`。
- 因此沒有一般目標的 Digital Twin 檢查。
- 執行 segment 時也沒有 `move_compound` 那組 Strap rotation staging readback 驗證。

也就是說，目前「正式拍攝路徑」有規則式 dog-leg/interlock，但保護程度弱於直接使用 `move_compound()`。

### C. Strap macro_cam_1 掃描

這是目前最保守的一條路徑：

1. 驗證整份 scan config。
2. 驗證每個 pose 的軸範圍及最低 Y。
3. 建立 Y → Z → fold R_X → rotate R_Z → restore R_X → X → Y → Z 的外層 waypoint。
4. 每個 waypoint 再送入 `move_compound()`。
5. 每個 waypoint 完成後核對五軸 readback。

主要位置：

- Config preflight：[App/main.py (line 34250)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34250)
- Pose 限位：[App/main.py (line 34489)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34489)
- 安全 compound wrapper：[App/main.py (line 34525)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34525)
- View transition/readback：[App/main.py (line 34593)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34593)
- Waypoint 產生器：[strap_macro1_scan.py (line 1151)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_macro1_scan.py:1151)

### D. Strap Keyence 探針

Keyence 專用路徑在 [unified_driver.py (line 3296)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:3296)：

- 先拒絕非數值、非有限值及超過軸限位的目標。
- Strap 先退到固定 staging。
- 檢查 safety lock。
- 執行 Digital Twin。
- 執行 `_plan_motion_segments()`。
- 旋轉前核對 staging 與 R_X fold。
- 完成後核對五軸 readback。

這條路徑和 macro_cam_1 scan 是目前保護最完整的兩條。

---

## `_plan_motion_segments` 實際優先序

核心位於 [unified_driver.py (line 606)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:606)。它是 first-match-return，實際順序為：

1. `strap_special_staging`
    
    特定 `internalnum1` 必須同時匹配白名單中的完整五軸位置；進入、離開都經 Y=160、Z=70。白名單是 `4029–4032`。
    
2. Strap Keyence staging
    
    探針移動使用同一套 special staging contract。
    
3. Strap wall envelope
    
    依公式檢查：
    
    `clearance = Y - wall_y - half_length × abs(sin(R_Z))`
    
4. Large R_X transition
    
    大幅 R_X 變化時，使用：
    
    `Y safe → Z safe → fold R_X → R_Z → restore R_X → X → Z → Y`
    
5. Strap high-risk interlock
    
    R_X ≥ 30° 且 R_Z 路徑接近 90°/270°時觸發；小 Y 的最終位置必須有明確 semantic whitelist。
    
6. Generic fixture rotation interlock
    
    `Z retract → fold R_X → R_Z+XY → restore R_X → target Z`
    
7. View-mode dog-leg
    
    `REHAUT/CROWN/SIDE/LEFT_HAND_CROWN` 一律先 retract Z；其他 view 依 Z 移動方向決定先後。
    

---

## 設定位置

主要設定集中在 [hardware_config.yaml (line 78)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:78)：

|設定|位置|目前值/作用|
|---|---|---|
|門禁 MQTT|78–82|localhost:1883、DI0|
|運動學 pivot/arm|104–108|Digital Twin 幾何來源|
|Digital Twin boxes|110–113|全部為 `0.0`|
|Generic safe retract|115|Z=20|
|Home poses|116–146|watch/strap/box|
|舊 Strap interlock|147–157|fallback/重複設定|
|Fixture profile|159–166|active=`watch`、mode=`4`|
|Watch motion safety|168–239|large-RX、special、wall、high-risk|
|Box motion safety|240–259|box 專用角度窗口|
|Zaber 軸限位|296–301|X/Y/Z/RX 有界，RZ 為連續軸|

Strap macro_cam_1 的額外設定在 [strap_macro_cam1_scan.yaml (line 267)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:267)：

- `calibration_confirmed: true`
- `require_hardware_safety_planner: true`
- `require_nonzero_collision_envelopes: false`
- Strap 全長 240 mm
- 最低牆面間隙 30 mm
- staging Y=160、Z=70
- 額外 stage limits
- 強制 view transition staging
- readback tolerance

---

## 相似或重複機制整理

### 1. 兩套幾何碰撞引擎

- 正式引用的是 [DigitalTwinGuard (line 6)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/kinematics/digital_twin_guard.py:6)。
- [SafeZoneManager (line 11)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/safe_zone_manager.py:11) 是另一套 AABB、另一套 config 格式及 pivot 定義。
- 全 repo 沒有找到 `SafeZoneManager` 的實際呼叫者，目前可視為未接線/遺留實作。

兩者不應被誤認為雙重保護；現況只有 Digital Twin 接在部分正式路徑上。

### 2. 三套相似的安全路徑

- Unified driver 的通用 `_plan_motion_segments`。
- Strap scan 的外層 `build_safe_transition_waypoints`。
- App 關閉/換頁的 `_move_zaber_to_strap_safe_retract`。

後兩者不是完全重複：它們增加 waypoint readback 或 shutdown staging，但固定數值和移動順序高度相似。

### 3. 安全常數重複

- Y=160：large-RX、special staging、high-risk、scan transit。
- Z=70：large-RX、special staging、high-risk、scan retract。
- Wall clearance=30：hardware wall envelope 與 scan safety。
- Strap half-length：
    - hardware legacy envelope：100 mm
    - motion recorder legacy：100 mm
    - macro_cam_1 scan：120 mm

macro_cam_1 會透過 semantic metadata 把 driver 的 100 mm 加嚴至 120 mm，因此不是單純衝突；但目前來源分散，容易日後漂移。

### 4. 軸限位重複

- Hardware 真正限位在 `hardware_config.yaml`。
- macro_cam_1 scan 又有一份 `safety.stage_limits`。

兩份目前數值相同，但沒有單一來源自動同步。

### 5. 兩種 emergency stop

- `trigger_emergency_stop()`：停止硬體並把 `_system_locked=True`，必須人工 reset。
- `emergency_stop()`：只停止 Zaber/light/camera，不會 latch `_system_locked`。

因此不同 UI/呼叫者使用不同 API 時，停止後的鎖定行為並不一致。

---

## 現有底層保護

ZaberManager 在 [hardware_managers.py (line 249)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_managers.py:249) 提供：

- 啟動時將 YAML limits 寫入 Zaber firmware。
- 每次 `move_axis` 再做軟體 clamp。
- `RLock` 避免多執行緒同時發送移動。
- compound move 固定命令順序為 Z、R_X、R_Z、X、Y。
- R_Z 空陣列表示連續旋轉；韌體限位被擴成極大值。
- 急停呼叫所有裝置的 `all_axes.stop()`。

但低階移動超界時是「clamp 後繼續」，不是 fail-closed 拒絕；上層若沒有 readback 驗證，可能誤以為已到達原始目標。

---

## 門禁與急停

實作在：

- [SafetyManager (line 4261)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_managers.py:4261)
- [trigger_emergency_stop (line 4566)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:4566)

行為：

- Production 才連 MQTT。
- `DI0 == 0/false` 觸發急停。
- JSON 解析錯誤也 fail-closed 觸發急停。
- 停止順序為 Zaber → lighting → camera。
- 鎖定後需要 `manual_safety_reset()`。

缺口：

- MQTT 初始連線失敗或後續斷線只記錄 disconnected，不會自動 latch 急停。
- `manual_safety_reset()` 不檢查門是否已關閉，也不檢查 SafetyManager/MQTT 是否恢復；它只恢復 Zaber/light/camera/IO。

---

## 明確繞過完整防撞的入口

|入口|實際保護|
|---|---|
|[WorkflowManager.manual_move (line 3056)](D:/Provenance Laboratories projects/ImagingLibWatch/core/workflow_manager.py:3056)|直接呼叫 Zaber；只有軸 clamp/firmware limit|
|[controller_server move_stage (line 98)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/controller_server.py:98)|直接 Zaber，沒有 planner、Digital Twin、SafetyManager|
|測試工具直接使用 ZaberManager|通常只有軸限位|
|多個 autofocus 內部微調|初始 pose 可能安全，但後續直接單軸移動，不逐步跑 Digital Twin|
|`execute_template_point`|有鎖及 planner，沒有一般 `move_compound` 的 Digital Twin/readback 層|

---

## 稽核與測試

Strap 執行時會在 `DB/strap_motion_logs` 記錄：

- 每個 Zaber command。
- 移動前、目標、移動後的位置。
- Strap 對 Y=0 牆的計算間隙。
- 最低間隙及是否出現負值。

程式在 [strap_motion_recorder.py (line 36)](D:/Provenance Laboratories projects/ImagingLibWatch/core/strap_motion_recorder.py:36)。這是事後稽核，不會即時阻止移動。

主要測試：

- [test_unified_motion_planning.py (line 36)](D:/Provenance Laboratories projects/ImagingLibWatch/tests/test_unified_motion_planning.py:36)：high-risk、special staging、readback fail-closed、連續 R_Z。
- [test_strap_macro1_scan.py (line 963)](D:/Provenance Laboratories projects/ImagingLibWatch/tests/test_strap_macro1_scan.py:963)：safe transition、軸限位、wall envelope、零碰撞盒 gate。
- [test_safety_system.py (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/test_hardware/test_safety_system.py:1)：MQTT 門禁模擬；不是實體門鎖/HIL 測試。

## 建議優先處理

1. 實測並填入非零 `safe_zone` 包絡，然後將 `require_nonzero_collision_envelopes` 改為 `true`。
2. 讓 `execute_template_point` 統一走 `move_compound`，或補上相同 Digital Twin/readback contract。
3. 禁止 production 的 `manual_move`/`controller_server.move_stage` 直接繞過 planner。
4. MQTT disconnected 應 fail-closed，manual reset 應確認門已關閉且 safety channel healthy。
5. 合併重複的 Y/Z、wall geometry、stage limits 成單一安全設定來源。
6. 明確區分「latched emergency stop」與「shutdown stop」，避免 UI 呼叫錯誤 API。

變更檔案：無。實體設備仍需要空治具、低速、最長 Strap、門禁斷線及實體 E-stop 的 HIL 驗證。




#### move_compound function analysis
```
所以在執行App main.py的每個有移動跟轉動的job都是使用move_compound function去控制? 還是有拍照task的有移動跟轉動的job都是使用move_compound function去控制? 那在App裡面的create template的拍照流程跟App裡面的watchentry的拍照流程是否也都使用move_compound function? 或像移動到home or center or 要放置watch的是否也使用move_compound function?

也請介紹以下列出的move_compound完整步驟是否正確  
flowchart TD  
A["App / Keyence / Exit cleanup"] --> B["move_compound"]  
B --> C["check_safety_lock"]  
C --> D["讀取目前 5 軸位置"]  
D --> E["可選運動學補償"]  
E --> F["DigitalTwinGuard 檢查最終目標"]  
F --> G["_plan_motion_segments"]  
G --> H["依優先序產生 staging segments"]  
H --> I["Strap 旋轉前 Y/Z 與 R_X readback 驗證"]  
I --> J["ZaberManager move_axis / move_axes_absolute"]  
J --> K["軟體 clamp + firmware soft limits"]  
並介紹每個步驟A,B,C....  
以及各跟那些functions and config有關
```

結論：**不是所有 App 移動，也不是所有拍照移動，都會呼叫 `move_compound()`。**

更精確地說，目前有三種運動路徑：

1. 直接呼叫 `move_compound()`。
2. 不呼叫 `move_compound()`，但直接使用同一個 `_plan_motion_segments()` planner。
3. 直接呼叫 `zaber.move_axis()`／`move_axes_absolute()`，完全繞過 planner 與 Digital Twin。

## 各 App 流程實際走法

| App 功能                         | 是否呼叫 `move_compound()` | 實際路徑                                                                                  |
| ------------------------------ | ---------------------- | ------------------------------------------------------------------------------------- |
| A. Create Template pre-capture | 否                      | `execute_template_point()` → `_plan_motion_segments()` → Zaber                        |
| B. Create Template 點選後正式拍照     | 通常否                    | `execute_template_point()` → planner → AF → final approach                            |
| C. WatchEntry 正式拍照             | 否                      | `_capture_images_routine_internal()` → `execute_template_point()` → planner           |
| D. Standard/HDR capture        | 否                      | 都進 `execute_template_point()`；HDR 同點可能略過重複 XYZ                                        |
| E. Control panel 手動五軸移動        | 是                      | `_move_control_pose()` → `move_compound()`                                            |
| F. Control panel Go Home       | 是                      | `_resolve_control_home_pose()` → `_move_control_pose()` → `move_compound()`           |
| G. 移到 holder／放錶位置              | 是，production driver 下  | `_move_zaber_to_start_pos()` → `_move_zaber_to_safe_strap_pose()` → `move_compound()` |
| H. 移到 watch center             | 是                      | `_move_zaber_to_center_pos()` → safe strap pose → `move_compound()`                   |
| I. App Exit 最終回 Home           | 部分                     | 先用直接 `move_axis()` 做 Strap safe retract，再用 `move_compound()` 回 Home                   |
| J. Keyence probe 移動            | 否                      | `_execute_keyence_planned_pose()` 直接做 Digital Twin + planner + Zaber                  |
| K. Final focus approach        | 否                      | `_do_final_approach()` 直接呼叫 Zaber                                                     |
| L. 某些 AF 掃描／Keyence XY 掃描      | 混合                     | 有些 Y scan 用 `move_compound()`；有些直接 `move_axis()`                                      |
| M. simulation `move_compound`  | 名稱相同但不是安全路徑            | 只有 sleep，沒有 planner 或碰撞檢查                                                             |

主要依據：

- Create Template pre-capture 呼叫 [`execute_template_point()` (line 1640)](/D:/Provenance Project/ImagingLibWatch/App/main.py:1640)
- Create Template prepared capture 呼叫 [`execute_template_point()` (line 8788)](/D:/Provenance Project/ImagingLibWatch/App/main.py:8788)
- WatchEntry capture routine 呼叫 [`execute_template_point()` (line 32043)](/D:/Provenance Project/ImagingLibWatch/App/main.py:32043)
- `execute_template_point()` 直接呼叫 planner，而非 `move_compound()`：[unified_driver.py (line 9420)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:9420)
- Control panel 移動呼叫 `move_compound()`：[main.py (line 26483)](/D:/Provenance Project/ImagingLibWatch/App/main.py:26483)
- Holder/Center 共用 safe pose：[main.py (line 9325)](/D:/Provenance Project/ImagingLibWatch/App/main.py:9325)
- Exit cleanup：[main.py (line 3545)](/D:/Provenance Project/ImagingLibWatch/App/main.py:3545)、[main.py (line 3657)](/D:/Provenance Project/ImagingLibWatch/App/main.py:3657)

所以「拍照 task 有移動與轉動」並不能直接推導成「一定呼叫 `move_compound()`」。正式拍照通常只是**共用它的 planner**。

---

## 你的 A–K 流程是否正確

整體方向接近，但 A、I、K 需要修正，而且漏掉 `move_compound()` 的參數正規化及錯誤處理。

````
```mermaid
flowchart TD
    A["呼叫來源：App control / Home / Holder / Exit final leg / 部分 AF"] --> B["move_compound"]
    B --> C["清除 last error + check_safety_lock"]
    C --> D["解析 dict/kwargs，讀取目前五軸位置"]
    D --> E{"auto_compensate=True？"}
    E -->|是| F["CoordinateMapper 運動學補償 XYZ"]
    E -->|否| G["保持要求的 XYZ"]
    F --> G
    G --> H["DigitalTwinGuard 檢查最終目標"]
    H --> I["_plan_motion_segments"]
    I --> J["依優先序產生安全 segments"]
    J --> K{"Strap 且 segment 有標記的實際旋轉？"}
    K -->|是| L["驗證 Y/Z staging；R_Z 前驗證 R_X fold"]
    K -->|否| M["執行 segment"]
    L --> M
    M --> N["Zaber move_axis / move_axes_absolute"]
    N --> O["每個 move_axis 先做軟體 clamp"]
    O --> P["Zaber firmware soft limits 最後保護"]
    P --> Q["成功 True；例外則記錄 last_move_error 並回傳 False"]
```
````

### A：App / Keyence / Exit cleanup

你的：

> App / Keyence / Exit cleanup → move_compound

只對一部分來源成立。

- App control、Go Home、holder、center：會進 `move_compound()`。
- Exit：safe retract 是直接 `move_axis()`；safe retract 成功後，最後回 Home 才進 `move_compound()`。
- Keyence：主要走 `_execute_keyence_planned_pose()`，它直接呼叫 Digital Twin、planner 和 Zaber，不會經過 `move_compound()`。
- Create Template／WatchEntry 拍照：主要走 `execute_template_point()`，同樣直接使用 planner。

### B：`move_compound`

定義位於 [unified_driver.py (line 475)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:475)。

接受：

- `target_x/y/z`
- `target_rx/rz`
- 或傳入 `{"stage_L_X": ..., ...}` dict
- `auto_compensate`
- `cam_alias`
- `wait`
- `semantic_target`

未指定的軸會使用目前位置，而不是歸零。

### C：`check_safety_lock`

位於 [unified_driver.py (line 4560)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:4560)。

它只檢查 `_system_locked`。Door-open／SafetyManager 觸發 `trigger_emergency_stop()` 後會：

- 停止 Zaber
- 關閉燈光
- 停止相機
- 將系統保持 locked

必須透過 `manual_safety_reset()` 完成硬體 recovery 後才解除。

這不是幾何碰撞檢查；它是「系統是否已被安全事件鎖住」。

### D：讀取目前五軸位置

讀取：

- `stage_L_X`
- `stage_L_Y`
- `stage_L_Z`
- `stage_R_X`
- `stage_R_Z`

其中 R_Z 是 continuous rotation axis，所以會先用 `canonical_angle_deg()` 正規化成等價的 0–360° 角度，規劃與比較才不會把例如 23760° 認為需要轉很多圈。

Zaber 真正送命令時又會用 `_nearest_equivalent_angle()` 選擇離目前位置最近的等價角度。

### E：可選運動學補償

方向正確，但它只在：

```
auto_compensate=True
```

且 R_X/R_Z 發生改變時執行。

相關 function：

- `CoordinateMapper.calculate_full_eccentric_compensation()`
- `move_compound()` 內將 `dX/dY/dZ` 加到 requested XYZ

相關 config：

- `hardware.zaber.kinematics`
- `pivot_z_L1`
- `camera_arm_L2_macro`
- `camera_arm_L2_micro`
- rotation pivot 等參數

重要現況：App 中找到的 production `move_compound()` callers 幾乎都明確傳入 `auto_compensate=False`，目前沒有非測試 caller 明確使用 `True`。

另外，`execute_template_point()` 有另一套不同的 `_apply_angle_pose_compensation()`，受以下設定控制：

- `hardware.autofocus.keyence_autofocus.angle_aware`
- `pose_compensation`
- `angle_pose_compensation`

這套不能和 `move_compound(auto_compensate=True)` 當成同一件事。

### F：DigitalTwinGuard 檢查最終目標

正確，但只檢查**最終 target pose**：

```
predict_collision(x, y, z, rx, rz, cam_alias)
```

相關程式：

- [digital_twin_guard.py (line 6)](/D:/Provenance Project/ImagingLibWatch/Controller/kinematics/digital_twin_guard.py:6)
- `DigitalTwinGuard.predict_collision()`
- `_get_aabb()`
- `_check_overlap()`

相關 config：

- `hardware.zaber.kinematics`
- `hardware.zaber.safe_zone.platform_box`
- `macro_box`
- `micro_box`

目前 config 中這些 box 尺寸全部是 `0.0`：

```
platform_box: {width: 0.0, depth: 0.0, height: 0.0}
macro_box: {width: 0.0, length: 0.0}
micro_box: {width: 0.0, length: 0.0}
```

見 [hardware_config.yaml (line 110)](/D:/Provenance Project/ImagingLibWatch/config/hardware_config.yaml:110)。

因此目前 Digital Twin 無法代表真實平台、錶與鏡頭體積，實際幾何保護能力非常有限。現有安全主要依靠 `_plan_motion_segments()` 的規則式 interlocks。

另一個限制是：`move_compound()` 並沒有對 planner 產生的每一個中間 segment 再逐段做 Digital Twin 檢查。

### G、H：`_plan_motion_segments` 與 staging priority

定義位於 [unified_driver.py (line 606)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:606)。

規劃大致依下列優先序：

1. Strap 特殊 internalnum whitelist：
    
    - 4029–4032
    - 驗證 internalnum 和完整五軸 target 必須吻合
    - 經 Y=160、Z=70 staging
2. 離開 Strap 特殊點：
    
    - 先退出到 safe Y/Z
    - 再遞迴規劃後續目的地
3. Strap Keyence transition：
    
    - 強制經特殊 staging
4. Strap wall envelope：
    
    - 根據 Y、R_Z、strap half-length 計算靠牆端點 clearance
    - 不足時直接 reject，不產生 motion
5. Large R_X transition：
    
    - Y safe
    - Z safe
    - 必要時 fold R_X
    - R_Z
    - restore R_X
    - final X
    - final Z
    - final Y
6. Strap high-risk interlock：
    
    - 特別處理 R_X 高角度且 R_Z 接近 90°／270° 的情況
    - 可能要求 `allow_high_risk_final_approach`
7. 一般 fixture rotation interlock：
    
    - retract Z
    - fold R_X
    - rotate R_Z + XY
    - restore R_X
    - final Z
8. `REHAUT_VIEW`／`CROWN_VIEW`／`SIDE_VIEW`：
    
    - Z retract
    - 高度安全時移動 XY/RX/RZ
    - Z descend
9. Front／Back 一般路徑：
    
    - 依 target Z 相對 current Z 決定先 Z 還是先 XY/rotation

相關 config 集中在 [hardware_config.yaml (line 147)](/D:/Provenance Project/ImagingLibWatch/config/hardware_config.yaml:147)：

- `fixture.active_profile`
- `motion_interlock_mode`
- `safe_retract_z`
- `rotation_interlock`
- `large_rx_transition`
- `strap_special_staging`
- `strap_wall_envelope`
- `strap_high_risk_interlock`
- `strap_x_only_keep_rx_at_rz_0_180`

### I：Strap 旋轉前 readback 驗證

你的描述需要加上「條件式」。

在 `move_compound()` 中，只有同時符合以下條件才驗證：

- segment 是 `COMPOUND`
- 包含 R_X 或 R_Z
- semantic target 被辨識為 Strap
- segment 有 `strap_rotation_staging_required=True`
- readback 顯示真的有 rotation delta

相關 functions：

- `_is_strap_semantic_target()`
- `_strap_segment_has_rotation_delta()`
- `_verify_strap_rotation_staging()`：驗證 Y/Z
- `_verify_strap_rz_fold_readback()`：R_Z 前驗證 R_X 已 fold

它不是所有 rotation segment 都無條件執行。

另外，`execute_template_point()` 在 [unified_driver.py (line 9434)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:9434) 自己執行 planner segments 時，沒有走 `move_compound()` 裡這段相同的 Strap rotation readback wrapper。Keyence 的 `_execute_keyence_planned_pose()` 則有自己的 readback 驗證。

### J：ZaberManager 執行

Planner 只產生兩種主要 segment：

- `Z_ONLY` → `zaber.move_axis("stage_L_Z", ...)`
- `COMPOUND` → `zaber.move_axes_absolute({...})`

`move_axes_absolute()` 的預設命令順序為：

1. Z
2. R_X
3. R_Z
4. X
5. Y

但實際 segment 通常只包含其中幾軸；planner 已經事先將安全順序拆成多個 segment。

相關程式：[hardware_managers.py (line 458)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_managers.py:458)、[hardware_managers.py (line 500)](/D:/Provenance Project/ImagingLibWatch/Controller/hardware_managers.py:500)。

### K：軟體 clamp + firmware soft limits

概念正確，但它不是在 Zaber 命令完成後才做，而是在 J 裡面分成兩層：

1. 軟體 clamp  
    `ZaberManager.move_axis()` 送命令前根據 `self.limits` clamp。
    
2. Firmware limits  
    Zaber 連線時 `_apply_hardware_limits()` 將同一組 limits 寫入：
    
    - `limit.min`
    - `limit.max`

相關 config 位於 [hardware_config.yaml (line 296)](/D:/Provenance Project/ImagingLibWatch/config/hardware_config.yaml:296)：

- X：0–435 mm
- Y：0–292 mm
- Z：0–150 mm
- R_X：0–90°
- R_Z：空 list，代表 continuous rotation

這一層只限制各軸 travel range，無法判斷多軸組合是否會碰撞。

## 最重要的架構判讀

目前不是：

```
所有移動 → move_compound → 安全
```

而是：

```
App Home/Holder/Control
    → move_compound
    → Digital Twin final target
    → planner
    → Zaber

Create Template / WatchEntry capture
    → execute_template_point
    → angle-aware compensation
    → planner
    → Keyence/AF
    → direct final approach
    → capture

Keyence planned probe
    → Keyence-specific preflight
    → Digital Twin
    → planner
    → Strap readback verification
    → Zaber

其他 AF / cleanup / compatibility paths
    → 可能直接 Zaber
```

因此，`move_compound()` 是重要的安全入口，但目前還不是全 App 唯一的五軸運動入口。最值得注意的殘餘風險是：

- Digital Twin box 尺寸目前為零。
- `execute_template_point()` 複用 planner，但不複用完整 `move_compound()` wrapper。
- `_do_final_approach()` 直接走 Zaber。
- 仍存在直接 `move_axis()` 的 AF、safe retract、compatibility 路徑。
- `move_compound()` 捕捉例外後回傳 `False`，caller 必須檢查；而且它目前沒有檢查每個 Zaber segment 的 boolean 回傳值。
-




```
如果依App main.py的各 App 流程實際走法依照下列的錶列出的13個流程:


## 各 App 流程實際走法

| App 功能                         | 是否呼叫 `move_compound()` | 實際路徑                                                                                  |
| ------------------------------ | ---------------------- | ------------------------------------------------------------------------------------- |
| A. Create Template pre-capture | 否                      | `execute_template_point()` → `_plan_motion_segments()` → Zaber                        |
| B. Create Template 點選後正式拍照     | 通常否                    | `execute_template_point()` → planner → AF → final approach                            |
| C. WatchEntry 正式拍照             | 否                      | `_capture_images_routine_internal()` → `execute_template_point()` → planner           |
| D. Standard/HDR capture        | 否                      | 都進 `execute_template_point()`；HDR 同點可能略過重複 XYZ                                        |
| E. Control panel 手動五軸移動        | 是                      | `_move_control_pose()` → `move_compound()`                                            |
| F. Control panel Go Home       | 是                      | `_resolve_control_home_pose()` → `_move_control_pose()` → `move_compound()`           |
| G. 移到 holder／放錶位置              | 是，production driver 下  | `_move_zaber_to_start_pos()` → `_move_zaber_to_safe_strap_pose()` → `move_compound()` |
| H. 移到 watch center             | 是                      | `_move_zaber_to_center_pos()` → safe strap pose → `move_compound()`                   |
| I. App Exit 最終回 Home           | 部分                     | 先用直接 `move_axis()` 做 Strap safe retract，再用 `move_compound()` 回 Home                   |
| J. Keyence probe 移動            | 否                      | `_execute_keyence_planned_pose()` 直接做 Digital Twin + planner + Zaber                  |
| K. Final focus approach        | 否                      | `_do_final_approach()` 直接呼叫 Zaber                                                     |
| L. 某些 AF 掃描／Keyence XY 掃描      | 混合                     | 有些 Y scan 用 `move_compound()`；有些直接 `move_axis()`                                      |
| M. simulation `move_compound`  | 名稱相同但不是安全路徑            | 只有 sleep，沒有 planner 或碰撞檢查                                                             |


我希望以此為基準完善各 App 流程的並模組化整理(也先不要修改code而是釐清每條流程), 包括個別流程的主要流程function(舉例來說在第一個Create Template pre-capture你有寫到實際路徑`execute_template_point()` → `_plan_motion_segments()` → Zaber那這是否是包含在哪個function裡面, 我會稱這個function為主要流程function or 入口function, 也就是在這function裡包含`execute_template_point()` → `_plan_motion_segments()` → Zaber, 且這function最可以代表Create Template pre-capture). 

然後就像Create Template pre-capture, Create Template 點選後正式拍照, WatchEntry 正式拍照, Standard/HDR capture都有execute_template_point, 所以這些execute_template_point不同流程使用到execute_template_point下面的每一步小流程是否都一樣? 那這些小流程各是甚麼? 

也請將一些類似execute_template_point這樣常用且有很多步小流程的也要整理這些小流程
```

已依目前程式做完靜態追蹤，沒有修改任何 code。

先講核心結論：

1. `execute_template_point()` 是「單一拍攝點的共用拍攝引擎」，但不同呼叫者不會走完全相同的小流程；Standard、HDR、AF 模式、焦點重用與位置快取都會造成分支或跳過。
2. `move_compound()` 是「一般五軸安全移動入口」，它與 `execute_template_point()` 都使用 `_plan_motion_segments()`，但兩者不是同一條安全路徑。
3. `execute_template_point()` 的初始定位雖使用 planner，卻不經過 `move_compound()`，因此也沒有 `move_compound()` 裡的 Digital Twin final-target 檢查；後段 `_do_final_approach()` 更是直接操作 Zaber。
4. 目前 [system_config.yaml (line 2)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.yaml:2) 是 `simulation`。以下 hardware 路徑是 production/hybrid 時的實際走法。
5. `_execute_point_capture()` 目前只有定義、沒有靜態呼叫者，應視為舊路徑，不能代表現行 Create Template 正式拍照。

## 一、建議統一成三層理解

````
```mermaid
flowchart TD
    UI["App / UI 流程入口"]

    A["A TemplatePreCaptureWorker.run"]
    B["B _execute_prepared_point_capture"]
    C["C _capture_images_routine_internal"]
    EH["E–H 控制面板／Home／Holder／Center"]
    I["I Exit Home"]
    J["J Keyence pose"]
    L["L AF / HDR scan"]

    ETP["execute_template_point<br/>單點拍攝引擎"]
    MC["move_compound<br/>一般五軸移動入口"]
    KP["_execute_keyence_planned_pose"]
    PLAN["_plan_motion_segments<br/>共用規則式路徑規劃"]
    AF["Keyence / Vision / Liquid AF"]
    FINAL["_do_final_approach<br/>直接 Zaber"]
    HDR["_capture_and_fuse_hdr_v2"]
    TWIN["DigitalTwinGuard<br/>只預測 final target"]
    ZABER["Zaber move_axis / move_axes_absolute"]

    UI --> A --> ETP
    UI --> B --> ETP
    UI --> C --> ETP
    UI --> EH --> MC
    UI --> I
    UI --> J --> KP
    UI --> L

    MC --> TWIN --> PLAN --> ZABER
    KP --> TWIN
    KP --> PLAN

    ETP --> PLAN
    ETP --> AF
    ETP --> FINAL --> ZABER
    ETP --> HDR --> ZABER

    I --> ZABER
    I --> MC
    L --> MC
    L --> ZABER
```
````

這張圖最重要的地方是：`_plan_motion_segments()` 是共用 planner，但「有使用 planner」不等於「完整走過同一個安全入口」。

---

# 二、A–M 每條 App 流程的主要入口

|流程|最具代表性的主要流程函式|production/hybrid 實際主路徑|判定|
|---|---|---|---|
|A. Create Template pre-capture|`TemplatePreCaptureWorker.run()`|`_start_template_pre_capture_sequence()` → Worker `run()` → 建立每張 `hw_cfg` → `execute_template_point()` → 儲存/旋轉/crop|原表方向正確，但真正代表整個流程的是 Worker `run()`|
|B. Create Template 正式點拍|`_execute_prepared_point_capture()`|`_scratch_next_capture()` → `_start_prepared_point_capture_async()` → `_execute_prepared_point_capture()` → `execute_template_point()`|現行路徑；舊 `_execute_point_capture()` 沒有呼叫者|
|C. WatchEntry 正式拍照|`_capture_images_routine_internal()`|`load_camimgs()` → `CaptureRoutineWorker.run()` → `_capture_images_routine_internal()` → capture loop → `execute_template_point()`|正確|
|D. Standard/HDR|不是獨立入口，是 B/C 內的 capture 分支|Standard/HDR 都建立各自 `hw_cfg` → `execute_template_point()`|共用入口，但內部步驟不完全相同|
|E. Control panel 五軸移動|`_move_control_pose()`|`_ctrl_move()` → `_move_control_pose()` → `move_compound()`|正確|
|F. Control panel Go Home|`_ctrl_action(...,"go_home")`|`_resolve_control_home_pose()` → `_move_control_pose()` → `move_compound()`|正確|
|G. Holder／放錶位置|`_move_zaber_to_start_pos()`|Macro 1：`_strap_macro1_transition_to_pose()`；其他 camera：`_move_zaber_to_safe_strap_pose()` → `move_compound()`|原表需補 Macro 1 特殊 transition|
|H. Watch center|`_move_zaber_to_center_pos()`|與 G 相同，依 camera 分成 Macro 1 特殊路徑或一般 safe strap pose|原表需補 Macro 1 分支|
|I. App Exit Home|`_move_zaber_to_home_on_exit()`|`closeEvent()` → `_move_zaber_to_strap_safe_retract()` 直接分軸 → `move_compound(home)`|正確|
|J. Keyence probe|App-facing：`position_height_measurement_probe()`；共用核心：`_execute_keyence_planned_pose()`|target validation → safety lock → Strap staging → Digital Twin → planner → Zaber → readback|正確，但 helper 並非唯一 UI 入口|
|K. Final focus approach|`_do_final_approach()`|near point → 切 capture profile → 直接 Zaber X/Y/Z|正確；不經 planner/Twin|
|L. AF/Keyence XY/Y scan|依 AF 類型不同|Side Y scan 可用 `move_compound()`；Keyence/vision/HDR Z-stack 有直接 `move_axis()`|正確，是混合安全邊界|
|M. Simulation move|`SimulationHardware.move_compound()`|log → sleep 0.5 秒|正確；沒有 planner、碰撞檢查、pose state，且介面也與 production 不完全一致|

## A. Create Template pre-capture

UI 啟動入口：

- [`_start_template_pre_capture_sequence()` (line 16836)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:16836)

真正包含完整五張圖循環的主要流程函式：

- [`TemplatePreCaptureWorker` (line 1508)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:1508)
- [`TemplatePreCaptureWorker.run()` (line 1540)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:1540)

路徑：

```
_start_template_pre_capture_sequence()
└─ TemplatePreCaptureWorker.run()
   ├─ 依 front / side1 / side2 / side3 / side4 建立 hw_cfg
   ├─ simulation：讀取模擬圖片
   └─ production/hybrid：
      └─ execute_template_point(hw_cfg)
         ├─ planner / AF / final approach
         └─ camera capture
   ├─ rotation
   ├─ center crop
   └─ 儲存 JPG / progress signal
```

完成後再進：

- [`_on_template_pre_capture_finished()` (line 17249)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:17249)：更新圖片、執行 color detection、復歸角度、切換頁面。

所以 A 的「主要流程函式」應定義為 `TemplatePreCaptureWorker.run()`；`_start_template_pre_capture_sequence()` 是 UI/執行緒入口。

## B. Create Template 點選後正式拍照

現行主要流程：

- [`_prepare_selected_point_from_enlarge()` (line 8172)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8172)：把點選座標轉成 pose，這時還沒有正式 AF/capture。
- [`_scratch_next_capture()` (line 15858)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:15858)：決定下一張 Standard/HDR。
- [`_start_prepared_point_capture_async()` (line 8886)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8886)：非同步入口。
- [`_execute_prepared_point_capture()` (line 8487)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8487)：最能代表正式拍照的主要流程函式。

```
_prepare_selected_point_from_enlarge()
└─ 準備/鎖定 pose

_scratch_next_capture()
└─ _start_prepared_point_capture_async()
   └─ _execute_prepared_point_capture()
      ├─ 解析 point/capture/internalnum
      ├─ 建立 canonical hw_cfg
      ├─ 建立 focus_hardware
      ├─ 套用 bezel policy
      ├─ HDR 可重用上一張 Standard focus
      ├─ execute_template_point()
      └─ 保存 scratch image / focus state
```

舊的 [`_execute_point_capture()` (line 8362)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:8362)：

- Micro 走 interactive micro capture。
- Macro 直接 `move_compound()` 後拍照。
- 目前 `rg` 只找到定義，沒有實際呼叫者，應標為 legacy/dead candidate。

## C. WatchEntry 正式拍照

App/UI 入口：

- [`load_camimgs()` (line 31355)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:31355)

Worker：

- [`CaptureRoutineWorker` (line 1478)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:1478)
- `CaptureRoutineWorker.run()` 在同一區塊呼叫 internal routine。

主要流程函式：

- [`_capture_images_routine_internal()` (line 31780)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:31780)

```
load_camimgs(side)
└─ CaptureRoutineWorker.run()
   └─ _capture_images_routine_internal()
      ├─ 建立 DB run / output dirs
      ├─ 依 watchpoint execution order 走訪
      ├─ normalization：point + capture + internalnum
      ├─ 分出 standard_captures / hdr_captures
      ├─ 建立 focus_hardware
      ├─ 套用 WatchShift / bezel policy
      ├─ HDR 判斷是否重用 Standard focus
      ├─ execute_template_point(hw_cfg)
      ├─ 儲存 local image / DB
      ├─ S3 / metadata
      └─ analysis / UI progress
```

所以 C 的代表函式確實是 `_capture_images_routine_internal()`；`load_camimgs()` 是 UI 啟動層。

---

# 三、`execute_template_point()` 的完整小流程

核心位置：

- [`UnifiedHardwareDriver.execute_template_point()` (line 8591)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:8591)

## Phase 0：輸入與狀態

- fixture profile override。
- `check_safety_lock()`。
- 建立 side-reference scope。
- 判斷：
    - `is_hdr`
    - `z_positions`
    - `reuse_previous_focus`
    - `skip_xyz_move`
    - 是否使用獨立 `focus_hardware`
    - 是否有 Standard focus 可以重用。

## Phase 1：解析拍照與 AF 路由

主要工作：

1. 解析 `X/Y/Z/RX/RZ`、`Xmod/Ymod/Zmod`。
2. 解析 camera、exposure、gain、lighting、postcolor。
3. HDR AF 使用中間 exposure 作 preview。
4. 依 `af_mode_id` 載入 AF recipe。
5. 決定 capture camera 與 focus camera。
6. 呼叫 `resolve_autofocus_routing()` 決定：
    - Keyence coarse AF
    - vision AF
    - liquid-lens AF
    - side-Zaber AF
    - 是否略過 first camera pose。
7. 計算 Keyence angle-aware pose compensation。
8. 建立 position signature，判斷是否 `is_same_pos`。

## Phase 2：初始定位與機械 AF

只有 `not is_same_pos` 時執行：

```
讀取 current pose
→ 建立 target pose + semantic target
→ _plan_motion_segments()
→ 直接 zaber.move_axis / move_axes_absolute
→ 選擇 coarse AF
   ├─ option 4：_af_keyence_absolute()
   ├─ option 1：_af_keyence_matlab_like()
   ├─ option 2/3：_af_vision_method()
   └─ disabled：使用 base Z
→ 計算 final X/Y/Z
→ _do_final_approach()
```

重點：

- 這裡有 planner。
- 但沒有呼叫 `move_compound()`。
- 也沒有 `move_compound()` 裡的通用 Digital Twin preflight。
- `_do_final_approach()` 又是另一段直接 Zaber 路徑。

## Phase 3：拍照硬體設定

- 開 capture lights。
- 切換 camera。
- 套用 AWB/AGC/Gamma/ColorTransform lock。
- 設定並驗證 exposure/gain。

## Phase 4：Liquid AF 與最後修正

依設定走其中一條：

- `reuse_previous_focus`：保持目前 liquid current，不重新 AF。
- HDR skip AF：使用 manual cammag/current。
- Side-Zaber two-stage：
    - `_run_side_zaber_liquid_lens_autofocus()`
- Side reference reuse。
- FPGA/hardware liquid AF。
- software sweep fallback。
- AF failure fallback/manual current。
- 若 `focus_required` 且 AF 失敗，capture 前中止。
- 套用最後 `Xmod/Ymod/Zmod`。
- 若 mod 非零，第二次呼叫 `_do_final_approach()`。
- final stage readback。
- 恢復 capture lighting/camera/exposure。
- capture settle。

## Phase 5：真正拍照

- Standard：`cam.capture_image()`。
- HDR：[`_capture_and_fuse_hdr_v2()` (line 5201)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:5201)。
- finally 關燈。
- color calibration / post-color。
- rotation。
- 更新位置 signature cache。
- 回傳：

```
(image, final_z, glass_thickness)
```

---

# 四、Standard/HDR 是否走完全相同的小流程？

答案是：共用同一支函式，但不保證執行同一組步驟。

|情況|XYZ 初始移動|Coarse AF|Liquid AF|最終 capture|
|---|---|---|---|---|
|一般 Standard|通常執行|依 `use_autofocus`/AF mode|依 `use_liqud_af`|單張|
|同點 HDR，重用 Standard focus|通常跳過|跳過|保留 Standard liquid current|HDR brackets/fusion|
|HDR 有不同 `z_positions`|可能先到 base pose|視 Z bucket 政策重跑|視設定|多 exposure／多 Z|
|明確 `skip_xyz_move=True`|跳過|通常跳過|可能保留或手動設定|單張或 HDR|
|position signature 相同|跳過|跳過機械 AF|視 reuse/設定|繼續拍照|
|AF mode 7/8 等 side mode|執行 planner 定位|常停用 Keyence|Side-Zaber/FPGA two-stage|單張或 HDR|

因此真正決定差異的不是「誰呼叫它」，而是呼叫者組出的 `hw_cfg`：

- A：pre-capture 固定設定。
- B：prepared point + capture + focus_capture + bezel policy。
- C：template/watchpoint/capture + WatchEntry 修正 + Standard/HDR focus reference。
- D：`is_hdr`、`hdr_enable`、`z_positions`、`reuse_previous_focus`、`skip_xyz_move`。

---

# 五、其他常用複合函式

## 1. `move_compound()`

位置：

- [`move_compound()` (line 475)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:475)

內部步驟：

```
check_safety_lock()
→ 讀取五軸 current pose
→ target 正規化
→ optional kinematic auto compensation
→ DigitalTwinGuard.predict_collision(final target)
→ _plan_motion_segments()
→ 逐 segment 執行 Z_ONLY / COMPOUND
→ Strap rotation staging readback
→ 成功 True；例外轉成 False + _last_move_error
```

這是目前最完整的一般五軸移動入口，但 Digital Twin 只檢查 final target，沒有逐一檢查所有中間 segment 的幾何碰撞。

## 2. `_plan_motion_segments()`

位置：

- [`_plan_motion_segments()` (line 606)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:606)

它是規則式 planner，主要負責：

- fixture profile 與 `motion_interlock_mode`。
- safe retract Z。
- Strap/Box semantic 判斷。
- large R_X transition staging。
- Strap fixed-point whitelist。
- Strap wall envelope。
- high-risk R_X/R_Z staging。
- R_Z hazard crossing。
- generic dog-leg。
- 最後 target segment。

它不是 Digital Twin，也不直接操作硬體；輸出的是 `Z_ONLY`／`COMPOUND` segment 清單。

## 3. `_execute_keyence_planned_pose()`

位置：

- [`_execute_keyence_planned_pose()` (line 3296)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:3296)

步驟：

```
numeric/finite/axis-limit validation
→ safety lock
→ Strap Keyence staging
→ 讀 current pose
→ Digital Twin final-target check
→ _plan_motion_segments()
→ segment execution
→ Strap rotation/RX fold readback
→ final pose readback
→ report
```

這條比 `execute_template_point()` Phase 2 多了明確 axis-limit 與 Digital Twin preflight。

App-facing 六點量測入口是：

- [`position_height_measurement_probe()` (line 3483)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:3483)

一般 AF direct-probe 會經：

- [`_start_keyence_probe_motion()` (line 3435)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:3435)

## 4. `_do_final_approach()`

位置：

- [`_do_final_approach()` (line 4426)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:4426)

步驟：

```
讀 final_approach config
→ 套用 default motion profile
→ 移到 near target
→ 切 capture profile
→ 最後 X/Y/Z
→ 恢復 default profile
```

特殊 Strap case 會採 Y-last/reassert X/Z，但本質仍是直接 Zaber，不經：

- `move_compound()`
- `_plan_motion_segments()`
- Digital Twin

## 5. `_capture_and_fuse_hdr_v2()`

位置：

- [`_capture_and_fuse_hdr_v2()` (line 5201)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:5201)

包含：

- HDR defaults。
- 自動 exposure plan。
- exposure/Z bucket。
- 每 bucket lighting/exposure。
- 必要時遞迴呼叫 `execute_template_point()` 做 Z-bucket AF。
- 其他 Z-stack 移動則直接 `move_axis(stage_L_Z)`。
- bracket 儲存、fusion、auto processing。
- 最後 restore Z。

這也是 L 類「混合安全路徑」的一個來源。

## 6. AF 複合函式

- [`perform_two_stage_autofocus()` (line 1516)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:1516)  
    Control/API 型入口：coarse AF → `move_compound()` 到 focus Z → liquid AF。
    
- [`run_robust_autofocus()` (line 1622)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:1622)  
    依 AF option 路由到 Keyence absolute、Matlab-like 或 vision AF。
    
- [`_run_side_zaber_liquid_lens_autofocus()` (line 5932)](D:/Provenance Laboratories projects/ImagingLibWatch/Controller/hardware_drivers/unified_driver.py:5932)  
    Side Y candidates + FPGA liquid current 掃描；Y 移動使用 `move_compound()`。
    

要注意：`execute_template_point()` 不只是簡單呼叫 `perform_two_stage_autofocus()`；它自己編排 coarse AF、liquid AF、focus/capture lighting、final approach 與失敗政策，因此兩者存在部分概念重疊。

---

# 六、設定檔位置與控制範圍

|設定檔|主要用途|
|---|---|
|[system_config.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/system_config.yaml:1)|simulation/production/hybrid 模式|
|[hardware_config.yaml (line 84)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:84)|Zaber、camera、lighting、Keyence、AF 的主要硬體設定|
|[hardware_config.yaml (line 110)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:110)|Digital Twin `safe_zone` 幾何|
|[hardware_config.yaml (line 116)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:116)|Home poses|
|[hardware_config.yaml (line 159)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:159)|fixture profile、`motion_interlock_mode`|
|[hardware_config.yaml (line 186)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:186)|large R_X staging|
|[hardware_config.yaml (line 199)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:199)|Strap special staging/validated targets|
|[hardware_config.yaml (line 214)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:214)|Strap wall envelope|
|[hardware_config.yaml (line 222)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:222)|Strap high-risk interlock|
|[hardware_config.yaml (line 261)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:261)|Zaber motion profiles|
|[hardware_config.yaml (line 283)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:283)|final approach|
|[hardware_config.yaml (line 335)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:335)|camera-to-Keyence offset|
|[hardware_config.yaml (line 789)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hardware_config.yaml:789)|AF 基本設定|
|[af_mode_config.yaml (line 8)](D:/Provenance Laboratories projects/ImagingLibWatch/config/af_mode_config.yaml:8)|各 `af_mode_id` 對 hardware config 的 override recipe|
|[hdr_config.yaml (line 234)](D:/Provenance Laboratories projects/ImagingLibWatch/config/hdr_config.yaml:234)|HDR 自動 exposure/fusion 處理|
|[template_create_config.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/template_create_config.yaml:1)|Template pre-capture、bezel tilt 等 Create Template 行為|
|[internalnum_config.yaml](D:/Provenance Laboratories projects/ImagingLibWatch/config/internalnum_config.yaml)|internalnum 對 point/capture 預設 pose、camera、AF、HDR 參數|
|[strap_macro_cam1_scan.yaml (line 1)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:1)|Strap Macro 1 scan、endpoint、tile、anchor AF、安全限制|

目前需特別注意：

- `hardware.zaber.safe_zone` 的 platform/macro/micro collision box 尺寸目前是 `0.0`，因此 Digital Twin 幾何保護實質上沒有正常物理 envelope。
- Strap scan 設定在 [strap_macro_cam1_scan.yaml (line 273)](D:/Provenance Laboratories projects/ImagingLibWatch/config/strap_macro_cam1_scan.yaml:273) 設為 `require_nonzero_collision_envelopes: false`。
- 所以目前真正主要的防撞力量來自 `_plan_motion_segments()` 的 staging/interlock/wall-envelope 規則，而不是 Digital Twin AABB。

---

# 七、13 條表之外，App 內還有的路徑

若目標是「App/main.py 完整盤點」，原本 A–M 還漏了：

1. Bezel 自動傾角量測  
    [`_measure_bezel_surface_tilt()` (line 7573)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:7573)  
    準備/恢復 pose 使用 `move_compound()`，Keyence sample 有自己的量測動作。
    
2. Strap Macro 1 dynamic scan  
    [`_strap_macro1_capture_frame()` (line 35286)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:35286)  
    Capture 使用 `execute_template_point()`。
    
3. Strap Macro 1 transition  
    [`_strap_macro1_transition_to_pose()` (line 34593)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:34593)  
    外層有額外 waypoint/readback，再進安全 compound move。
    
4. Strap extension probe  
    [`_move_strap_keyence_probe_pose()` (line 37276)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:37276)  
    優先使用 `move_compound()`，但仍保留沒有 driver 時的 raw-axis fallback。
    
5. Macro top-view preview  
    [`_capture_and_show_macro_top_view()` (line 39273)](D:/Provenance Laboratories projects/ImagingLibWatch/App/main.py:39273)  
    production 使用 `execute_template_point()`。
    

---

# 八、依相似性整理後的模組分組

目前可以先在文件/設計上歸為五組，不必立刻修改 code：

1. App Orchestrators  
    A/B/C、Strap scan、top preview：負責建立 capture payload、循環、儲存、UI/DB。
    
2. Capture Point Engine  
    `execute_template_point()`：單一點的 motion + AF + light + camera + HDR。
    
3. Safe Pose Motion  
    `move_compound()`、`_execute_keyence_planned_pose()`、`_plan_motion_segments()`。
    
4. Focus Pipeline  
    Keyence、vision、liquid-lens、side-Zaber、focus reuse。
    
5. Raw/Terminal Motion  
    `_do_final_approach()`、HDR Z-stack、Exit safe retract、legacy fallback。
    

最值得後續統一的不是 A/B/C 本身，而是第 3 與第 5 組之間的安全邊界：目前「一般移動」、「capture 初始定位」、「Keyence」、「final approach」、「HDR Z-stack」、「Exit retract」各自有不同程度的 planner、Twin、limit validation 與 readback。這正是目前看起來很多功能相似、實際保護程度卻不完全一致的主因。