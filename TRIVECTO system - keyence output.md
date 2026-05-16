


|                                          |     |
| ---------------------------------------- | --- |
| [[#### Keyence 讀值內容OUT1, OUT2, OUT3的差異]] |     |
| [[#### 比對option1跟option4的每個步驟]]          |     |
|                                          |     |
|                                          |     |
|                                          |     |
|                                          |     |


#### Keyence 讀值內容OUT1, OUT2, OUT3的差異
```
請詳細中文解釋5. Keyence 讀值內容OUT1, OUT2, OUT3的差異在哪裡, 甚麼是把 OUT1 做 normalization. 所以option 1 讀值內容OUT3對決定最後z position有甚麼幫助? 請詳細解釋option1跟option4 有玻璃時的 Z 公式跟無玻璃時的 Z 公式, 為何公式是這樣?
```

下面我分兩大段說明：

1. OUT1 / OUT2 / OUT3 分別代表什麼，以及 Option 1 / Option 4 怎麼用它們
2. Option 1 / Option 4 在「有玻璃」與「無玻璃」時的 Z 公式差異，以及為什麼公式長這樣

---

**一、Keyence OUT1 / OUT2 / OUT3 的差異**  
在目前這個 refactor 裡，我們把 MATLAB 裡的 sensor output 對應成：

`OUT1 = sensordist OUT2 = glassthick1 OUT3 = sensordist2`

可以先用一句話理解：

`OUT1 是主要表面距離 OUT2 是玻璃厚度 OUT3 是第二個表面 / 另一個 peak 的距離`

---

**OUT1 是什麼**  
OUT1 是最主要的 Keyence 距離讀值。  
在 Option 1 裡，它對應 MATLAB 的：

`sensordist`

在 micro MATLAB：

`[sensordist, hasglass, glassthick1] = watch_sensordistance_combined;`

在 macro/top MATLAB：

`[sensordist, hasglass, sensordist2, glassthick1] = watch_sensordistance_combined2;`

也就是說，OUT1 是用來決定主要 focus Z 的核心讀值。

目前 Option 1 的公式裡，OUT1 會直接影響最後 Z：

`target_z = zbase + zdisp - OUT1 + glasscorrect`

或無玻璃：

`target_z = zbase + zdisp_no_glass - OUT1`

所以：

`OUT1 越大，算出來的 target_z 越小 OUT1 越小，算出來的 target_z 越大`

這是 MATLAB-like empirical algorithm 的核心。

---

**OUT2 是什麼**  
OUT2 是玻璃厚度讀值。  
在 MATLAB 裡叫：

`glassthick1`

Option 1 用它來算玻璃補償量：

`glass_thick = OUT2 * out2_thickness_scale glasscorrect = (glass_thick - glass_base_mm) * glass_mod_z_per_mm`

目前 config：

`out2_thickness_scale: 1.0 glass_base_mm: 1.79 glass_mod_z_per_mm: -0.4`

所以預設：

`glass_thick = OUT2 glasscorrect = (OUT2 - 1.79) * -0.4`

例子：

`OUT2 = 1.79 glasscorrect = (1.79 - 1.79) * -0.4 = 0`

如果：

`OUT2 = 2.00 glasscorrect = (2.00 - 1.79) * -0.4 = -0.084`

意思是玻璃比 calibration baseline 厚，Z 位置要做一點經驗補償。

注意：  
Option 1 不自動乘 glass_refractive_index = 1.76。這是因為 Keyence CL-3000 可能已經開了折射率補償，OUT2 可能已經是物理厚度。

---

**OUT3 是什麼**  
OUT3 對應 MATLAB macro/top 裡的：

`sensordist2`

在 watch_findsurface3.m 裡有兩個 Z：

`Znew = Zbase + Zdisp2 - sensordist + glasscorrect; Znew2 = Zbase + Zdisp - sensordist2;`

對應 Python：

`target_z = zbase + zdisp_glass - OUT1 + glasscorrect target_z_secondary = zbase + zdisp_secondary - OUT3`

目前 macro 設定：

`zdisp_glass: 17.7 zdisp_secondary: 19.5197`

所以：

`target_z = zbase + 17.7 - OUT1 + glasscorrect target_z_secondary = zbase + 19.5197 - OUT3`

重點是：

`OUT1 決定最後回傳的 focus Z OUT3 只產生 target_z_secondary，預設不決定最後 Z`

---

**那 Option 1 裡 OUT3 對最後 Z position 有什麼幫助？**  
目前預設下，OUT3 **不直接決定最後 Z position**。

它的幫助主要是三種：

**1. 診斷 macro/top camera 的雙表面量測是否合理**

在 macro/top MATLAB 裡，sensordist 和 sensordist2 分別代表兩個 surface / peak。  
Znew 是最後 movement 用的 Z，Znew2 是另一個表面的診斷 Z。

如果你在 log 裡看到：

`target_z = 48.7 target_z_secondary = 49.0`

代表兩個 surface 推估出來的位置接近，讀值可能合理。

如果看到：

`target_z = 48.7 target_z_secondary = 58.0`

那可能代表 OUT3 peak 不穩、Keyence 抓到錯的表面、玻璃反射異常，或樣品局部形貌造成讀值不一致。

**2. 可以選擇變成 gatekeeper**

目前 config 是：

`require_out3_valid: false`

所以 OUT3 無效時，不會拒絕 candidate。

如果改成：

`require_out3_valid: true`

那 macro camera candidate 必須滿足：

`out3_valid_min < OUT3 < out3_valid_max`

目前就是：

`-6 < OUT3 < 8`

這會更貼近 MATLAB：

`max(sensordist, sensordist2) < 8 && min(sensordist, sensordist2) > -6`

也就是 OUT1 和 OUT3 都要在範圍內。

**3. 幫助日後比較 / 校正**  
即使現在不讓 OUT3 控制最後 Z，它仍然記錄在：

`self._last_keyence_af_report["attempts"]`

每次 attempt 都有：

`out1 out2 out3 target_z target_z_secondary out3_valid`

這對硬體 debug 很有用。你可以比較：

`OUT1 算出的 Z OUT3 算出的 secondary Z 實際影像清晰度`

如果長期發現 target_z_secondary 比 target_z 更接近最佳焦點，之後可以再調整模型。但目前根據你的要求，OUT3 預設只 debug，不控制最後 Z。

---

**什麼是把 OUT1 做 normalization？**  
這個只發生在 Option 4，不是 Option 1。

Option 4 的讀值函式是：

`_read_keyence_absolute_output_with_expansion()`

它可能不只在一個 Z 高度讀 OUT1。  
它會從：

`reference_z = keyence2_measure_start_z`

開始讀。如果中心點讀不到，它可能嘗試：

`reference_z + keyence_read_roi_dist reference_z - keyence_read_roi_dist ...`

也就是 Z 位置可能不完全等於 reference_z。

問題是：  
Keyence OUT1 是「在目前 probe_z 下量到的距離」。如果你把 Z 軸移動了，例如從 20 mm 移到 21 mm，OUT1 數值本身會跟著改變。那不同 probe_z 讀到的 OUT1 不能直接混在一起比較。

所以 Option 4 會做：

`normalized_out1 = raw_out1 + z_delta`

其中：

`z_delta = actual_z - reference_z`

簡單講：

`把在不同 Z 高度量到的 OUT1，換算回好像都是在 reference_z 量到的 OUT1。`

舉例：

`reference_z = 20.0 actual_z = 21.0 raw_out1 = 3.0`

如果 Z 軸多移了 1 mm，程式會依照 Z 軸方向設定修正 OUT1：

`normalized_out1 = 3.0 + 1.0 = 4.0`

這樣 Option 4 後面的公式：

`target_z = keyence2_base_z + normalized_out1`

就仍然可以基於同一個 reference_z calibration 來計算。

Option 1 為什麼不做 normalization？  
因為 Option 1 的公式本來就是：

`target_z = zbase + zdisp - OUT1`

每次計算都把「當下 zbase」放進公式裡了。  
它不是把所有 OUT1 換算到同一個 reference_z，而是每個 candidate 都用自己的 zbase 計算。

所以：

`Option 4 需要 OUT1 normalization，因為它要把不同 probe_z 的讀值統一回同一個 reference_z。 Option 1 不需要 normalization，因為公式本身已經包含當次 zbase。`

---

**二、Option 1 和 Option 4 的 Z 公式比較**  
下面分「無玻璃」和「有玻璃」。

---

**Option 1 無玻璃公式**  
micro：

`target_z = zbase + 17.0868 - OUT1`

macro：

`target_z = zbase + 18.5767 - OUT1`

這來自 MATLAB：

micro watch_findsurface2.m：

`Znew = Zbase + Zdispnoglass - sensordist;`

macro watch_findsurface3.m：

`Znew = Zbase + Zdispnoglass - sensordist;`

只是 micro 和 macro 的 Zdispnoglass 不一樣：

`micro: 17.0868 macro: 18.5767`

**為什麼是 zbase + zdisp - OUT1？**

可以這樣理解：

`zbase 是你目前把 Z 軸放到的量測高度 OUT1 是 Keyence 在這個高度看到的表面距離偏差 zdisp 是從 Keyence 測距幾何轉換到相機焦點位置的經驗校正常數`

如果 OUT1 變大，代表在目前 zbase 下，表面相對 sensor 的位置讓你需要把 target_z 往另一方向修正，所以公式是：

`- OUT1`

這不是純物理折射公式，而是 MATLAB calibration 出來的 empirical mapping。

---

**Option 1 有玻璃公式**  
micro：

`glasscorrect = (glass_thick - 1.79) * -0.4 target_z = zbase + 16.2847 - OUT1 + glasscorrect`

macro：

`glasscorrect = (glass_thick - 1.79) * -0.4 target_z = zbase + 17.7 - OUT1 + glasscorrect target_z_secondary = zbase + 19.5197 - OUT3`

來自 MATLAB：

micro：

`glasscorrect = (glassthick1 - glassbase) * glassmod; Znew = Zbase + Zdisp - sensordist + glasscorrect;`

macro：

`glasscorrect = (glassthick1 - glassbase) * glassmod; Znew = Zbase + Zdisp2 - sensordist + glasscorrect; Znew2 = Zbase + Zdisp - sensordist2;`

其中：

`glassbase = 1.79 glassmod = -0.4`

**為什麼有玻璃時不是用無玻璃的 zdisp？**

因為玻璃會造成光學路徑改變。  
相機看到的焦平面和 Keyence 測到的表面距離之間，不再跟無玻璃一樣。

所以 MATLAB calibration 分成：

`有玻璃 zdisp 無玻璃 zdispnoglass`

例如 micro：

`有玻璃: 16.2847 無玻璃: 17.0868`

這代表同一個 OUT1，在有玻璃和無玻璃時，推算出的相機焦點 Z 應該不同。

**為什麼還要 glasscorrect？**

zdisp_glass 是以某個基準玻璃厚度校正的。  
MATLAB 裡的基準是：

`glass_base_mm = 1.79`

如果實際玻璃厚度剛好是 1.79：

`glasscorrect = (1.79 - 1.79) * -0.4 = 0`

如果玻璃比較厚或比較薄，就做線性補償：

`glasscorrect = (實際厚度 - 基準厚度) * 每 mm 厚度造成的 Z 修正`

目前每 mm 厚度修正量是：

`-0.4 mm Z / mm glass`

也就是玻璃厚 1 mm，Z 修正約 -0.4 mm。

---

**Option 4 無玻璃公式**  
Option 4 走：

`calculate_keyence_focus_z()`

無玻璃時：

`target_z = keyence2_base_z + OUT1`

目前 config：

`keyence2_base_z: 22.7855`

所以：

`target_z = 22.7855 + OUT1`

**為什麼是 base + OUT1？**

Option 4 是另一套 calibration 思路：  
它假設 Keyence OUT1 本身可以直接作為相對於某個 calibrated base Z 的高度 offset。

也就是：

`keyence2_base_z 是 calibrated absolute focus base OUT1 是從這個 base 推出去的修正量`

所以公式是加法：

`base_z + OUT1`

---

**Option 4 有玻璃公式**  
有玻璃時：

`target_z = keyence2_base_z_glass + OUT1`

目前 config：

`keyence2_base_z_glass: 22.7855`

所以：

`target_z = 22.7855 + OUT1`

玻璃厚度則是：

`glass_thick = OUT2 * glass_refractive_index`

目前：

`glass_refractive_index: 1.76`

所以：

`glass_thick = OUT2 * 1.76`

**為什麼 Option 4 有玻璃時只是換 base？**

Option 4 把玻璃造成的光學差異濃縮進：

`keyence2_base_z_glass`

也就是說：

`無玻璃用 keyence2_base_z 有玻璃用 keyence2_base_z_glass`

它沒有像 Option 1 那樣根據每次 OUT2 厚度去調 target_z。  
Option 4 的 OUT2 主要是用來回傳 / 記錄玻璃厚度，不直接進入 target_z 公式。

---

**最重要的公式差異**  
可以直接看這張表：

|情境|Option 1|Option 4|
|---|---|---|
|micro 無玻璃|zbase + 17.0868 - OUT1|keyence2_base_z + OUT1|
|micro 有玻璃|zbase + 16.2847 - OUT1 + glasscorrect|keyence2_base_z_glass + OUT1|
|macro 無玻璃|zbase + 18.5767 - OUT1|keyence2_base_z + OUT1|
|macro 有玻璃|zbase + 17.7 - OUT1 + glasscorrect|keyence2_base_z_glass + OUT1|

最大差異是：

`Option 1: OUT1 是減掉 Option 4: OUT1 是加上`

這代表兩套 calibration 對 OUT1 的座標意義不同。  
Option 1 是 MATLAB empirical scan formula；Option 4 是 Keyence2 absolute formula。

---

**為什麼不能把兩個公式混用？**  
因為它們的 calibration reference 不一樣。

Option 1 的 reference 是：

`每次實際移到的 Zbase 相機別 zdisp constant OUT1 在該 Zbase 下的距離讀值`

所以公式必須含：

`zbase + zdisp - OUT1`

Option 4 的 reference 是：

`固定 keyence2_measure_start_z normalized OUT1 keyence2_base_z / keyence2_base_z_glass`

所以公式是：

`keyence2_base_z + OUT1`

如果把 Option 4 的 calculate_keyence_focus_z() 拿給 Option 1 用，就會破壞 MATLAB-like algorithm。  
反過來，如果把 Option 1 的 zbase + zdisp - OUT1 用在 Option 4，也會破壞 Keyence2 legacy benchmark。

所以現在程式刻意分開：

`Option 1 -> _calc_matlab_like_keyence_z() Option 4 -> calculate_keyence_focus_z()`

這也是保留 Option 4 作為比較基準的原因。






#### 比對option1跟option4的每個步驟
```
請比對option1跟option4的每個步驟, 並指出一樣跟不一樣的地方
```

下面是目前 repo 裡 **Option 1** 和 **Option 4** 的逐步比對。  
對應程式主要在 unified_driver.py (line 570)。

**一句話總結**  
Option 1 是新的 **MATLAB-like empirical scan**：嘗試多個 Zbase，每次讀 Keyence，套 MATLAB 經驗公式，找到第一個合理的 target_z。

Option 4 是原本的 **Keyence2 absolute calculation**：移到固定量測 Z，讀 OUT1/OUT2，走 calculate_keyence_focus_z() 用 keyence2_base_z / keyence2_base_z_glass 算 Z。

---

**總覽比較**

|步驟|Option 1|Option 4|一樣/不一樣|
|---|---|---|---|
|路由|af_option == 1 呼叫 _af_keyence_matlab_like()|af_option == 4 呼叫 _af_keyence_absolute()|不一樣|
|用途|MATLAB-like empirical Keyence AF|Keyence2 legacy absolute AF|不一樣|
|是否用 Keyence|是|是|一樣|
|是否移到 Keyence XY|是|是|一樣|
|是否套 z_focus profile|是|是|一樣|
|Z 量測策略|多個 Zbase 候選掃描|單一 keyence2_measure_start_z，可做 ROI/Z expansion|不一樣|
|OUT1 讀法|read_stable_value() 優先|_read_keyence_absolute_output_with_expansion() 內讀 read_stable_value()|類似但不一樣|
|OUT2 厚度轉換|OUT2 * out2_thickness_scale，預設 1.0|OUT2 * glass_refractive_index|不一樣|
|OUT3|可讀，用於 macro diagnostic|不使用 OUT3|不一樣|
|Z 公式|MATLAB empirical formula|keyence2_base_z + OUT1 類公式|不一樣|
|是否呼叫 calculate_keyence_focus_z()|否|是|不一樣|
|失敗 fallback|fallback_z_default 或 base_z|fallback_z_default 或 base_z|類似|
|finally cleanup|還原 profile、XY 回相機中心|還原 profile、Z 回 measurement Z、XY 回相機中心|類似但不完全一樣|
|上層 final Z|上層加 parfocal_z_offset 後移動|同樣由上層加 offset 後移動|一樣|

---

**1. 路由階段**

在 run_robust_autofocus() 裡：

python

`if af_option == 4: self._af_keyence_absolute(...) elif af_option == 1: self._af_keyence_matlab_like(...)`

也就是說：

text

`Option 1 -> 新 MATLAB-like 流程 Option 4 -> 舊 Keyence2 absolute 流程`

這裡兩者最大的差異是：  
**Option 1 不再走舊 closed-loop，也不走 Option 4 的公式。**

---

**2. 初始安全與 motion profile**

兩者都會：

python

`self.check_safety_lock() self.zaber.apply_motion_profile("z_focus", ["stage_L_Z"])`

Option 4 的 _af_keyence_absolute() 本身沒有直接呼叫 check_safety_lock()，但一般是由外層 run_robust_autofocus() 先呼叫。Option 1 的 _af_keyence_matlab_like() 內部也有再呼叫一次。

相同點：

text

`都會使用 z_focus motion profile 都會在 finally 裡恢復 default profile`

不同點：

text

`Option 1 helper 自己也檢查 safety lock Option 4 主要依賴外層入口檢查`

---

**3. XY 移動到 Keyence 雷射位置**

兩者都使用：

python

`_get_keyence_xy(base_x, base_y, cam_id)`

也就是從相機中心：

text

`base_x, base_y`

加上 Keyence offset，變成：

text

`keyence_x, keyence_y`

macro camera 使用：

yaml

`macro_offset_x macro_offset_y`

micro camera 使用：

yaml

`micro_offset_x micro_offset_y`

相同點：

text

`兩者都會先把 XY 從 camera center 移到 Keyence probe XY。 量測完成後，兩者都會把 XY 移回 camera center。`

---

**4. Z 量測策略**

這是核心差異之一。

Option 1：

python

`zbase_candidates = [48, 40, 32, 24, 16, 8]`

流程是：

text

`移到 Zbase=48 -> 讀 Keyence -> 算 target_z -> 合法就用 如果不合法 -> 試 Zbase=40 再不合法 -> 試 32, 24, 16, 8`

也就是 MATLAB watch_findsurface2.m / watch_findsurface3.m 的掃描方式。

Option 4：

python

`measure_start_z = af_cfg["keyence2_measure_start_z"]`

目前 config 是：

yaml

`keyence2_measure_start_z: 20.0`

Option 4 先移到這個固定 Z，再呼叫：

python

`_read_keyence_absolute_output_with_expansion(...)`

它會先讀中心點。如果中心點讀不到，才根據這些參數做擴展：

yaml

`keyence_read_roi_expan keyence_read_z_expan keyence_read_roi_dist`

所以：

text

`Option 1 = 多個固定 Zbase 掃描 Option 4 = 固定基準 Z + ROI/Z expansion robust read`

---

**5. Keyence 讀值內容**

Option 1 讀：

text

`OUT1 = sensordist OUT2 = glassthick1 OUT3 = sensordist2`

讀值 helper 是：

python

`_read_keyence_out123_for_matlab_like()`

讀法：

python

`OUT1: read_stable_value(samples=5, delay=0.01) OUT2: read_out_channel("02") OUT3: read_out_channel("03")`

Option 4 讀：

text

`OUT1 OUT2`

透過：

python

`_read_keyence_absolute_output_with_expansion()`

它讀中心點與擴展點，會記錄每個 probe sample，並把 OUT1 做 normalization：

python

`_normalize_keyence_out1_for_probe_z(...)`

不同點：

text

`Option 1 不做 ROI expansion，不做 OUT1 normalize。 Option 4 做 ROI/Z expansion，並可能把不同 probe_z 的 OUT1 normalize 回 reference_z。 Option 1 會讀 OUT3。 Option 4 不讀 OUT3。`

---

**6. OUT1 合法條件**

Option 1 對 OUT1 有 MATLAB window。

micro camera：

python

`-6 < OUT1 < 2`

macro camera：

python

`-6 < OUT1 < 8`

如果 OUT1 不在範圍內，這個 Zbase 失敗，繼續試下一個 Zbase。

Option 4 對 OUT1 的主要有效條件比較寬：

python

`OUT1 > -90 OUT1 != -999`

只要不是 Keyence 無效值，就會進入公式。

所以：

text

`Option 1 對 OUT1 有相機別的 MATLAB 有效窗口。 Option 4 只判斷 OUT1 是否為 Keyence invalid/error。`

---

**7. OUT2 / 玻璃厚度算法**

Option 1：

python

`glass_thick = OUT2 * out2_thickness_scale`

目前：

yaml

`out2_thickness_scale: 1.0`

所以預設就是：

python

`glass_thick = OUT2`

這是因為 Keyence CL-3000 可能已經開了 refractive index correction。

Option 4：

python

`glass_thick = OUT2 * glass_refractive_index`

目前：

yaml

`glass_refractive_index: 1.76`

所以：

python

`glass_thick = OUT2 * 1.76`

這是很重要的差異。

text

`Option 1 預設不乘 1.76。 Option 4 會乘 glass_refractive_index。`

---

**8. 有玻璃時的 Z 公式**

Option 1，micro：

python

`glasscorrect = (glass_thick - 1.79) * -0.4 target_z = zbase + 16.2847 - OUT1 + glasscorrect`

Option 1，macro：

python

`glasscorrect = (glass_thick - 1.79) * -0.4 target_z = zbase + 17.7 - OUT1 + glasscorrect target_z_secondary = zbase + 19.5197 - OUT3`

Option 4，有玻璃：

python

`target_z = keyence2_base_z_glass + OUT1 glass_thick = OUT2 * glass_refractive_index`

目前 config：

yaml

`keyence2_base_z_glass: 22.7855`

所以 Option 4 是：

python

`target_z = 22.7855 + OUT1`

差異非常大：

text

`Option 1 用 zbase + calibrated displacement - OUT1 + empirical glass correction。 Option 4 用 keyence2_base_z_glass + OUT1。 Option 1 的 OUT1 是被減掉。 Option 4 的 OUT1 是被加上。`

---

**9. 無玻璃時的 Z 公式**

Option 1，micro：

python

`target_z = zbase + 17.0868 - OUT1`

Option 1，macro：

python

`target_z = zbase + 18.5767 - OUT1 target_z_secondary = zbase + 18.5767 - OUT3`

Option 4，無玻璃：

python

`target_z = keyence2_base_z + OUT1`

目前 config：

yaml

`keyence2_base_z: 22.7855`

所以 Option 4 是：

python

`target_z = 22.7855 + OUT1`

同樣差異：

text

`Option 1 依照目前量測 Zbase 算相對 focus。 Option 4 依照固定 base_z 算 absolute focus。`

---

**10. OUT3 的角色**

Option 1：

text

`macro_cam_1 / macro_cam_2 會讀 OUT3 OUT3 用來計算 target_z_secondary 預設不要求 OUT3 有效`

目前 config：

yaml

`require_out3_valid: false`

所以 OUT3 壞掉時：

text

`Option 1 仍可用 OUT1 算 target_z target_z_secondary = None`

如果你想完全貼近 MATLAB watch_findsurface3.m，可以把 macro 的：

yaml

`require_out3_valid: true`

Option 4：

text

`完全不使用 OUT3`

---

**11. target_z 合法範圍**

Option 1 會檢查：

python

`z_valid_min < target_z < z_valid_max`

目前：

yaml

`z_valid_min: 0.0 z_valid_max: 60.0`

所以：

python

`0 < target_z < 60`

不合法就試下一個 Zbase。

Option 4 沒有同樣的 0~60 candidate reject 機制。只要 calculate_keyence_focus_z() 回傳不是 None，就接受。

所以：

text

`Option 1 會因 target_z 超過範圍而繼續找下一個 Zbase。 Option 4 沒有多候選 retry；OUT1 有效就直接用公式結果。`

---

**12. 成功條件**

Option 1 成功條件：

text

`某一個 Zbase 的 OUT1 合法 有玻璃時 OUT2 合法 如果 require_out3_valid=true，OUT3 也要合法 target_z 在 0~60 內`

成功後：

python

`return best_z, True, glass_thick`

Option 4 成功條件：

text

`_read_keyence_absolute_output_with_expansion() 成功取得有效 OUT1 calculate_keyence_focus_z() 回傳 target_z`

成功後：

python

`return best_z, True, glassthick`

相同點：

text

`最後都回傳 best_z, success, glass_thick。`

不同點：

text

`Option 1 是 first valid Zbase candidate。 Option 4 是 one resolved robust measurement。`

---

**13. 失敗處理**

Option 1 如果全部 Zbase 都失敗：

python

`best_z = fallback_z_default if provided else base_z success = False glass_thick = 0.0`

Option 4 如果 OUT1 無效或無法計算：

python

`best_z = fallback_z_default if provided else base_z success = False glass_thick = 0.0`

相同點：

text

`都不 crash。 都會 fallback。 都會回傳 success=False。`

不同點：

text

`Option 1 的 report 會列出每個 Zbase attempt 和失敗原因。 Option 4 的 report 會列出 measurement/probe_samples/roi expansion 結果。`

---

**14. cleanup 行為**

Option 1 finally：

python

`apply_motion_profile("default") XY 回 camera center`

Option 4 finally：

python

`apply_motion_profile("default") 如果 measure_cleanup_z 有值，Z 回 keyence2_measure_start_z XY 回 camera center`

所以：

text

`兩者都會恢復 default profile。 兩者都會把 XY 回到相機中心。 Option 4 額外會把 Z 回到 measurement Z。 Option 1 不額外把 Z 回固定 measurement Z，Z 會停在最後一次測量的 Zbase，等上層移到 final target。`

---

**15. 上層 final Z 移動**

兩者回傳的都是：

python

`best_z`

上層再做：

python

`target_z = best_z + parfocal_z_offset`

或在 template pipeline 裡再加：

python

`parfocal_offset + z_mod + z_micro_offset`

這點是一樣的：

text

`Option 1 和 Option 4 都不應該自己完成最終拍照 Z。 它們只提供 coarse AF 算出的 best_z。 最後 camera-center 的拍照位置由上層決定。`

---

**16. report 差異**

Option 1 report：

python

`method: "keyence_matlab_like" zbase_candidates attempts selected_attempt target_z_after_laser glass_thickness fallback_used`

每個 attempt 會有：

python

`zbase out1 out2 out3 glass_thick glasscorrect target_z target_z_secondary valid reason`

Option 4 report：

python

`method: "keyence_absolute" measure_start_z measurement probe_samples out1 out2 target_z_after_laser glass_thickness fallback_used`

所以：

text

`Option 1 report 適合看 MATLAB candidate scan 為什麼選/不選。 Option 4 report 適合看 robust sampling、ROI expansion、normalized OUT1。`

---

**最核心差異再濃縮一次**

Option 1：

text

`多 Zbase 掃描 OUT1/OUT2/OUT3 相機別 MATLAB empirical formula OUT2 預設不乘折射率 OUT1 是公式中被減掉 target_z 要在 0~60 第一個合法 candidate 成功`

Option 4：

text

`單一 keyence2_measure_start_z OUT1/OUT2 ROI/Z expansion robust read calculate_keyence_focus_z() OUT2 乘 glass_refractive_index OUT1 是公式中被加上 沒有 Zbase candidate scan`

所以如果你在硬體上看到兩者算出來的 Z 差很多，這是預期的：  
它們不是同一套物理模型，只是都使用 Keyence 讀值來推估 focus Z。




```
我要做的變更是把本來的autofocus_option:1 MATLAB-like empirical scan 修改下列: 4. 將option 1的Z 量測策略改成option 4的固定基準 Z + ROI/Z expansion robust read. 6. OUT1 合法條件把option 1跟option 4都設成out1_min < OUT1 < out1_max, 然後這兩個值都在hardware_config的autofocus.af_settings設定
```

已完成你這次指定的兩個變更。

**改了什麼**  
在 unified_driver.py (line 833)：

Option 1 現在不再掃：

yaml

`[48, 40, 32, 24, 16, 8]`

而是改成：

text

`固定 reference Z + Option 4 同一套 ROI/Z expansion robust read + MATLAB-like empirical Z 公式`

也就是 Option 1 現在的量測流程變成：

text

`1. 移到 Keyence XY 2. Z 移到 keyence1_measure_start_z 3. 呼叫 _read_keyence_absolute_output_with_expansion() 4. 取得 normalized OUT1 / OUT2 5. 另外讀 OUT3 作 debug 6. 用 Option 1 的 MATLAB-like 公式算 target_z`

在 hardware_config.yaml (line 267) 新增：

yaml

`out1_min: -6.0 out1_max: 8.0`

這兩個值現在是 **Option 1 和 Option 4 共用的 OUT1 合法窗口**。

---

**Option 1 現在如何決定 Z**  
以前 Option 1 是：

text

`Zbase=48 讀一次 失敗再試 40 失敗再試 32...`

現在 Option 1 是：

yaml

`keyence1_measure_start_z: 20.0`

所以它會以 20.0 當 reference Z，使用 Option 4 的 robust read：

python

`_read_keyence_absolute_output_with_expansion( reference_z=keyence1_measure_start_z )`

這個 helper 會：

text

`先讀中心點 中心點 OUT1 不合法就試 ROI expansion ROI 也失敗就試 Z expansion 不同 Z 讀到的 OUT1 會 normalize 回 reference_z`

最後 Option 1 用：

python

`target_z = reference_z + zdisp - normalized_OUT1 + glasscorrect`

例如 micro 有玻璃：

python

`target_z = keyence1_measure_start_z + 16.2847 - OUT1 + glasscorrect`

---

**Option 4 的 OUT1 合法條件也改了**  
Option 4 原本只判斷：

python

`OUT1 > -90 OUT1 != -999`

現在會套：

python

`out1_min < OUT1 < out1_max`

目前是：

python

`-6.0 < OUT1 < 8.0`

這個判斷發生在兩層：

text

`1. robust read sample 階段： normalized OUT1 不在窗口內，該 sample 不算有效 2. calculate_keyence_focus_z() 階段： OUT1 不在窗口內，直接回傳 None，Option 4 fallback`

---

**Config 清理**  
我也移除了 matlab_like_focus 裡舊的：

yaml

`zbase_candidates settle_s out1_valid_min out1_valid_max`

因為現在 Option 1 已經不再掃多個 Zbase，而且 OUT1 window 改由：

yaml

`hardware.autofocus.af_settings.out1_min hardware.autofocus.af_settings.out1_max`

統一控制。

---

**保留不變**  
Option 4 的核心公式仍然是：

python

`target_z = keyence2_base_z + OUT1 target_z = keyence2_base_z_glass + OUT1`

Option 1 的核心 MATLAB-like 公式仍然是：

python

`target_z = reference_z + zdisp - OUT1 + glasscorrect`

所以這次是：

text

`Option 1 的讀值策略向 Option 4 靠攏 但 Option 1 的 Z 公式仍然保留 MATLAB-like empirical 公式`

---

**已跑安全檢查**  
我沒有執行任何硬體 script。只跑了：

text

`python compile OK hardware_config.yaml parse OK mock formula/window test OK`

mock 結果確認：

text

`Option 1: OUT1=0.5 通過，OUT1=9.0 被 out1_max=8.0 擋掉 Option 4: OUT1=0.5 通過，OUT1=9.0 回傳 None`