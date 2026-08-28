
|                                              |     |
| -------------------------------------------- | --- |
| [[#### 目前的系統設計可以做到單一數值就很重要譬如Dial文字上的Y的高度. ]] |     |
| [[#### pdf report應該包含上萬(?)個identifier]]      |     |
| [[#### pdf report目前有的identifier數量]]          |     |
|                                              |     |
|                                              |     |
|                                              |     |
|                                              |     |



#### 目前的系統設計可以做到單一數值就很重要譬如Dial文字上的Y的高度. 
```
這個moonlight systemy最重要的分析結果就是authentication result(and pdf report). 所以分析結果裡面(analysis_results_v2(+ experiment_results 舊表))的單一數值就很重要譬如Dial文字上的Y的高度. authentication result就是基於這些非常多的許多單一數值判斷, 譬如在Rolex某一系列的100個真的手錶的Dial文字上的Y的高度會是一個distribution, 而如果現在檢測一個這系列手錶的Dial文字上的Y的高度在distribution太遠, 就可能這是假的. 所以能夠很快從非常多同樣系列手錶的找到特定單一數值就很重要也要快速, 因為這樣才能建立distribution. 那目前的系統設計可以做到這一點嗎? 如果不行要怎麼修改
```
我追蹤了「Dial 文字 Y 高度」這類單一測量值實際上是怎麼從演算法產生、流向資料庫/資料湖的完整路徑,直接回答你的問題。

## 一、直接回答:目前系統「本身」做不到,但你們已經在往正確方向走了

**如果只看本地 SQLite(`analysis_results_v2`/`experiment_results`)和 DynamoDB,答案是不行,而且不是「效能差一點」,是「架構上就不是為這種查詢設計的」**——這點我在上一輪深挖時已經確認:單一數值(像 Y 高度這種)被整包塞進一個 `data_json`/`data` TEXT 欄位,整個專案**沒有任何地方用到 SQLite 的 `json_extract`**,要拿出一個數值必須把每一列的 JSON 整包讀出來、在 Python 裡 `json.loads()` 解析。100 支錶或許還能忍,但你自己也提到這個判斷方法的核心就是要建立 distribution——一旦資料量到幾千、上萬支錶,這種「查一列、解一次 JSON、再看有沒有你要的那個 key」的做法,速度會線性變差,完全稱不上「快速」。

**但好消息是**:我們前面幾輪已經一起設計並且部分實作了 Parquet/Iceberg 資料湖(`moonlight_lake.analysis_facts`)+ Athena 分析層,**這個架構的存在理由,就是精確對應你現在描述的這個用例**——不是要另外再設計一套新東西。

## 二、驗證:Y 高度這類數值,實際上會不會真的流進資料湖

我追蹤了實際的呼叫鏈,確認結論是「會,但有一個前提沒補上」:

演算法(例如 diallume_algo.py / diallume_shape_algo.py 這類跟 dial 文字/圖案幾何相關的演算法)
  輸出一個 dict,例如 {"text_Y_height_mm": 2.31, "confidence": 0.94, ...}
        ↓
core/workflow_manager.py:1043 的 _run_asset_driven_analysis()
  這是「對指定 asset 觸發一組演算法、逐一把結果寫進資料庫」的通用機制
  for algo_name, algo_res in (res or {}).items():
        ↓
core/workflow_manager.py:1068 呼叫
  self.data_mgr.db.register_analysis_result_v2({...})
        ↓
DB/db_manager.py 的 register_analysis_result_v2()
  【這一步很關鍵】會自動 join image_assets + watch_metadata,
  把 brand/model/reference 一起攤平進這筆記錄的快照(見上一輪我確認過的程式碼)
        ↓
同一個 transaction 裡插入三筆 sync_outbox 記錄(target='s3'/'catalog'/'lake')
        ↓
target='lake' 這筆被 LakeETLJob.flatten_payload() 撿走
  它會遞迴走過整個結果 dict,把每一個數值型的 leaf 欄位
  (只要 key 名稱不在 metadata 排除清單裡)各自變成一列
  {result_key: "text_Y_height_mm", result_value: 2.31, brand: "Rolex", 
   model: "Daytona", reference: "116500LN", ...}
        ↓
真正 commit 進 Iceberg table moonlight_lake.analysis_facts

也就是說:**只要「Y 高度」這個數值在演算法輸出的 JSON 裡有一個清楚命名的獨立 key(例如 `text_Y_height_mm`),整條管線是全自動的,不需要為這個特定測量值另外寫程式碼**——這是我們設計 `LakeETLJob._iter_metrics()` 時刻意做成「通用攤平」的原因,任何數值型的結果都會自動被撿起來變成一個可查詢的 `result_key`。這點建議你直接去確認一下實際的 dial 文字幾何演算法(`tasks/task_algos/diallume_*.py` 或相關的 OCR 模組)輸出的 dict 裡,這個高度值到底叫什麼名字,確保它是一個乾淨的數值欄位,不是被包在一張圖片路徑或一段字串描述裡。

## 三、我找到一個具體、明確、需要補的小缺口:`reference` 沒有被開放成查詢參數

這是我這輪深挖後發現的重點:

- Iceberg table `analysis_facts` **本來就有 `reference` 這個欄位**(我們最早設計 DDL 時就放進去了)。
- `register_analysis_result_v2()` **在寫入當下就已經把 reference join 進去了**(上一輪確認過的程式碼)。
- 但是 `data_manager/analytics_query.py` 的 `AnalyticsQueryService.get_distribution()`/`compare_series()`,**目前只公開 `brand`、`model` 兩個篩選參數,沒有 `reference`**。

這個缺口對你的用例影響很大——你講的「Rolex **某一系列**」如果精確到 Reference number(例如 116500LN vs 116509 是同一個 model「Daytona」底下完全不同的兩支錶,材質、尺寸、Dial 排版都可能不一樣),用現在的 `get_distribution(brand="Rolex", model="Daytona")` 查出來的 distribution 其實混雜了好幾個不同 Reference 的資料,對「這支特定 Reference 的真品應該長怎樣」這個判斷會失真。

**建議修改**(這是一個很小的改動,不涉及資料模型,資料早就在表裡了,只是查詢層沒開放):

def get_distribution(
    self,
    brand: str,
    model: Optional[str] = None,
    reference: Optional[str] = None,   # 新增
    result_key: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
) -> List[Dict[str, Any]]:

`_where_clauses()` 裡比照 `model` 的驗證方式加一段 `reference` 驗證,SQL 裡多一個 `AND reference = ?` 子句即可。`compare_series()` 也建議同步加上,因為你的用例裡也提到「同 reference 但比較不同批次/不同檢測時間點」這種需求。

## 四、真正的查詢會長什麼樣子

補上 `reference` 參數之後,你要的查詢就是這樣:

SELECT result_value, COUNT(*) AS cnt
FROM moonlight_lake.analysis_facts
WHERE brand = 'Rolex'
  AND reference = '116500LN'
  AND result_key = 'text_Y_height_mm'
GROUP BY result_value
ORDER BY result_value;

或者直接用 `AnalyticsQueryService.get_distribution()` 已經寫好的自動分桶(bucket)邏輯,拿到一個直方圖形式的分布,不需要自己刻。因為 Iceberg 是欄式儲存 + 依 `brand`/日期做 partition,這個查詢**只會讀取 `result_key`/`result_value` 這兩欄的資料、只掃描 `brand=Rolex` 這個 partition**,不會像 SQLite 那樣得把每一列的整包 JSON 都讀出來解析——這正是「快」的來源。

## 五、一個更深層的架構提醒:Athena 本身不是「即時(毫秒級)」的

這點我覺得對你的用例特別重要,必須誠實提出來:**Athena 查詢有固定的啟動延遲**(送出查詢、排隊、執行,通常是幾秒鐘等級,不是毫秒級),而且是**批次查詢引擎,不是給「拍完一張照片馬上要跟 distribution 比對、判斷是否異常」這種即時互動場景直接用的**。如果你的使用情境是「檢測站當下拍完 Dial 文字,系統要立刻告訴操作員這個高度是不是離 distribution 太遠」,直接每次都對 Athena 下一次 distribution 查詢,體驗會偏慢。

**建議的兩層設計**(這是我認為值得你們規劃的下一步,不是現在就要做,但值得先想清楚):

1. **離線/週期性**:用 Athena 定期(例如每天,或每次某個 Reference 累積到一定樣本數時)計算每個 `(reference, result_key)` 組合的統計摘要——平均值、標準差、幾個百分位數——這件事 Athena 很適合做。
2. **即時比對用的小型摘要表**:把算好的統計摘要(不是原始資料,只是幾個數字:mean、std、p5、p95)存進一張很小的表——可以是 DynamoDB 一個新的 item 類型(例如 `PK=Reference#116500LN, SK=STAT#text_Y_height_mm`),也可以是本地 SQLite 新增一張小表——供拍照當下**直接查這張小表算 z-score**(`(觀測值 - mean) / std`),不需要每次都去問 Athena。這樣「建立 distribution」這個重活交給 Athena/Iceberg 做,「即時判斷是否異常」則是查一張隨時可以秒回的小表,兩邊各司其職。

## 六、總結

| 你的需求                                          | 現況                                                         | 需要的動作                                         |
| --------------------------------------------- | ---------------------------------------------------------- | --------------------------------------------- |
| 快速從很多同系列手錶找到特定單一數值                            | ❌ 本地 SQLite/DynamoDB 做不到;✅ 已設計好的 Iceberg/Athena 資料湖架構上可以做到 | 確認 `outbox_enabled` 有開、演算法輸出的數值 key 命名清楚      |
| 依 Reference number(而不只是 model)篩選 distribution | ⚠️ 資料早就有這個欄位,但查詢層沒開放                                       | 在 `analytics_query.py` 補上 `reference` 參數(小改動) |
| 拍照當下即時判斷是否偏離 distribution 太遠                  | ❌ 目前架構(Athena)不是為毫秒級即時查詢設計的                                | 建議規劃「離線算統計摘要 + 即時查小表」的兩層設計                    |





#### pdf report應該包含上萬(?)個identifier
```
那以一個watch來說, 之前設計建立的pdf report應該包含上萬(?)個identifier(每個identifier代表之前提到的單一數值譬如Dial文字上的Y的高度). 所以要判斷一個watch是否是真的, 就要拿現在這個手錶的所有identifier, 跟同個系列的所有identifier的distribution做比對. 目前系統可以順暢做到這一點嗎? 還是有更好的設計? 還有目前scan的手錶在App裡面authentication section可以選擇是learning mode or authentication mode, 應該就是目前這個watch scan的identifier要不要加入distribution. 在learning mode每個watch component(譬如Dial, Hands, Case, Bracelet-links....)也可以由user給予"original","Authentic replacements", "Forgery", "Aftermarket", "Modified", "Incorrect Authentic", "Missing", "Not Applicable". 請深入分析設計如何在learning mode可以結合每個watch component(譬如Dial, Hands, Case, Bracelet-links....)也可以由user給予"original","Authentic replacements", "Forgery", "Aftermarket", "Modified", "Incorrect Authentic", "Missing", "Not Applicable".建立一些distributions(or rules), 然後讓未來放入新的watch scans在 authentication mode可以準確判斷這個watch的狀態(maybe "original","Authentic replacements", "Forgery", "Aftermarket", "Modified", "Incorrect Authentic", "Missing", "Not Applicable"). 也要搭配上目前Database的設計以及可能會掃數萬個watch等等. 請先深入分析設計並提供優點缺點評價
```
我先查了目前 App 裡「learning mode / authentication mode」這個元件標記功能實際上是怎麼寫的,結果找到一個**在你開始設計統計比對之前,必須先解決的根本性缺口**——這個發現會影響整個設計的第一步,所以我先講這個,再進入完整架構設計。

---

## 一、關鍵發現:元件標記功能目前完全跟其他資料「斷連」

我確認了 `App/main.py:12433-12442` 的 8 種狀態清單,跟你講的完全一致:`Original`、`Authentic replacements`、`Forgery`、`Aftermarket`、`Modified`、`Incorrect Authentic`、`Missing`、`Not applicable`。使用者在 UI 上點選每個元件(Dial、Hands、Case、Bracelet-links…)的狀態後,存進 `self.comp_auth_states` 這個記憶體字典。

但存檔的地方(`App/main.py:30081-30103` 的 `_save_component_results()`)是這樣寫的:

save_dir = str(get_report_result_dir("component_results", create=True))
save_path = os.path.join(save_dir, f"component_results_{timestamp}.json")
final_results = {
    "timestamp": timestamp,
    "learning_mode": self.comp_learning_mode,
    "components": self.comp_auth_states
}

我對照了實際存出來的檔案(`output_results/component_results_20260726_032402.json`):

{
    "timestamp": "20260726_032402",
    "learning_mode": true,
    "components": {"Movement": "Forgery", "Case": "Forgery", "Dial": "Forgery", ...}
}

**這個 JSON 檔案裡完全沒有 `watchid`、沒有 `Reference`、沒有 `TemplateID`、沒有任何可以跟這次掃描對應起來的識別碼**——只是一個時間戳記 + 一個元件狀態字典,存在一個跟 `Local_Data/{watchid}/...` 完全無關的獨立資料夾裡。也就是說:**「這個 Dial 被標記成 Forgery」跟「這個 Dial 的 Y 高度是 2.1mm」這兩筆資訊,目前完全沒有共同的 key 可以連起來**——你想做的「用元件標籤建立 distribution」這件事,現在連最基本的資料連結都不存在。

這是我整個設計的第一步一定要先解決的地基問題,不能跳過。

---

## 二、整體設計原則:分階段,不要一開始就做 ML

在講細節之前先講一個判斷:你描述的問題本質上是「用少量標記樣本,對多維度數值特徵做多分類判斷」——這聽起來很適合機器學習,但我建議**不要一開始就往深度學習/複雜模型的方向走**,原因:

1. 8 個類別裡,樣本量會**極度不平衡**——`Original`(正品基準)跟可能大量測試過的 `Forgery` 應該會有較多樣本,但 `Aftermarket`/`Modified`/`Incorrect Authentic`/`Missing` 這幾類,在某個特定 Reference 底下,可能一開始只有個位數樣本,根本不夠訓練任何統計模型。
2. 現有基礎設施是 SQL/Athena 為核心,沒有 ML 訓練/推論的 pipeline(模型版本管理、特徵工程、serving)。  
    3 你要的是「可解釋、可稽核」的鑑定依據(這是鑑定服務的核心價值),純黑箱 ML 模型不容易跟操作員解釋「為什麼判斷這是 Forgery」。

我建議的是**三階段漸進式設計**,每一階段都能立刻用、且隨著資料量增加自然升級到下一階段,不需要推倒重來。

---

## 三、Phase 0(先決條件):把元件標籤正確接進資料模型

**新增一張本地 SQLite 表 `component_labels`**,並比照這幾輪已經建立的 Transactional Outbox 模式接進去:

CREATE TABLE component_labels (
    label_id TEXT PRIMARY KEY,
    watchid TEXT NOT NULL,
    run_id TEXT,
    template_id TEXT,
    reference TEXT,
    component_name TEXT NOT NULL,   -- 'Dial', 'Hands', 'Case', 'Bracelet-links', ...
    label TEXT NOT NULL,            -- 8 選 1
    is_learning_mode INTEGER,       -- 對應 comp_learning_mode
    labeled_by TEXT,                -- 操作員帳號(稽核用)
    labeled_at REAL,
    model_version TEXT              -- 用來標記這筆標籤是在哪個「規則/統計模型版本」之前打的
)
CREATE INDEX idx_component_labels_ref ON component_labels(reference, component_name, label)

`_save_component_results()` 改成寫進這張表(在同一個 transaction 裡,跟現有 `register_image_asset`/`register_analysis_result_v2` 一樣的模式),並且**同時插入 `sync_outbox` 記錄**(`target='catalog'` 推去 DynamoDB、`target='lake'` 推去 Iceberg)——這樣元件標籤才會跟我們前幾輪設計的整條資料湖管線接上。

**另外需要一個「元件 → 哪些量測值屬於它」的對應表**——這是目前系統裡完全沒有的東西,而且是整個設計能不能運作的關鍵:`Dial` 這個元件包含哪些 `(view_name, point_name, algo_task_name/result_key)` 組合?這個對應關係目前只存在人的認知裡,沒有寫進任何 config。建議新增一個版本化的設定檔(可以比照 `config/internalnum_config.yaml` 的做法),例如:

component_identifier_map:
  Dial:
    - {view_name: "Front", point_name: "DialText", result_key_prefix: "text_"}
    - {view_name: "Front", point_name: "DialMarkers", result_key_prefix: "marker_"}
  Hands:
    - {view_name: "Front", point_name: "Hands", result_key_prefix: "hand_"}
  ...

沒有這張對應表,「這個 Dial 元件對應到哪些數值」永遠只能靠人工硬記,沒辦法自動化。

---

## 四、Phase 1:規則 + Z-score 異常偵測(現有基礎設施就能做,樣本量要求最低)

這個階段的核心想法是:**把 8 個類別拆成兩種完全不同性質的問題來處理**,不要用同一套邏輯硬套。

### ① 「有沒有拍到/有沒有這個元件」——這是存在性問題,不是統計問題

`Missing`(缺失)、`Not applicable`(不適用,例如某些 Reference 天生沒有這個元件)——這兩類**根本不需要比對數值分布**,用規則判斷就好:

- 如果這個 watchpoint 該拍的 capture 在 `image_assets`/`capture_instances` 裡完全沒有記錄 → `Missing`
- 如果這個 Reference 的 template 裡,這個元件本來就標記為不存在(例如某些款式沒有 Bracelet-links)→ `Not applicable`

這兩類可以最先、最快實作,而且**不受樣本量不足的影響**,因為判斷依據是「有沒有資料」而不是「數值像不像」。

### ② 「數值像不像正品」——這才是真正需要 distribution 比對的部分

對 `Original`/`Authentic replacements`/`Forgery`/`Aftermarket`/`Modified`/`Incorrect Authentic` 這 6 類,設計如下:

**Step 1 — 建立每個 (Reference, Component, result_key, label) 組合的統計摘要**

這是上一輪我們討論的 Athena/Iceberg 查詢的自然延伸——只要把剛剛新增的 `component_labels` 併進 Iceberg ETL(`LakeETLJob` 寫入 `analysis_facts` 的時候多帶一個 `label` 維度,或者另外用一張 `component_labels` 對應的 lake table 跟 `analysis_facts` 用 `watchid` join),就能算出:

SELECT reference, component, result_key, label,
       COUNT(*) AS n,
       AVG(result_value) AS mean_value,
       STDDEV_SAMP(result_value) AS std_value,
       APPROX_PERCENTILE(result_value, 0.05) AS p5,
       APPROX_PERCENTILE(result_value, 0.95) AS p95
FROM analysis_facts af JOIN component_labels cl ON af.watch_id = cl.watchid
WHERE af.brand = 'Rolex' AND af.reference = '116500LN'
GROUP BY reference, component, result_key, label;

**Step 2 — 樣本量不足時的降級策略**(這點很重要,你的用例裡有些類別天生樣本稀少)

|樣本量情境|處理方式|
|---|---|
|`n >= 30`(經驗法則門檻,可調整)|用這個 Reference 自己的 mean/std 做 Z-score 比對,可信度高|
|`n` 太少,但同 Model 底下其他 Reference 加起來夠多|退而求其次,借用同 Model 的分布(前提是這個識別碼在同 Model 底下確實是穩定的規格,不是因 Reference 而異的設計差異——這點需要領域專家確認,不是每個識別碼都能這樣借用)|
|`n` 極少(例如 `Aftermarket`/`Modified` 這類稀有標籤)|**不勉強建立統計分布**,改用專家定義的規則式門檻(例如:某個尺寸超出正品公差範圍 X% 以上,直接判定不屬於 Original,但不細分是哪一種異常類別),把這類標籤標記為「低信心/待人工複核」|

**Step 3 — 把很多個識別碼合併成一個「元件層級」判斷**

一個元件(如 Dial)底下可能有幾十個識別碼,不能只看單一數值,需要合併:

- 對每個識別碼算 Z-score:`z = (觀測值 - mean_Original) / std_Original`
- 元件層級的異常分數:用「有幾個識別碼超過 ±2 個標準差」或「所有 Z-score 的加權平均/最大值」當作聚合指標
- 如果聚合異常分數在門檻內 → 判為 `Original`;超出門檻 → 進一步比對這個聚合特徵組合**離哪個非 Original 類別的分布最近**(用最簡單的「跟每個類別的平均 Z-score 距離比較,選最近的」,這其實就是一個簡化版的最近中心點分類器,不需要複雜的 ML 套件,SQL/Python 都能算)

這個階段的優點是**完全可解釋**——每個判斷都能回答「因為哪幾個識別碼、偏離正品分布多少」,這對鑑定服務的稽核/申訴流程非常重要。

---

## 五、Phase 2:當某個 Reference 的標記資料量夠大時,升級成正式多變量統計分類

當某個 Reference 底下,`Original` 的樣本數夠多(例如 >100)、而且同一元件的多個識別碼彼此有已知的相關性(例如 Dial 文字的 X 位置跟 Y 位置通常會一起偏移)時,單純的獨立 Z-score 加總會低估/高估真實的異常程度。這時候可以升級成:

- **Mahalanobis 距離**:用該 Reference+元件+`Original` 標籤的樣本,算出識別碼之間的共變異數矩陣,再算新掃描的樣本到這個分布中心的 Mahalanobis 距離——這比獨立 Z-score 加總更準,因為它考慮了識別碼之間的相關性。
- 這一步需要的資料量比 Phase 1 高(共變異數矩陣要穩定估計,樣本數建議至少是識別碼數量的好幾倍),所以只在資料量真的夠的 Reference 上啟用,資料量不夠的 Reference 繼續留在 Phase 1 的規則式判斷。

---

## 六、Phase 3(長期):當累積到數萬筆標記掃描,才考慮正式 ML 模型

到了這個規模,才真正有本錢訓練一個「輸入:某元件的所有識別碼數值,輸出:8 類機率分布」的分類模型(例如 Gradient Boosting)。這階段的設計要點:

- 訓練資料來源就是 Iceberg `analysis_facts` + `component_labels` join 出來的表,天然就是訓練集
- 模型**依 Reference 或至少依 Model 分開訓練**(不同款式的正常範圍不同,混在一起訓練會失真)
- 需要額外的模型版本管理(哪個模型是哪個時間點、用哪批資料訓練出來的)、以及定期重訓練排程——這部分目前完全沒有基礎設施,是一筆額外的工程投資,建議等真的有這個資料量規模、且 Phase 1/2 的準確度已經不夠用時才啟動,不要提早做。

---

## 七、跟現有資料庫設計整合的具體修改

|需要新增/修改的東西|說明|
|---|---|
|`component_labels`(新表,本地 SQLite)|見第三節,取代目前斷連的 `output_results/*.json`|
|`component_identifier_map`(新設定檔)|元件 → 識別碼對應關係,版本化管理|
|Iceberg `analysis_facts` ETL 增加 label 維度|`LakeETLJob` 寫入時 join `component_labels`,或另建一張 lake table 讓 Athena 可以 join|
|一張「統計摘要表」(比照上一輪提過的即時比對設計)|定期(例如每晚)用 Athena 算好每個 (Reference, Component, result_key, label) 的 mean/std/percentile,存進一張小表(DynamoDB 或本地 SQLite),供 Authentication Mode **即時**查詢比對用,不要每次掃描都現場跑 Athena|
|`authentication_verdicts`(新表)|存 Authentication Mode 的判斷結果:watchid、component、predicted_label、confidence、用了哪個規則/模型版本、哪些識別碼貢獻了異常分數(供稽核用)——這也順便補上我們前幾輪提過的「本地完全查不到真偽鑑定結果」這個缺口|

---

## 八、規模化考量(數萬支錶)

- 統計摘要計算是**批次/離線**工作,交給 Athena 做,不會因為錶的數量變多就拖慢「單次鑑定」的速度——這正是我們前面設計 Iceberg/Athena 分層架構的初衷。
- Authentication Mode 的即時判斷**只查小型統計摘要表**(不是每次都掃過所有 Original 樣本),所以即使 Reference 底下累積了上萬筆 Original 樣本,查詢速度也不會變慢。
- 稀有標籤類別(`Aftermarket`/`Modified`…)即使到數萬支錶規模,單一 Reference 底下的樣本數可能還是很少——這是資料本質決定的,不是系統設計能解決的,必須接受 Phase 1 的規則式判斷會長期並存,不會完全被統計/ML 取代。

---

## 九、優缺點總評

|面向|優點|缺點/風險|
|---|---|---|
|Phase 0(先接資料模型)|風險最低、必要、不做的話後面全部空談|需要改 UI 存檔邏輯+新增表,有一定工程量|
|Phase 1(規則+Z-score)|可解釋、樣本需求低、可以馬上用現有 Athena 基礎設施做|準確度上限有限,無法捕捉識別碼之間的相關性;稀有類別靠規則,可能不夠精準|
|Phase 2(Mahalanobis 多變量統計)|比 Phase 1 準,仍然可解釋(距離分數)|需要足夠樣本估計共變異數矩陣,只能在資料量夠的 Reference 上啟用,系統要能「因 Reference 而異」動態切換方法,增加複雜度|
|Phase 3(正式 ML)|理論上準確度上限最高,能處理複雜非線性關係|需要額外訓練/版本管理/serving 基礎設施(目前完全沒有);黑箱程度較高,對鑑定服務的可解釋性/稽核要求是風險;過早導入會因樣本不足而過擬合|
|整體分階段策略|每階段都能立刻產生價值,不需要等所有基礎設施到位才能用;跟現有 Athena/Iceberg 架構自然銜接|需要你們自己定義「什麼時候該從 Phase 1 升到 Phase 2」的門檻,以及持續維護 `component_identifier_map` 這個對應關係(未來新增 template/元件時要記得更新)|

**如果只能做一件事,我建議優先做第三節的 Phase 0**——這是目前唯一「完全沒有」而不是「不夠精準」的缺口,而且不解決這個,後面所有統計/ML 設計都無從談起。



#### pdf report目前有的identifier數量
```
先請幫我整理在pdf report目前有的identifier數量, 在watch各個view, strap, box, material, external measurements各有多少個? 那如果以components去分類又各有多少個?

```
我深入追蹤了 PDF 報告產生 identifier 的實際程式碼路徑(`core/report_identifiers.py` 的 `DEFAULT_IDENTIFIER_MAPPINGS` + `config/pdf_config.yaml` 的 `multi_point_inspection` 清單),這兩份是**目前系統裡唯一對「identifier」有明確、可countable定義的地方**。以下是精確盤點結果,並誠實標示哪些是「確定的靜態數字」、哪些是「執行時才會變動的數量」。

---

## 一、先講清楚:「identifier」在這個系統裡怎麼被定義出來

每個 identifier 由兩層決定:

1. **哪些「檢測點」(pdf_inspection_image)會出現在報告裡**——這是 `config/pdf_config.yaml` 的 `multi_point_inspection` 清單,**靜態、可精確計數**。
2. **每個檢測點實際跑了哪個演算法(task),那個演算法輸出定義了幾個 identifier**——這是 `core/report_identifiers.py` 的 `DEFAULT_IDENTIFIER_MAPPINGS`,**大部分是靜態欄位,但有一部分被標記 `"expand": True`,代表這個 spec 會依「實際偵測到幾個東西」動態展開成好幾個 identifier**(例如 OCR 偵測到幾個字元,就展開成幾組)。

所以「一支錶總共有幾個 identifier」**不是一個固定常數**,而是這兩層相乘的結果,而且第二層裡有一大塊是動態的。以下我先給你兩份都精確算過的清單,再說明怎麼組合估算總數。

---

## 二、依 View/Strap/Box/Material/External measurements 分類的檢測點數量

我把 `config/pdf_config.yaml` 裡全部 92 筆 `pdf_inspection_image` 逐一數過:

|View|檢測點數量|備註|
|---|---|---|
|**Front**|18|含 Dial、Bezel、Hands、Crown、Case(1 點)、Crystal|
|**Back**|3|全部屬於 Case|
|**OpenBack**|2|全部屬於 Movement|
|**OpenBackCrown**|8|全部屬於 Movement|
|**StrapRightSide(Strap)**|61|21 個 macropoint + 40 個 micropoint,全部屬於 Strap|
|**Box**|**0**|見下方重要發現|
|**Material**|**0**|見下方重要發現|
|**External measurements**|**0**|見下方重要發現|
|**合計**|**92**||

**重要發現**:我逐行掃過整份 701 行的 `pdf_config.yaml`,`multi_point_inspection` 清單裡**完全沒有任何 Box、Material、External measurements 的檢測點**——目前這份「identifier 文件產生機制」根本沒有涵蓋這三類。這跟我們前幾輪討論過的發現互相呼應:Material/XRF 資料本來就完全不在這套「以 view/point 為單位」的分析結果體系裡(它活在 `watchentry.watchmaterial`/DynamoDB `MAT#` items,是完全獨立的資料結構),External measurements 也是獨立的 `external_measurements_by_internalnum` 資料結構,目前沒有被接進 PDF 的 identifier pipeline。如果你希望這三類未來也能產生可比對的 identifier,需要另外設計(它們目前連「有沒有算出數值」都要另外去查,不是走這條路徑)。

---

## 三、依 Component 分類的檢測點數量

同一份清單,改用 `pdf_inspection_title`(元件)分組:

|Component|檢測點數量|
|---|---|
|**Strap**|61|
|**Movement**|10|
|**Dial**|7|
|**Hands**|5|
|**Case**|4|
|**Bezel**|3|
|**Crystal**|1|
|**Crown**|1|
|**合計**|92|

值得注意的是這裡的元件分類(8 個:Movement、Case、Dial、Hands、Crystal、Bezel、Crown、Strap)跟你之前提到的「Learning Mode」元件標記清單(12 個,多了 Box、Papers/warranty card、Watch case pouch or holder、Original stickers/tags/film wrapping)**不是同一套**——PDF 這邊目前完全沒有 Box 相關的檢測點,跟第二節的發現一致。

---

## 四、每個演算法(task)定義了幾個 identifier——這是「每個點會產生幾個值」的依據

我逐一數過 `DEFAULT_IDENTIFIER_MAPPINGS` 裡全部 17 種演算法的定義:

|演算法(task)|固定數量的 identifier|會動態展開(expand)的 identifier 種類數|
|---|---|---|
|features_letter_service|13|0|
|features_crown_service|19|0|
|features_marker_service|15|0|
|bezel_marker_service|29|2|
|lume_hour_shape_service|1|4|
|lume_hour_texture_service|2|13|
|lume_hand_shape_service|5|0|
|lume_hand_texture_service|12|0|
|diallume_shape_service|1|3|
|diallume_texture_service|12|0|
|**ocr_service**|6|**31**|
|crown2_service|6|3|
|side_crown_service|7|0|
|texture_service|12|0|
|**front_stitch_service**|19|**15**|
|stitched_band2_service|3|2|
|watchband_stitch_service|3|2|
|**合計(17 種演算法)**|**165**|**75(種類數,不是實際展開後的數量)**|

**「expand」是什麼意思**:例如 `ocr_service` 裡的 `char_value`/`char_width_px`/`char_height_px` 這幾十個 spec,每一個都是「**每偵測到一個字元就多產生一筆**」——如果 Dial 文字有 20 個字元,光這一個檢測點,`ocr_service` 這 31 種 expand spec 理論上就可能展開成 20×31 ≈ 600 多個 identifier(實際數字會依每個 spec 能不能取到值而有差異,不會每個都剛好乘滿)。這正是為什麼你之前估計「上萬個 identifier」**是合理的**——`ocr_service`(用在 Dial 文字/刻字類的檢測點)跟 `front_stitch_service`(用在整個 Front 拼接分析,裡面也有大量 expand,例如每個 lume 標記、每個發光點各一組)這兩個演算法本身就是設計成「會隨實際偵測到的細節數量線性增長」的。

---

## 五、把兩層合起來:估算一支錶總 identifier 數量的方法

**精確總數無法只靠讀程式碼算出來**,因為 expand 的實際展開量取決於「這支錶這次到底拍到/偵測到多少細節」(例如 Dial 上刻了幾個字、Bezel 上有幾條刻線、Strap 上有幾個 lume 點),這是**執行期資料**,不是寫死在 config 裡的常數。但我可以給你一個合理的估算框架:

單一檢測點的 identifier 數
  = 這個點跑的演算法之「固定 identifier 數」
  + Σ(每個 expand spec 的「固定 spec 數」 × 「這次實際偵測到幾個實例」)

以 92 個檢測點、其中 Strap 佔 61 個(這些多半是紋理/lume 分析類,固定 12 個左右,expand 較少)、Dial/Front 這 18 個裡有幾個明確是 OCR/刻字/拼接類(expand 很重),粗估一支錶的總 identifier 數量落在**幾百到上萬**這個範圍是合理的,實際數字高度取決於這支錶的刻字複雜度跟 lume 標記密度。**如果你需要精確數字,唯一可靠的方式是實際跑一次完整流程,然後直接數 `report_identifiers_{watchid}.json` 這個檔案裡 `identifiers` 陣列的總長度**——這個檔案已經是系統自動產生的(`_build_identifier_document()` 存在 `DB/report_identifiers_{watchid}.json`),你手上如果有任何一支已經完整掃描過的錶,可以直接打開這個 JSON 檔案數,會比我從程式碼推算準確得多。

---

## 六、額外發現,跟前幾輪的討論直接相關

- 這個 `_build_identifier_document()`/`_evaluate_identifier_goldstand()` 機制,其實就是**你之前想要的「建立標準值、比對新掃描」概念的一個雛形**——但它目前是「單一 golden 標準值 ± 容忍百分比(`metric_tolerance_pct`)」,**不是真正的統計分布**,而且存在 `DB/goldstand_identifiers.json` 這個跟 SQLite 完全無關的獨立檔案裡,**不分 Reference**(全部錶共用同一份 golden standard)。這跟我們上兩輪設計的「依 Reference 建立 distribution」的方向不一樣,而且比較脆弱(只認一個值,不是統計分布),值得你知道這是目前系統裡「最接近但還不是」你要的東西。
- `report_id = "5473454756"` 這個寫死的假值,我在這輪重新讀程式碼時(`App/main.py:38213`)又確認了一次,跟上一輪的發現一致。


```


```



```


```