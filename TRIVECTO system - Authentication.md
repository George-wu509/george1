
|                                                  |     |
| ------------------------------------------------ | --- |
| [[#### 目前的系統設計可以做到單一數值就很重要譬如Dial文字上的Y的高度. ]]     |     |
| [[#### pdf report應該包含上萬(?)個identifier]]          |     |
| [[#### pdf report目前有的identifier數量]]              |     |
| [[#### 深入解釋類似判別watch真偽系統的authentication評分要怎麼設計]] |     |
| [[#### 基於8 classification的new 架構]]               |     |
|                                                  |     |
|                                                  |     |



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


#### 深入解釋類似判別watch真偽系統的authentication評分要怎麼設計
```
請幫我非常深入解釋類似判別watch真偽系統的authentication評分要怎麼設計, 譬如我們有Rolex五個系列的手錶, Series A1, A2, A3. Series B1, B2. 我們每個手表會拍照40張watch各個view的各個component的images, 譬如watch正面錶面的dial 20張, watch背面bridge 3張, 還有其他. 而每個view每個component可以從image擷取很多features. 譬如文字可能每個擷取100個features, 而時針擷取10個features. 所以每個watch scan過後就會得到所有images跟所有擷取的features, 但有可能不同watch有些images沒有或沒有特定features. 那我們陸續的輸入不同watch進行scan, 在scan時user可以針對不同component給予答案: "Original", "Authentic replacements", "Forgery", "Aftermarket", "Modified", "Incorrect Authentic", "Missing", "Not applicable”, 並依這些當作衡量authentication的參考答案, 而且隨著scan watch時可以選擇learning mode就是user可以針對不同component給予答案, 也就是評分邊準隨著輸入scan watch會一直成長, 也可以在某scan watch選擇authentication mode就是看這部分根據目前的評分機制得出的預估, 而scan的也可能有些series已經scan了很多個, 有些series很少,  這樣的system要怎麼建立這樣的評分系統去判別watch真偽系統沒有特定features. 那我們陸續的輸入不同watch進行scan, 在scan時user可以針對不同component給予答案: 

```

![[ChatGPT Image Sep 1, 2026, 12_04_04 PM.png]]


你描述的系統，本質上不是一般的「影像分類器」，而是一個 **階層式、持續學習、可處理缺失證據、可處理不同 Series 資料量不平衡，而且必須能解釋為什麼判真/判假的 Authentication Decision System**。

而且有一個非常重要的設計原則：

> **不要直接把所有 feature 丟進一個模型，最後輸出 0–100 的 Authenticity Score。**

這種方法一開始可能看起來有效，但資料越來越多、Series 越來越多、feature 越來越多之後，很容易失控。

比較適合你的架構是：

**Watch → Series/Reference → Component → Evidence Group → Feature → Component State → Watch-level Authentication**

---

# 一、先把最重要的問題釐清：你的 8 個答案其實不是同一個維度

目前 user 可以標：

|Label|意義|
|---|---|
|Original|原裝、真品、屬於這隻錶|
|Authentic replacement|真 Rolex/OEM replacement|
|Forgery|仿冒|
|Aftermarket|非 OEM 第三方零件|
|Modified|被加工、修改|
|Incorrect Authentic|真 Rolex 零件，但不是這個型號應有的|
|Missing|應該存在但缺少|
|Not applicable|這個 Series 本來就沒有|

表面上看像是 8-class classification。

但其實這 8 個 label 混合了至少 **5 種不同問題**。

我會強烈建議資料庫內部不要直接只保存：

```
dial = Original
```

而是拆成：

|Dimension|問題|
|---|---|
|Presence|有沒有這個 component？|
|Authenticity / Provenance|OEM genuine / counterfeit / aftermarket？|
|Originality|原來就屬於這隻錶，還是後來換的？|
|Compatibility|是不是這個 Series / Reference 正確的零件？|
|Modification|有沒有加工、修改、重新印刷、拋光等？|

例如：

### Original

```
presence = present
oem_authenticity = authentic
originality = original
compatibility = correct
modified = false
```

### Authentic replacement

```
presence = present
oem_authenticity = authentic
originality = replacement
compatibility = correct
modified = false
```

### Incorrect Authentic

```
presence = present
oem_authenticity = authentic
compatibility = incorrect
```

### Forgery

```
presence = present
oem_authenticity = counterfeit
```

### Aftermarket

```
presence = present
oem_authenticity = third_party
```

### Modified

則特別重要：

```
modified = true
```

因為 **Modified 其實不應該和 Original / Authentic replacement / Aftermarket 完全互斥。**

例如一個 genuine Rolex dial 重新印刷：

```
OEM genuine = Yes
Correct for reference = Yes
Original to watch = Yes
Modified = Yes
```

如果硬要把它塞成單一 `Modified` class，你會損失很多資訊。

---

# 二、另外一個非常重要的問題：「Original」不一定能從 image 判斷

這是整個 authentication system 最容易犯的根本錯誤。

假設：

Series A1 的原廠 dial。

另外有一個 **完全相同 reference、完全 genuine 的 Rolex service replacement dial**。

如果兩個：

```
字體一樣
材質一樣
尺寸一樣
logo 一樣
manufacturing process 一樣
```

那只靠 image，可能根本無法知道：

> 這個 dial 是 30 年前跟這隻 watch 一起出廠，還是 5 年前 Rolex service center 換上去的。

也就是：

```
Authentic OEM
```

可能是 image-observable。

但是：

```
Original-to-this-watch
```

可能不是 image-observable。

這兩件事一定要分開。

否則 ML 最後會開始學一些假的 correlation，例如：

```
比較舊的錶面 → Original

比較乾淨 → Replacement
```

然後碰到保存良好的 original watch 就判錯。

因此推薦架構是：

```
Visual Authentication
+
Reference Compatibility
+
Historical / Provenance Evidence
=
Final Component Assessment
```

---

# 三、整個系統可以拆成 7 層

我會把 Moonlight 類似的 Authentication pipeline 設計成：

```
Layer 1
Watch Reference / Series Knowledge

        ↓

Layer 2
Image & Feature Extraction

        ↓

Layer 3
Evidence Group Models

        ↓

Layer 4
Component Authentication Model

        ↓

Layer 5
Component State / Category

        ↓

Layer 6
Watch-level Authentication

        ↓

Layer 7
Business / Authentication Policy
```

這樣後面不論從 5 個 Series 增加到 50 個 Series，都比較容易維護。

---

# 四、Layer 1：先建立 Series / Reference Knowledge Base

假設目前：

```
Family A
    A1
    A2
    A3

Family B
    B1
    B2
```

不要讓模型自己從頭重新學：

> A1 有什麼 component。

應該建立一個 reference database。

例如：

|Component|A1|A2|A3|B1|B2|
|---|---|---|---|---|---|
|Dial|Required|Required|Required|Required|Required|
|Hour Hand|Required|Required|Required|Required|Required|
|Bridge A|Required|Required|N/A|N/A|N/A|
|Bridge B|N/A|N/A|Required|Required|Required|
|Crown|Required|Required|Required|Required|Required|

這一層可以直接處理：

```
Not applicable
```

而不是交給 ML 猜。

例如：

```
Series A3
Component Bridge-A
```

reference database 已經知道：

```
not_applicable = true
```

就不應該讓 authentication model 嘗試判定。

---

# 五、一定要分清楚三種「Missing」

你前面特別提到：

> 有些 watch 沒有某些 image，或某些 feature 沒有。

這非常重要。

因為以下三種情況完全不同。

|Missing 類型|例子|Authentication 意義|
|---|---|---|
|Image missing|Camera 沒拍成功|通常不是 counterfeit evidence|
|Feature missing|OCR 沒讀到某個字|可能只是 extraction failure|
|Physical component missing|Rolex 應該有 crown，但真的不存在|Authentication evidence|

所以資料庫不能只有：

```
missing = true
```

至少要有：

```
capture_missing
feature_missing
component_missing
not_applicable
```

否則模型會開始把「camera 拍攝失敗」當成假的特徵。

---

# 六、40 張 images 不應該直接視為 40 個獨立 vote

例如你說：

```
Watch
    ↓
40 images

Dial
    ↓
20 images

Bridge
    ↓
3 images
```

其中 Dial 20 張 image 可能包含：

```
Front
Logo
ROLEX text
minute markers
Swiss Made
date window
hour hand area
...
```

這些影像並不是 20 個完全獨立的 Authentication votes。

因為它們高度相關。

同樣的：

> text component 有 100 features，而 hour hand 只有 10 features。

絕對不能做：

```
100 text features × weight
+
10 hand features × weight
```

否則 text 會因為 feature 數量比較多，天然得到 10 倍影響力。

---

# 七、應該加入 Evidence Group

例如 Dial 的 100 個 feature，我會先分成：

|Evidence Group|Features|
|---|---|
|Text content|OCR characters、word sequence|
|Typography|字體、stroke width、character ratio|
|Geometry|字距、baseline、position|
|Logo morphology|crown shape、spacing|
|Printing quality|edge、ink texture|
|Color/material|RGB、spectral、surface|
|Manufacturing geometry|index spacing、marker dimension|
|Micro texture|surface / print microstructure|

所以架構變成：

```
100 raw features

↓ group

8 evidence groups

↓ model

Dial Component Model
```

這樣「文字有 100 features」不會因為 feature 多就壓過 hour hand。

---

# 八、每個 feature 本身不要直接變 Authentication Score

例如有一個 feature：

```
ROLEX "R" width ratio
```

不要做：

```
符合 = +2
不符合 = -2
```

比較好的方式是學習：

P(x∣Authentic)P(x|Authentic)

和：

P(x∣Forgery)P(x|Forgery)

甚至：

P(x∣Series=A1,Authentic)P(x|Series=A1, Authentic)

例如 authentic A1 的某尺寸：

```
mean = 0.582
std = 0.012
```

Forgery：

```
mean = 0.631
std = 0.041
```

新 watch：

```
x = 0.587
```

那這個 feature 就提供：

LR=P(x∣Authentic)P(x∣Forgery)LR = \frac {P(x|Authentic)} {P(x|Forgery)}

也就是 **Likelihood Ratio**。

更方便使用 log：

LLR=log⁡P(x∣Authentic)P(x∣Forgery)LLR = \log \frac {P(x|Authentic)} {P(x|Forgery)}

如果：

```
LLR > 0
```

代表支持 genuine。

如果：

```
LLR < 0
```

支持 forgery。

---

# 九、但不要把 100 個 LLR 直接相加

因為很多 feature 是 correlated。

例如：

```
letter height
letter width
aspect ratio
area
bounding box
```

其實來自同一個字。

如果全部算成獨立證據：

```
Evidence = LLR1 + LLR2 + LLR3 + ...
```

會嚴重 overconfidence。

所以應該是：

```
raw features
     ↓
evidence-group model
     ↓
group score
```

再聚合。

---

# 十、實際上可以讓每個 Evidence Group 輸出 probability

例如 Series A1 的 Dial：

|Evidence|Authentic probability|Quality|
|---|---|---|
|Typography|0.96|0.98|
|Text layout|0.94|0.95|
|Logo geometry|0.91|0.90|
|Printing|0.87|0.92|
|Color|0.75|0.65|
|Microtexture|0.53|0.30|

最後一項：

```
Microtexture = 0.53
```

不能理解成「有一半是假」。

因為它的 quality：

```
0.30
```

代表這張照片可能失焦、反光或者 coverage 不足。

所以 authentication evidence 還需要：

EvidenceStrength=ModelEvidence×Quality×ReliabilityEvidence Strength = Model Evidence \times Quality \times Reliability

---

# 十一、因此每個 evidence 都應該有 Quality Score

我建議每次 feature extraction 同時產生：

```
feature_value
feature_quality
feature_validity
feature_source
extractor_version
```

例如 OCR：

```
text = ROLEX
confidence = 0.997
focus_quality = 0.94
exposure_quality = 0.98
occlusion = 0.02
```

另外一張：

```
text = RO?EX
confidence = 0.51
focus_quality = 0.28
```

第二張不是：

> 比較像 counterfeit。

而是：

> evidence 不夠可靠。

這兩個概念完全不同。

---

# 十二、Component 才是最重要的 Authentication Unit

假設 Dial 有：

```
20 images
100+ features
8 evidence groups
```

最後都應該集中到：

```
Dial Component State
```

例如模型輸出：

|Component inference|Probability|
|---|---|
|OEM Genuine|0.94|
|Counterfeit|0.03|
|Aftermarket|0.03|

再另外輸出：

|Compatibility|Probability|
|---|---|
|Correct for A1|0.97|
|Incorrect Authentic|0.03|

另外：

|Modification|Probability|
|---|---|
|Unmodified|0.92|
|Modified|0.08|

另外：

```
Originality:
Original to this watch      0.55
Replacement                 0.25
Cannot determine            0.20
```

這會比單純：

```
Original = 83%
```

有價值得多。

---

# 十三、然後才轉換成 user 現在看到的 category

可以建立一個 mapping/policy engine。

例如：

```
OEM Genuine > threshold
AND
Correct Reference
AND
Original evidence sufficient
AND
Not Modified
```

→

```
Original
```

如果：

```
OEM Genuine
AND
Correct Reference
AND
Replacement evidence
```

→

```
Authentic replacement
```

如果：

```
OEM Genuine
AND
Incorrect Reference
```

→

```
Incorrect Authentic
```

這代表：

> ML model 跟 business label 是分開的。

這會非常重要。

---

# 十四、不要只有一個「Authentication Score」

我會推薦 UI 至少保存四個 score：

|Score|問題|
|---|---|
|Authenticity Score|這是不是 genuine OEM？|
|Originality Score|是不是原裝於這隻 watch？|
|Compatibility Score|是否符合這個 Series/reference？|
|Modification Score|是否有修改？|

最後才可以有：

```
Overall Authentication Score
```

但是它的定義必須非常清楚。

例如定義：

> Probability that all critical components are genuine OEM and compatible with the selected watch reference.

這時：

```
Authentication Score = 94%
```

才有統計上的意義。

不然「94% 真」其實不知道代表什麼。

---

# 十五、不同 customer 甚至可以使用不同 Authentication Policy

這在 Rolex authentication 特別有用。

例如三種 policy。

|Profile|Authentic replacement|Modified|Aftermarket|
|---|---|---|---|
|Authenticity|可以接受|視情況|不接受|
|Collector Originality|扣很多|不接受|不接受|
|Service Validation|可以接受|部分接受|不接受|

例如一隻錶：

```
case genuine
dial genuine replacement
hands genuine
movement genuine
bracelet genuine
```

對：

```
Is this a genuine Rolex?
```

可能非常高：

```
98%
```

可是：

```
Is this completely original?
```

可能只有：

```
54%
```

這比硬給它：

```
Authentication = 76%
```

合理很多。

---

# 十六、Watch-level score 不應該單純平均所有 component

這是第二個最容易出問題的地方。

例如：

```
Dial       99%
Hands      98%
Case       99%
Bridge     10%
Bracelet   98%
```

平均：

(99+98+99+10+98)/5=80.8(99+98+99+10+98)/5=80.8

如果顯示：

```
Watch Authenticity = 81%
```

會非常誤導。

因為如果 Bridge 是核心 movement component，而且 90% probability 是 counterfeit：

> 這可能是一個重大 failure。

因此 watch-level 應該分成：

**Probabilistic scoring + Policy rules**

兩層。

---

# 十七、Component 可以有 Criticality

例如：

```
Dial
Movement
Case
Serial engraving
Hands
Bracelet
Clasp
...
```

每個 component 可以有：

wcw_c

但這個 weight 不應該只代表：

> feature 很多。

而應該代表：

> 對整隻 watch authentication 的重要程度。

而且可以依 Series 改變：

wSeries,Componentw_{Series,Component}

例如：

```
A1 / Dial
```

的重要性可能和：

```
B2 / Dial
```

不同。

---

# 十八、甚至可以使用 Risk Aggregation，而不是 Average

假設 component cc 的 unacceptable probability 是：

rcr_c

例如：

```
Forgery
Aftermarket
Incorrect Authentic
```

根據 policy 組成：

rc=P(Forgery)+P(Aftermarket)+...r_c = P(Forgery) + P(Aftermarket) + ...

可以使用類似 Noisy-OR：

Rwatch=1−∏c(1−κcrc)R_{watch} = 1 - \prod_c (1-\kappa_c r_c)

其中：

κc\kappa_c

代表 component criticality。

這會讓：

> 一個非常嚴重的 counterfeit component

不容易被其他 30 個正常 component 平均掉。

---

# 十九、資料夠多以後，可以讓 Watch-level model 自己學 aggregation

例如輸入：

```
Dial:
 P_authentic
 P_aftermarket
 P_forgery
 quality
 coverage

Movement:
 ...

Case:
 ...

Bracelet:
 ...
```

然後 training 一個：

```
Watch Authentication Meta Model
```

模型可能是：

```
logistic regression
LightGBM
CatBoost
small neural network
```

來預測：

P(WatchPolicyPass)P(WatchPolicyPass)

但是一開始資料不夠時，我會推薦：

```
Component models
+
explicit policy engine
```

而不是直接訓練 end-to-end Watch classifier。

因為比較可解釋。

---

# 二十、Series A1/A2/A3/B1/B2 資料量不平衡，是你的另一個核心問題

假設：

```
A1 = 1,200 watches
A2 = 800
A3 = 350

B1 = 75
B2 = 18
```

絕對不要直接建立：

```
Model_A1
Model_A2
Model_A3
Model_B1
Model_B2
```

因為 B2 很容易 overfit。

我推薦：

```
Global model
      ↓
Family model
      ↓
Series specialization
```

例如：

```
Global Dial Model
      ↓
Family A Dial
      ↓
A1 / A2 / A3

Global Dial Model
      ↓
Family B Dial
      ↓
B1 / B2
```

這叫：

**Hierarchical / Partial Pooling**

---

# 二十一、少資料的 Series 會自動借用其他 Series 的 knowledge

概念上可以寫成：

θseries=λθseries−local+(1−λ)θfamily/global\theta_{series} = \lambda \theta_{series-local} + (1-\lambda) \theta_{family/global}

其中：

λ=NseriesNseries+K\lambda = \frac{N_{series}} {N_{series}+K}

假設：

```
K = 100
```

Series B2：

```
N = 10
```

那：

λ=10/(10+100)=0.091\lambda = 10/(10+100) =0.091

所以模型主要相信：

```
Family B / Global knowledge
```

而不是相信只有 10 隻 watch 的 B2 local model。

等到：

```
B2 N = 500
```

λ=500/600=0.833\lambda = 500/600 = 0.833

這時才大量相信 B2 自己的資料。

這正好符合你所說：

> 有些 Series 會持續 scan 很多；有些 Series 很少。

---

# 二十二、Series 可以當 feature，但不能只是普通 categorical feature

例如：

```
series = A1
```

模型需要理解：

```
A1 / A2 / A3
```

有 family relationship。

可以建立：

```
brand
family
reference
production_period
component_variant
```

例如：

```
Rolex
 └─ Family A
      ├─ A1
      ├─ A2
      └─ A3
```

這樣 A3 少資料時還可以借用 A-family。

---

# 二十三、甚至 Component 也可以共享 knowledge

比如：

```
A1 dial
A2 dial
A3 dial
```

很多 authentication pattern 可能相似：

```
printing edge
metal finishing
logo morphology
manufacturing tolerance
```

可以共享 base model。

例如：

```
Dial Global Encoder
+
Series-specific adapter
```

而不是每個 series 重新 training 一個完全獨立 model。

---

# 二十四、你的 Learning Mode 不應該等於「每掃一隻就立刻 retrain」

這點很重要。

你希望：

> 越 scan 越準。

這是正確的。

但是最好做：

```
Continuously growing dataset
```

而不是：

```
Continuous uncontrolled model update
```

也就是 Learning Mode scan 後：

```
Human label
↓
加入 training repository
↓
等待 validation
↓
candidate model
↓
evaluation
↓
calibration
↓
promote to production
```

而不是：

```
User label
↓
立刻改 production model
```

---

# 二十五、否則很容易發生 Model Poisoning / Catastrophic Forgetting

假設某 engineer 誤標：

```
Forgery → Original
```

如果每 scan 一次就 online update：

模型可能立刻受影響。

或者連續 scan：

```
20 個 A1 Original
```

模型會開始偏向：

```
Everything is Original
```

所以建議：

> Dataset 可以 continuous learning；Production model 則 versioned learning。

---

# 二十六、Learning Mode 最適合的流程

你目前有：

```
Learning Mode
Authentication Mode
```

這個方向是非常好的。

推薦架構是：

|Learning Mode|Authentication Mode|
|---|---|
|收集新資料|不更改 training data|
|Human label|Model prediction|
|可以修正 label|顯示 confidence|
|加入 training dataset|使用 frozen model|
|支援 expert comments|顯示 evidence|
|加入 future training|記錄 model version|

Authentication Mode 的 scan 絕對不要因為模型說：

```
Original 98%
```

就自動把它加入：

```
Original training data
```

這叫 self-labeling feedback loop。

非常危險。

---

# 二十七、Learning Mode 甚至可以加入 Active Learning

不必讓 expert 每次標完 40 個 component。

模型可以自己找到：

```
最值得 human review 的 case
```

例如：

Entropy=−∑kpklog⁡pkEntropy = -\sum_k p_k\log p_k

如果：

```
Original              0.49
Authentic replacement 0.43
Forgery               0.02
...
```

entropy 很高。

代表模型不確定。

就標記：

```
Human Review Recommended
```

另外以下也值得優先 learning：

```
rare Series
rare labels
OOD cases
new forgery styles
model disagreement
low-confidence components
```

這會讓 expert labeling 效率提升非常多。

---

# 二十八、Authentication 系統一定要有「不知道」的能力

高品質 Authentication system 最大的差別之一不是：

> 每一次都敢給答案。

而是知道：

> 哪一次不能可靠回答。

例如：

```
A1 Dial

Forgery       0.36
Aftermarket   0.28
Original      0.22
Replacement   0.14
```

不要硬 output：

```
Forgery
```

最好：

```
Status:
INCONCLUSIVE

Reason:
Insufficient discriminative evidence.
```

---

# 二十九、可以設三區 threshold

例如概念上：

```
PASS
REVIEW
FAIL
```

而不是：

```
Authentic / Fake
```

例如：

P(unacceptable)<TpassP(unacceptable) < T_{pass}

→ PASS

Tpass≤P(unacceptable)<TfailT_{pass} \le P(unacceptable) < T_{fail}

→ REVIEW

P(unacceptable)≥TfailP(unacceptable) \ge T_{fail}

→ FAIL

threshold 不要一開始拍腦袋決定。

而要透過 validation data 決定。

---

# 三十、Authentication 最重要的 metrics 不只是 Accuracy

假設：

```
1000 Original
20 Forgery
```

模型全部猜 Original：

```
Accuracy = 98%
```

但這個 Authentication system 完全沒有用。

真正需要看：

|Metric|重要性|
|---|---|
|Forgery Recall|假錶抓到多少|
|False Accept Rate|假貨被判真多少|
|False Reject Rate|真錶被誤判多少|
|Precision|判 Forgery 時有多可靠|
|Calibration|90% 是否真的約 90% 正確|
|Review rate|有多少 case 要人工|
|Coverage|有多少 case 能自動判|

Authentication system 很適合看：

```
False Accept Rate
```

因為：

> 把 fake 判成 authentic

通常是最大的商業風險。

---

# 三十一、而且 Train/Test split 必須以「Watch」為單位

這對你現在 40 images/watch 特別重要。

千萬不能：

```
Watch #001 image 1 → train
Watch #001 image 2 → train
Watch #001 image 3 → test
```

這是嚴重 data leakage。

因為同一隻錶的：

```
lighting
texture
aging
manufacturing characteristics
```

高度相關。

模型看起來可能：

```
99.8% accuracy
```

但實際新 watch 可能只有：

```
75%
```

正確的方法：

```
Watch #001:
全部 train

Watch #002:
全部 train

Watch #003:
全部 test
```

也就是：

**Group split by physical watch ID.**

如果同一隻 watch scan 過 20 次：

> 這 20 次也全部必須在同一個 split。

---

# 三十二、最好再做 Time-based validation

例如：

```
Training:
2026-01 ~ 2027-06

Testing:
2027-07 ~ 2027-12
```

這比較接近真實 production：

> 用以前的 counterfeit pattern 預測未來的新 counterfeit。

因為 forgery 本身會 evolution。

---

# 三十三、Counterfeit Detection 還有一個很大的特殊問題：Open-set Recognition

假設 training data 看過：

```
Forgery Type A
Forgery Type B
Forgery Type C
```

未來突然出現：

```
Forgery Type Z
```

普通 multi-class model 很可能非常有自信地說：

```
Original 97%
```

因為它不知道：

> 世界上還有沒看過的 counterfeit。

所以 Authentication system 最好同時有：

```
Supervised classifier
+
Authentic anomaly detector
```

---

# 三十四、Authentic reference distribution 非常重要

例如對 Series A1 Dial 建立 genuine distribution：

Xauth∼DA1,DialX_{auth} \sim D_{A1,Dial}

新 component feature vector：

xx

計算：

distance(x,Dauth)distance(x,D_{auth})

例如：

```
Mahalanobis distance
embedding distance
density estimate
one-class model
```

如果：

```
classifier:
Original = 91%
```

但是：

```
Authentic prototype distance:
extremely abnormal
```

就應該：

```
OOD WARNING
Manual Review
```

而不是直接 Pass。

---

# 三十五、因此我推薦每個 Component 最後至少有三個 AI subsystem

概念上：

```
A. Closed-set classifier
Original / Authentic Replacement /
Forgery / Aftermarket / ...

B. Authentic similarity / anomaly model
Is this inside known genuine distribution?

C. Reference compatibility engine
Does this component belong to A1?
```

最後：

```
Decision Fusion
```

這會比單一 classifier 強很多。

---

# 三十六、Reference Compatibility 很多時候甚至不需要 ML

例如：

```
movement bridge part X
```

模型可以先識別：

```
part_family = X
```

Reference database：

```
A1 → X allowed
A2 → X allowed
A3 → Y only
```

如果：

```
A3 watch
+
genuine X part
```

直接得到：

```
Incorrect Authentic
```

所以：

> AI 負責「這是什麼」，Knowledge Base 負責「它應不應該在這裡」。

這比逼 AI 自己記所有 compatibility 關係更可靠。

---

# 三十七、再來談你說的「100 features vs 10 features」

假設：

### Dial text

```
100 features
```

### Hour hand

```
10 features
```

真正的架構不應該：

Score=∑1100TextFeature+∑110HandFeatureScore = \sum_{1}^{100} TextFeature + \sum_{1}^{10} HandFeature

而是：

```
100 text features
      ↓
Text Evidence Model
      ↓
Text Score

10 hand features
      ↓
Hand Evidence Model
      ↓
Hand Score
```

然後：

```
Text Score
Hand Score
Logo Score
Material Score
...
```

再送 Component / Watch Model。

---

# 三十八、模型選擇上，我反而不建議第一版就用非常複雜 Deep Learning

如果你的 feature 已經是 structured feature，例如：

```
text width
height
distance
color
angle
area
roundness
OCR confidence
material concentration
position
texture score
```

第一版通常非常適合：

```
CatBoost
LightGBM
XGBoost
Logistic Regression
```

尤其 CatBoost / LightGBM：

對：

```
missing values
nonlinear relationships
feature interactions
limited data
```

都很好。

而且比大型 neural network 容易 explain。

---

# 三十九、Image Deep Learning 可以當其中一種 Evidence

例如：

```
Dial Image
↓
ViT/CNN encoder
↓
embedding 512D
```

但是不要把它直接當最後 Authentication。

可以變成：

```
Visual similarity score
```

然後跟：

```
OCR
Geometry
Material
Reference DB
```

一起融合。

最後可能是：

```
Structured Evidence
+
Image Embedding
+
Knowledge-base checks
```

---

# 四十、Feature importance 應該讓資料學，而不是完全手工指定

一開始 expert 可以設定：

```
Dial logo important
Text spacing important
Color medium
```

但是隨著 data 增加，可以讓 model 學：

wfeaturew_{feature}

或者：

wevidence−groupw_{evidence-group}

之後用：

```
SHAP
feature attribution
```

解釋。

例如 Authentication report：

```
Dial — Suspicious

Main evidence:

ROLEX letter spacing:
strong disagreement with A1 reference

Crown logo width ratio:
moderate disagreement

Minute marker geometry:
consistent with A1

Color:
insufficient evidence
```

這種 report 對 authentication expert 會比：

```
Dial score = 72.46
```

有用得多。

---

# 四十一、Authentication Confidence 跟 Authentication Score 必須分開

例如：

### Case A

```
Authenticity probability = 95%
Evidence confidence = 98%
```

代表：

> 很像 genuine，而且 evidence 很完整。

### Case B

```
Authenticity probability = 95%
Evidence confidence = 31%
```

這代表：

> 模型目前偏向 genuine，但 evidence 很少。

不能把兩者當一樣。

我會顯示：

```
Authentication:
95%

Confidence:
Low

Evidence Coverage:
38%
```

---

# 四十二、Coverage Score 對你的系統會非常重要

因為每個 watch 不一定有全部 images/features。

可以定義：

Coveragec=∑gimportanceg×availableg∑gimportancegCoverage_c = \frac {\sum_g importance_g \times available_g} {\sum_g importance_g}

例如 Dial 原本需要：

```
Typography  20%
Geometry    20%
Logo        25%
Material    20%
Texture     15%
```

但這次：

```
Material image missing
Texture unusable
```

那 coverage：

```
65%
```

模型可能仍然說：

```
Authentic = 96%
```

但是 UI：

```
Authentication: 96%
Evidence Coverage: 65%
Confidence: Medium
```

---

# 四十三、這也解決「不同 Watch feature 數量不同」

不要求：

```
每一個 watch feature vector 完全一模一樣
```

而是建立：

```
Feature
↓
Evidence group
↓
Component
```

如果某 evidence group 缺少：

```
mask + quality + coverage
```

而不是：

```
填 0
```

因為：

```
0
```

本身通常是一個合法數值。

---

# 四十四、資料庫最好保存「Feature Schema Version」

因為未來你的 feature 一定會變。

例如：

2026：

```
text features = 80
```

2027：

```
text features = 100
```

2028：

```
加入 texture features
```

所以每個結果都要保存：

```
extractor_version
feature_schema_version
component_model_version
calibration_version
policy_version
```

例如：

```
feature_schema = 4.2
dial_model = 12.3
authentication_policy = 5.1
```

這樣兩年前的 scan 才能 reproduce。

---

# 四十五、Human Ground Truth 本身也需要版本和可信度

不要只保存：

```
label = Original
```

最好保存：

```
label
labeler
timestamp
label_confidence
reason
supporting evidence
review_status
```

例如：

```
Expert A:
Original
confidence = High

Expert B:
Authentic Replacement
confidence = Medium
```

這應該進：

```
needs adjudication
```

而不是隨便相信最新的一個 label。

---

# 四十六、不同 expert 的 label reliability 甚至可以估計

長期資料多了之後可以知道：

```
Expert A
movement accuracy excellent

Expert B
dial typography excellent

Expert C
bracelet expertise excellent
```

不一定要拿來「排名員工」，而是可以用於 ground-truth quality control。

---

# 四十七、Training Dataset 的單位最好是 Component，不是 Feature

例如：

```
Watch W000123

Series A1

Component Dial

Images:
 IMG001
 IMG002
 ...
 IMG020

Evidence:
 Typography
 Geometry
 Color
 Logo
 ...

Ground Truth:
 OEM = Authentic
 Compatibility = Correct
 Originality = Replacement
 Modified = False
```

這是一個 training example。

而不是：

```
feature #1 = Original
feature #2 = Original
...
```

因為 ground truth 本來就是 component level。

---

# 四十八、你現在這個系統最適合 Multiple Evidence / Multiple Instance 的概念

例如：

```
Dial
 ├─ image1
 ├─ image2
 ├─ ...
 ├─ image20
 └─ features...
```

這些全部共同支持一個：

```
Dial Ground Truth
```

這就是類似：

**Multiple Instance Learning / Evidence Fusion**

你不必假設：

```
每一張 dial image 都能單獨回答 Original/Forgery。
```

有些 image 只提供：

```
logo evidence
```

有些只提供：

```
text evidence
```

最後一起決定。

---

# 四十九、我會把整個 inference pipeline 做成這樣

這是我最推薦的實際架構：

```
Watch Scan
│
├── Series / Reference = A1
│
├── Dial
│   ├── 20 images
│   ├── Image QC
│   ├── OCR evidence
│   ├── Typography evidence
│   ├── Geometry evidence
│   ├── Visual embedding
│   ├── Material evidence
│   │
│   ├── Authenticity model
│   ├── Anomaly model
│   ├── Compatibility check
│   │
│   └── Component State
│
├── Bridge
│   ├── 3 images
│   └── ...
│
├── Hands
│   └── ...
│
├── Case
│   └── ...
│
└── Bracelet
    └── ...

         ↓

Component Probabilities

         ↓

Watch-level Meta Model

         +

Authentication Policy

         ↓

PASS / REVIEW / FAIL
+
Scores
+
Reasons
+
Evidence Coverage
```

---

# 五十、實際 Component prediction 可以長這樣

假設 A1 Dial：

```
Presence:
Present                    99.9%

OEM Authenticity:
Genuine                    95.1%
Counterfeit                 2.7%
Aftermarket                 2.2%

Compatibility:
Correct A1                 97.4%
Incorrect authentic         2.6%

Modification:
Unmodified                 91.8%
Modified                    8.2%

Originality:
Original                   55.1%
Replacement                23.8%
Indeterminate              21.1%

Evidence coverage:
87%

OOD:
Low
```

最後系統不要急著說：

```
Original
```

可以顯示：

```
Assessment:
Authentic OEM component

Compatibility:
Correct for A1

Originality:
Not conclusively determined

Overall confidence:
High
```

這其實比錯誤地給：

```
Original = 91%
```

專業很多。

---

# 五十一、再舉一個 Forgery 的例子

假設 Dial：

```
Typography authenticity        0.18
Logo authenticity              0.09
Printing authenticity          0.21
Geometry authenticity          0.72
Material authenticity          0.61
```

模型：

```
Forgery             88%
Original             5%
Aftermarket           4%
Replacement           3%
```

Authentic anomaly detector：

```
A1 genuine distance:
99.2 percentile anomaly
```

兩個 subsystem 都支持：

```
Suspicious
```

因此：

```
Component Status:
FAIL

Reason:
Strong typography/logo inconsistency
and out-of-distribution against
A1 authentic dial references.
```

這就是非常強的 authentication evidence。

---

# 五十二、但是如果兩個模型互相衝突

例如：

```
Classifier:
Original = 94%
```

但是：

```
Authentic anomaly:
extremely abnormal
```

不要 Pass。

應該：

```
REVIEW
```

因為這很可能是：

```
new counterfeit style
new production variant
bad feature extraction
wrong Series selection
```

其中之一。

---

# 五十三、Wrong Series selection 也一定要防止

假設實際 watch：

```
A2
```

user 選成：

```
A1
```

模型可能看到：

```
genuine A2 component
```

然後全部判：

```
Incorrect Authentic
```

所以最好：

```
User-selected Series
+
AI Series consistency score
```

例如：

```
Selected reference:
A1

Visual/reference consistency:
A1  12%
A2  84%
A3   4%
```

系統可以警告：

```
Possible incorrect reference selection.
```

---

# 五十四、Learning Mode 也不要平均收資料，要刻意補資料洞

例如現在：

|Series|Original|Replacement|Forgery|Aftermarket|
|---|---|---|---|---|
|A1|700|120|90|45|
|A2|450|80|35|20|
|A3|300|60|25|12|
|B1|55|5|3|2|
|B2|12|0|1|0|

這時下一批 learning priority 不應該還是 A1 Original。

而應該非常希望收：

```
B2
B1 forgery
B2 replacement
B2 aftermarket
```

這就是：

**dataset coverage management。**

---

# 五十五、Class imbalance 不能只用 Accuracy 解決

Training 時可以使用：

```
class weighting
balanced sampling
focal loss
hierarchical sharing
```

但最重要的是：

> 真的收集 rare negative examples。

尤其 Authentication：

```
Forgery
Aftermarket
Incorrect Authentic
Modified
```

通常比 Original 少。

可是恰恰是最重要的。

---

# 五十六、每個 Series / Component 要維護 Data Sufficiency

例如 UI 可以顯示：

```
A1 Dial
Model maturity: High

A2 Dial
Model maturity: High

B1 Dial
Model maturity: Medium

B2 Dial
Model maturity: Low
```

不要讓 user 看到：

```
B2 Authenticity = 98%
```

卻不知道 training 只有 8 隻。

---

# 五十七、Confidence 應該包含 Model Maturity

概念可以：

Confidence=f(model calibration,evidence coverage,image quality,training support,OOD)Confidence = f( model\ calibration, evidence\ coverage, image\ quality, training\ support, OOD )

例如：

```
Prediction = Authentic 96%
```

但是：

```
B2 training samples = 9
```

則：

```
Confidence = Low
```

---

# 五十八、Model Probability 一定要 Calibration

很多 ML model 的：

```
0.95
```

並不真的代表：

> 100 個 0.95 prediction 中有 95 個正確。

需要另外做：

```
Platt scaling
Isotonic regression
Temperature scaling
```

等 calibration。

如果校準後：

```
90% prediction
```

真的大約：

```
90/100 correct
```

你的 Authentication Score 才能叫 probability。

---

# 五十九、最後的 Watch-level report 我會推薦長這樣

例如：

## Overall

```
Reference: Series A1

Authentication Status:
REVIEW

OEM Authenticity:
94%

Originality:
72%

Reference Compatibility:
98%

Modification Risk:
18%

Evidence Coverage:
91%

Overall Confidence:
High
```

Component table：

|Component|State|Auth|Originality|Confidence|
|---|---|---|---|---|
|Dial|Authentic replacement suspected|97%|42%|High|
|Hands|Original|98%|91%|High|
|Case|Original|99%|88%|High|
|Bridge|Original|96%|82%|Medium|
|Bracelet|Modified|94% genuine|76%|High|

Reason：

```
Manual review required because the dial
appears OEM-authentic and reference-compatible,
but replacement probability exceeds the
collector-originality threshold.
```

這樣才是一個真正的 Authentication system。

---

# 六十、我尤其推薦把「ML」和「Authentication Policy」完全分開

這個觀念非常重要。

ML 只回答：

```
我看到的 evidence 表示什麼？
```

Policy 回答：

```
公司認為這種情況算 Authentic 嗎？
```

例如 ML：

```
OEM Genuine = 99%
Authentic Replacement = 96%
```

Authentication Policy A：

```
Replacement allowed
→ PASS
```

Collector Policy：

```
Original-only
→ REVIEW / FAIL
```

同一個 AI evidence，不需要重新 training model。

---

# 六十一、這也會讓未來產品容易擴充

以後甚至可以提供：

```
Authentication Mode

Originality Mode

Collector Mode

Service Verification Mode

Component Inspection Mode
```

全部使用同一套 feature/model。

只是 policy 不同。

---

# 六十二、如果是我要替你現在這個系統定第一版，我不會直接做一個大 AI

我會先建立下面這個最小但正確的 architecture：

```
Series Reference DB
        ↓
Component Definitions
        ↓
Feature Registry
        ↓
Evidence Groups
        ↓
Component ML Models
        ↓
Component Multi-axis State
        ↓
Authentication Policy
        ↓
Watch Result
```

Machine Learning 第一階段則：

```
structured feature:
CatBoost / LightGBM

visual:
CNN/ViT embedding

unknown detection:
authentic prototype / anomaly model

probability:
calibration layer
```

這會比一開始做：

```
40 images + 5000 features
        ↓
Huge neural network
        ↓
Authenticity = 87%
```

可靠、可解釋而且更容易 debug。

---

# 六十三、最終我認為你這個系統真正應該學的是「Evidence」，而不是直接學「答案」

這是整個架構裡最重要的一句。

不要讓：

```
feature → Final Original/Fake
```

直接一步完成。

最好是：

```
Feature
↓
Evidence

Evidence
↓
Component state

Component state
↓
Authentication judgment
```

因為 Rolex authentication 本身就是：

> 多個獨立與半獨立證據累積後做出的專家判斷。

AI 最適合模擬的也正是這個流程。

---

# 我會建議你的實際最終資料結構

核心可以接近：

```
Watch
 ├── watch_id
 ├── selected_series
 ├── predicted_series
 │
 ├── Component
 │    ├── component_id
 │    ├── expected/not_applicable
 │    │
 │    ├── Images
 │    │    ├── image
 │    │    ├── quality
 │    │    └── capture_status
 │    │
 │    ├── Features
 │    │    ├── feature_name
 │    │    ├── value
 │    │    ├── quality
 │    │    ├── valid
 │    │    └── extractor_version
 │    │
 │    ├── Evidence Groups
 │    │
 │    ├── Model Outputs
 │    │    ├── authenticity
 │    │    ├── originality
 │    │    ├── compatibility
 │    │    ├── modification
 │    │    ├── anomaly
 │    │    ├── confidence
 │    │    └── coverage
 │    │
 │    └── Human Ground Truth
 │
 ├── Watch-level Evidence
 │
 ├── Authentication Policy
 │
 └── Final Result
```

這套結構可以讓你從現在的 A1/A2/A3/B1/B2，之後擴展到幾十個 Rolex reference，而不用推翻整套 Authentication 系統。

---

## 最後濃縮成一句話

你的系統最理想的數學與 ML 架構並不是：

所有Features→加權平均→Authenticity Score\boxed{ 所有Features \rightarrow 加權平均 \rightarrow Authenticity\ Score }

而是：

Series→Component→Evidence→Multi-axis Component State→Calibrated Probability→Watch-level Risk→Authentication Policy\boxed{ Series \rightarrow Component \rightarrow Evidence \rightarrow Multi\text{-}axis\ Component\ State \rightarrow Calibrated\ Probability \rightarrow Watch\text{-}level\ Risk \rightarrow Authentication\ Policy }

再加上：

Hierarchical Learning+Missing Evidence+OOD Detection+Confidence+Human Learning\boxed{ Hierarchical\ Learning + Missing\ Evidence + OOD\ Detection + Confidence + Human\ Learning }

這樣才能真正處理你提到的 **不同 Series 樣本數差很多、每隻 watch feature 不完全相同、不同 component feature 數量差很多、Learning Mode 持續增加資料，以及 Authentication Mode 必須使用現有知識穩定判斷** 的問題。








#### 基於8 classification的new 架構
```
但這套系統就是將一個手錶的每個component相關的images都展示給專業watch工作人員, 然後讓他們對這個watch的Dial, Hands, Crystal, Case, Side, Bezel, Rehaut, Movement, Bracelet-links, Bracelet-endlinks, Bracelet-screws, Bracelet-clasp, Box, Material, Measurement這些compontnts依照他們的專業給予"Original", "Authentic replacements", "Forgery", "Aftermarket", "Modified", "Incorrect Authentic", "Missing", "Not applicable”的分類. 這是跟專業watch工作人員討論過的輸入流程所以不會變動, 那你提到的"你的 8 個答案其實不是同一個維度"就不成立. 請提供新的架構或流程, 或這不會改變?

```

會改，但不是整套推翻。**需要修改的是「Component 最後怎麼建模」這一層**；你原本的專業人員輸入流程完全可以保留，而且我現在反而會建議把它正式定義成：

> **每一個 Component 都是一個固定的 8-class Authentication Classification Problem。**

也就是 Dial、Hands、Crystal、Case、Side、Bezel、Rehaut、Movement、Bracelet-links、Bracelet-endlinks、Bracelet-screws、Bracelet-clasp、Box、Material、Measurement，每一項最後都只能得到你們已經定義好的 8 類之一：

Yc∈{Original, AuthenticReplacement, Forgery, Aftermarket, Modified, IncorrectAuthentic, Missing, NotApplicable}Y_c\in \{ Original,\ AuthenticReplacement,\ Forgery,\ Aftermarket,\ Modified,\ IncorrectAuthentic,\ Missing,\ NotApplicable \}

這樣設計是完全合理的。

我前面提到「8 個答案不是同一維度」，比較像是從純 ML ontology 的角度提出另一種可能的 representation；既然你們的 domain experts 已確認這 8 類就是實際 authentication workflow 的**互斥最終結果**，就沒有必要為了 ML 而改變專家的工作方式。

---

# 更新後我推薦的完整架構

最核心可以改成：

```
                     WATCH SCAN
                         │
                 Series = A1/A2/...
                         │
          ┌──────────────┴──────────────┐
          │                             │
    Component Images              Other Evidence
          │                             │
          └──────────────┬──────────────┘
                         │
                  Feature Extraction
                         │
                         ▼
              Component Evidence Bundle
                         │
            ┌────────────┴────────────┐
            │                         │
       Series Information       Missing / Quality
            │                         │
            └────────────┬────────────┘
                         ▼
              COMPONENT 8-CLASS MODEL
                         │
       ┌─────────────────┼─────────────────┐
       │                 │                 │
    Probability       Confidence       Explanation
    distribution      / Coverage       / Evidence
       │
       ▼
   Predicted label
       │
       ▼
 Original
 Authentic Replacement
 Forgery
 Aftermarket
 Modified
 Incorrect Authentic
 Missing
 Not Applicable
       │
       ▼
 Component Authentication Result
       │
       ▼
     Watch-level
 Authentication Aggregation
       │
       ▼
 Overall Score / PASS / REVIEW / FAIL
```

---

# 1. 最重要的 Training Unit：一個 Watch 的一個 Component

這跟你現在的 workflow 非常吻合。

例如：

```
Watch ID: W000812
Series: A1

Component: Dial

Images:
    dial_front_01
    dial_logo
    dial_text_01
    dial_text_02
    ...
    dial_marker_12

Features:
    OCR features
    text geometry
    logo geometry
    color
    spacing
    texture
    measurements
    image embedding
    ...

Expert label:
    Original
```

這整包才是一個 training example。

不是：

```
image 1 = Original
image 2 = Original
image 3 = Original
```

因為專業人員看的本來就是：

> 「這整組 Dial images 所提供的證據，綜合判斷這個 Dial 是 Original。」

所以模型也應該模仿同樣的 decision process。

---

# 2. 每個 Component 都有自己的 8-class classifier

這點我現在會非常明確地推薦。

不要建立：

```
所有 component
       ↓
一個巨大模型
       ↓
8 classes
```

比較好的架構是：

```
Dial Model
    ↓
8 classes

Hands Model
    ↓
8 classes

Crystal Model
    ↓
8 classes

Case Model
    ↓
8 classes

Movement Model
    ↓
8 classes

Bracelet-links Model
    ↓
8 classes

...
```

因為判斷 Dial 和判斷 Movement 的 evidence 完全不一樣。

例如 Dial 模型可能很重視：

```
OCR
Typography
Logo geometry
Marker spacing
Printing quality
Color
Texture
```

Movement 模型可能重視：

```
Bridge geometry
Engraving
Screw shape
Finishing
Part location
Surface texture
Movement marking
Material
```

Bracelet-screws 又可能主要看：

```
Diameter
Head geometry
Slot geometry
Thread dimensions
Material
Finish
```

所以它們本來就應該是不同模型。

---

# 3. 但 Series 不要完全拆成不同模型

這又是另外一個非常重要的設計。

例如 Dial 不建議直接做：

```
A1 Dial Model
A2 Dial Model
A3 Dial Model
B1 Dial Model
B2 Dial Model
```

全部完全獨立。

因為你已經說了：

> 有些 Series scan 很多，有些 Series scan 很少。

假設：

|Series|Dial training watches|
|---|---|
|A1|1600|
|A2|850|
|A3|420|
|B1|80|
|B2|12|

如果 B2 自己訓練：

```
12 samples
→ B2 Dial model
```

基本上不可靠。

所以我仍然推薦：

# Component-specific + Hierarchical Series Learning

例如：

```
                       DIAL MODEL
                           │
                Shared Dial Knowledge
                           │
              ┌────────────┴────────────┐
              │                         │
          Family A                  Family B
              │                         │
       ┌──────┼──────┐              ┌───┴───┐
       │      │      │              │       │
      A1     A2     A3             B1      B2
```

也就是：

> Dial 的知識是共享的，但是 Series-specific differences 仍然存在。

---

# 4. 所以真正的模型輸入會是

對某一個 component cc：

P(Yc∣Xc,Series,EvidenceMask,Quality)P(Y_c | X_c, Series, EvidenceMask, Quality)

其中：

- XcX_c：所有 component features
- Series：A1 / A2 / A3 / B1 / B2
- EvidenceMask：哪些 evidence 有、哪些沒有
- Quality：image / feature quality
- YcY_c：你們固定的八個 label

最後直接輸出八個 probability。

例如 A1 Dial：

|Class|Probability|
|---|---|
|Original|**0.91**|
|Authentic replacements|0.035|
|Forgery|0.018|
|Aftermarket|0.012|
|Modified|0.015|
|Incorrect Authentic|0.007|
|Missing|0.001|
|Not applicable|0.002|

結果：

```
Dial
Predicted: Original
Probability: 91%
```

非常直觀。

---

# 5. 這裡要特別區分一件事：Expert 的「Missing」和 AI 資料缺失

這不是改變你的 8 classes。

而是系統工程上一定要分開。

你的：

```
Missing
```

是專業人員對 component 的 authentication classification。

例如：

> 這隻 watch 本來應該有某個 component，但實體 component 缺失。

所以：

```
Expert label = Missing
```

完全保留。

但是另一件事是：

> Camera 拍攝失敗。

例如：

```
dial_image_07.jpg
沒有產生
```

這不代表：

```
Dial = Missing
```

同樣：

```
OCR 沒有讀出 ROLEX
```

也不代表：

```
Dial = Missing
```

因此系統內部需要保存：

```
Authentication label:
    Missing
```

和：

```
Evidence availability:
    image_missing
    feature_missing
    extraction_failed
```

兩套不同的東西。

這是非常重要的。

---

# 6. Missing Evidence 不需要增加第九種 classification

例如正常 Dial 預計有：

```
Text evidence
Logo evidence
Marker evidence
Color evidence
Texture evidence
Measurement evidence
```

這次：

```
Text           ✓
Logo           ✓
Marker         ✓
Color          ×
Texture        ×
Measurement    ✓
```

模型仍然輸出八類：

```
Original                82%
Authentic Replacement    7%
Forgery                   4%
Aftermarket               2%
Modified                  3%
Incorrect Authentic       2%
Missing                   0%
Not Applicable            0%
```

但是另外顯示：

```
Evidence Coverage: 71%
Confidence: Medium
```

所以：

**8-class ontology 完全不變。**

---

# 7. 如果 evidence 實在太少，不要增加新 label，而是增加 Decision Status

例如：

```
Prediction:
Original 52%
Authentic Replacement 23%
Forgery 11%
...
```

但是：

```
Evidence Coverage = 28%
```

系統可以顯示：

```
Component prediction:
Original

Decision Status:
MANUAL REVIEW REQUIRED
```

這個：

```
MANUAL REVIEW REQUIRED
```

不是第九個 Authentication classification。

只是 system workflow status。

所以 ontology 還是：

```
8 classes
```

---

# 8. Learning Mode 現在可以非常漂亮地直接對應你的流程

你的實際流程可以定義為：

```
             LEARNING MODE

Scan Watch
     │
     ▼
Identify Series
     │
     ▼
Display component images
     │
     ▼
Professional watch expert
     │
     ▼
Select exactly ONE label
     │
 ┌───┴───────────────────────────┐
 │ Original                     │
 │ Authentic replacements       │
 │ Forgery                      │
 │ Aftermarket                  │
 │ Modified                     │
 │ Incorrect Authentic          │
 │ Missing                      │
 │ Not applicable               │
 └───────────────────────────────┘
     │
     ▼
Save:
Watch ID
Series
Component
Images
Features
Expert Label
     │
     ▼
Training Dataset grows
     │
     ▼
Retrain Candidate Model
     │
     ▼
Validation
     │
     ▼
New Model Version
```

這跟你現在跟專家討論好的 UI 不需要改。

---

# 9. Authentication Mode 則是完全同樣的 scan

差別只在沒有 expert label。

```
          AUTHENTICATION MODE

Scan Watch
     │
     ▼
Series
     │
     ▼
Images
     │
     ▼
Features
     │
     ▼
Component Model
     │
     ▼
8-class probability
     │
     ▼
Predicted Classification
```

例如：

```
Dial
Original                 94%

Hands
Authentic Replacement    78%

Crystal
Original                 91%

Case
Original                 96%

Bezel
Aftermarket              83%

Movement
Original                 98%

Bracelet
Modified                 61%
```

這才是非常符合你 workflow 的設計。

---

# 10. Learning Mode 和 Authentication Mode 其實應該共用完全相同的 Feature Pipeline

非常重要。

不要：

```
Learning pipeline A
Authentication pipeline B
```

應該：

```
                    Scan Watch
                        │
                        ▼
               Image Preprocessing
                        │
                        ▼
               Feature Extraction
                        │
                        ▼
                Evidence Bundle
                        │
           ┌────────────┴────────────┐
           │                         │
      Learning Mode           Authentication Mode
           │                         │
      Expert Label              Model Prediction
           │                         │
      Training DB                 Result
```

這樣才不會有 training-serving skew。

---

# 11. 持續學習這部分完全不需要改

你原來的想法：

> scan 越多 watch → 系統越準

完全可以成立。

但是我仍然不建議：

```
Scan 一隻
↓
expert label
↓
立刻修改 production model
```

比較安全的是：

```
Scan W001
Scan W002
Scan W003
...
expert labels
      │
      ▼
Dataset Version 17
      │
      ▼
Train Candidate Model 18
      │
      ▼
Validation
      │
      ├─ better → Production
      │
      └─ worse → Reject
```

因此：

> **Continual Learning = Dataset 持續成長，而不是 production model 每隻 watch 都立刻改變。**

---

# 12. Series 不平衡仍然是核心問題

這完全不因 8-class ontology 改變。

例如：

```
             Dial Dataset

A1
████████████████████ 1500

A2
████████████ 900

A3
██████ 400

B1
██ 80

B2
▏12
```

B2 很危險。

所以仍然需要 hierarchical sharing。

概念可以寫成：

θB2=λB2θB2+(1−λB2)θFamilyB\theta_{B2} = \lambda_{B2}\theta_{B2} + (1-\lambda_{B2})\theta_{FamilyB}

例如只有 12 隻 B2：

```
大量依靠 Family B / shared Dial knowledge
```

以後 B2 到：

```
500 watches
```

就逐漸更多相信：

```
B2-specific knowledge
```

---

# 13. 但現在還多一個更大的 imbalance：8 Classes 的不平衡

實際資料很可能是：

```
Original               85%
Authentic Replacement   7%
Modified                3%
Aftermarket             2%
Forgery                 1%
Incorrect Authentic      1%
Missing                 0.5%
Not applicable           0.5%
```

如果直接 training：

模型很可能學到：

```
永遠猜 Original
```

還有很高 accuracy。

所以需要：

```
class weighting
balanced sampling
focal loss
rare-class monitoring
per-class metrics
```

尤其：

```
Forgery
Aftermarket
Incorrect Authentic
Modified
```

會是你真正要特別收集的資料。

---

# 14. 所以每個 Component 其實有兩種不平衡

例如 Dial：

### Series imbalance

```
A1 很多
B2 很少
```

### Class imbalance

```
Original 很多
Forgery 很少
```

模型必須同時解決：

Series imbalance+Class imbalanceSeries\ imbalance + Class\ imbalance

這是你 Authentication system 最重要的 ML 問題之一。

---

# 15. 可解釋性也完全可以保留，而且現在更直觀

例如：

```
Component:
Dial

Prediction:
Forgery

Confidence:
92%

Evidence coverage:
94%
```

下面直接列：

```
Strong evidence supporting Forgery

1. Crown logo geometry
   Strong deviation from known A1 Original samples

2. ROLEX typography
   Character spacing outside authentic distribution

3. Minute markers
   Marker-width distribution inconsistent with A1

Evidence supporting Original

1. Dial color
   Consistent with authentic A1 examples

2. Date-window geometry
   Within normal authentic range
```

最後：

```
Model decision:
Forgery — 92%
```

這非常符合 watch expert 的 reasoning。

---

# 16. Explainability 最好做成 Evidence Group

例如 Dial 有 100 features。

不要把 UI 顯示：

```
feature_001
feature_002
feature_003
...
feature_100
```

專家不會想看。

而是 grouping：

```
Typography             Strong suspicious evidence
Logo                   Strong suspicious evidence
Marker geometry        Moderate suspicious evidence
Color                   Supports Original
Material                Supports Original
Texture                 Weak evidence
```

需要 drill-down 才展開 raw features。

---

# 17. 實際可以用 SHAP 做這件事

例如 classifier 得到：

```
Forgery = 92%
```

SHAP 可以指出哪些 features 最推高 Forgery probability：

```
ROLEX_R_width_ratio       +0.31
crown_height_ratio        +0.27
text_baseline_offset      +0.19
marker_spacing_std        +0.14
```

再 mapping 回專家語言：

```
Typography inconsistency
Logo geometry inconsistency
Marker-spacing inconsistency
```

這樣 Authentication report 才容易理解。

---

# 18. Image model 也可以指出「看哪裡判斷」

例如：

```
Dial image
```

CNN / ViT 判 Forgery。

可以使用：

```
Grad-CAM
attention visualization
```

顯示：

```
模型主要注意：

ROLEX logo
皇冠
SWISS MADE
6 o'clock marker
```

這對你們將來做 Authentication PDF report 很有價值。

---

# 19. 接下來是很重要的 Watch-level Authentication Score

現在你的 component label 是固定 8 類，所以 scoring 可以設計得更乾淨。

模型先完全不需要決定：

> Forgery 到底扣多少分。

模型只負責：

```
Component → 8-class probabilities
```

例如 Bezel：

```
Original                 3%
Authentic Replacement    4%
Forgery                   5%
Aftermarket              83%
Modified                  2%
Incorrect Authentic       3%
```

然後另外一個：

# Authentication Policy Layer

決定每一個 label 的影響。

---

# 20. 例如你們可以建立 Authentication Risk Weight

僅舉概念例子，不代表我建議現在直接用這些數字：

|Expert classification|Authentication risk|
|---|---|
|Original|0|
|Authentic replacement|0.15|
|Modified|0.45|
|Incorrect Authentic|0.60|
|Aftermarket|0.80|
|Forgery|1.00|
|Missing|Component dependent|
|Not applicable|Excluded|

然後 component expected risk：

Rc=∑k=18P(Yc=k)rkR_c = \sum_{k=1}^{8} P(Y_c=k)r_k

例如：

```
Bezel:
Aftermarket 83%
```

risk 自然很高。

---

# 21. 但是 Missing 必須依 Component 不同

例如：

```
Box = Missing
```

和：

```
Movement = Missing
```

不能有相同 impact。

因此真正的 risk weight 可以是：

r(Component,Label)r(Component, Label)

例如：

```
Movement + Missing
very high risk

Box + Missing
small authentication impact
```

這也是為什麼：

> ML probability 和 business authentication score 最好分開。

---

# 22. Component 也需要 Criticality

例如你們專家可以定義：

|Component|Criticality|
|---|---|
|Movement|Very High|
|Dial|High|
|Case|High|
|Rehaut|High|
|Bezel|Medium|
|Bracelet|Medium|
|Box|Low|

然後：

WatchRisk=f(RDial,RMovement,RCase,...)WatchRisk = f(R_{Dial}, R_{Movement}, R_{Case}, ...)

而不是：

平均15個component平均15個component

---

# 23. 千萬不要把所有 component 單純平均

假設：

```
Dial          Original 98%
Hands         Original 97%
Crystal       Original 99%
Case          Original 99%
...
Movement      Forgery 96%
```

不能因為其他 14 項正常，就把 Movement forgery 稀釋掉。

所以可以有 hard rule：

```
If:
Critical component
AND
Forgery probability > threshold

→ Watch FAIL
```

這很合理。

---

# 24. 所以 Watch-level 其實最好是兩層

```
Component AI Predictions
        │
        ▼
Probabilistic Score
        │
        +
        ▼
Authentication Rules
        │
        ▼
PASS / REVIEW / FAIL
```

不是 AI 自己決定所有 business logic。

---

# 25. 完整新版 Architecture

我現在會把你的系統正式定義成：

```
┌──────────────────────────────────────────────────┐
│                  WATCH SCAN                      │
│            Series A1/A2/A3/B1/B2                │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│          COMPONENT IMAGE COLLECTION              │
│                                                  │
│ Dial │ Hands │ Crystal │ Case │ Side │ Bezel    │
│ Rehaut │ Movement │ Bracelet │ Box │ Material...│
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│            FEATURE / EVIDENCE EXTRACTION         │
│                                                  │
│ OCR │ Geometry │ Color │ Texture │ Material      │
│ Measurements │ CNN/ViT embedding │ etc.         │
│                                                  │
│ + quality                                   │
│ + missing evidence mask                     │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│       COMPONENT-SPECIFIC AUTHENTICATION MODEL    │
│                                                  │
│             + Series information                 │
│             + hierarchical sharing              │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│                  8-CLASS OUTPUT                  │
│                                                  │
│ Original                                         │
│ Authentic replacements                          │
│ Forgery                                          │
│ Aftermarket                                      │
│ Modified                                         │
│ Incorrect Authentic                             │
│ Missing                                          │
│ Not applicable                                   │
│                                                  │
│ + probability                                    │
│ + confidence                                     │
│ + evidence coverage                              │
│ + explanation                                    │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────┐
│              WATCH-LEVEL AGGREGATION             │
│                                                  │
│ Component criticality                            │
│ Label severity                                   │
│ Hard authentication rules                        │
│ Confidence / coverage                            │
└───────────────────────┬──────────────────────────┘
                        │
                        ▼
                AUTHENTICATION RESULT

                  Score
                  PASS
                  REVIEW
                  FAIL

                  +
                  Explanation
```

---

# 26. Learning feedback loop

旁邊再增加：

```
Professional Watch Expert
        │
        │ Learning Mode
        ▼
8-class Component Labels
        │
        ▼
Training Database
        │
        ▼
Dataset Version
        │
        ▼
Retrain Models
        │
        ▼
Validation
        │
        ▼
Production Model Version
        │
        └──────────────────────────────┐
                                       │
                                       ▼
                              Authentication Mode
```

這就是你真正需要的 **continual learning loop**。

---

# 所以回答你最後的問題

## 「這樣前面的架構是不是全部要改？」

**不用。**

大約 **80% 的架構仍然成立**：

- 階層式 Series learning → **保留**
- Component-level modeling → **保留，而且更重要**
- Evidence/feature grouping → **保留**
- Missing evidence handling → **保留**
- Series sample imbalance → **保留**
- Class imbalance → **保留，而且新增重點**
- Continual learning → **保留**
- Learning Mode / Authentication Mode → **保留**
- Confidence → **保留**
- Evidence coverage → **保留**
- OOD / unseen counterfeit detection → **保留**
- Explainability → **保留**
- Watch-level policy aggregation → **保留**

真正需要修改的是我上一版裡：

```
Authenticity
Originality
Compatibility
Modification
```

四個平行輸出。

**把它拿掉。**

直接替換成：

Component→8-class probability distribution \boxed{ Component \rightarrow 8\text{-class probability distribution } }

也就是：

Original/AuthenticReplacement/Forgery/Aftermarket/Modified/IncorrectAuthentic/Missing/NotApplicable\boxed{ Original / AuthenticReplacement / Forgery / Aftermarket / Modified / IncorrectAuthentic / Missing / NotApplicable }

這樣其實會讓你的整套系統更簡潔，而且完全吻合你們已經跟 professional watch experts 定好的實際操作流程。