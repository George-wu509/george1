

|                         |     |
| ----------------------- | --- |
| [[#### 詳細解釋Harness是甚麼]] |     |
|                         |     |
|                         |     |
|                         |     |
|                         |     |


#### 詳細解釋Harness是甚麼
```
請幫我詳細解釋Harness是甚麼, 它的原理是甚麼請解釋給intern以及senior engineer , 並具體舉例說明. 並分析這website(Google 開源 Harness2！AI 代理不用重新訓練模型，也能自己改進工作流程) 
https://www.inside.com.tw/article/42679-google-harness2-open-source-recursive-agent-harness-propose-probe-compose?fbclid=IwY2xjawVD_25leHRuA2FlbQIxMQBwZG9mBWJyaWQRMTdvS1JWTzJ5TVBDZGQwOXRzcnRjBmFwcF9pZBAyMjIwMzkxNzg4MjAwODkyAAEe0Q4mgHilquAAXXAeBY_a8vkuzAebiuNhx9ep8bwstnENA0i4zAm9Rhs3pp8_aem_rs46Cl10WGYSkKe6kOuWCA
```

**Harness**（原意為馬具、安全帶或測試載具）在 AI 代理（AI Agent）領域中，指的是**圍繞在 LLM 核心模型外圍的控管機制、工具調用介面與運行環境架構**。

LLM 本體（如 Gemini、GPT-4）就像無狀態（Stateless）的推理引擎，本身無法直接與外部世界互動或自我校正。要讓 LLM 變成能自主完成任務的 AI Agent，必須依靠 Harness 來串接提示詞編排、工具執行、記憶讀寫與安全護欄。

## 觀點一：解釋給 Intern（通俗比喻與具體範例）

### 1. 鋼鐵人戰衣比喻

- **LLM 核心模型 = Tony Stark 的大腦**：具備強大的通用邏輯思考能力，但如果要改變他的大腦，等於動開腦手術（對應到重新訓練或微調模型 Fine-tuning），代價昂貴且耗時。
    
- **Harness = 鋼鐵人戰衣與 J.A.R.V.I.S. 系統**：包含戰衣上的抬頭顯示器（Prompt）、感測器與武器介面（Tools/APIs）、安全防禦限制（Guardrails）以及戰鬥腳本（Workflow Script）。
    
- **核心觀念**：當 AI 任務表現不佳時，**我們不需要重新培養一個大腦，而是升級他的鋼鐵戰衣（優化 Harness）**。
    

### 2. 具體應用範例：自動化客服 Agent

- **問題**：客服 Agent 經常算錯折扣金額，且輸出格式混亂。
    
- **傳統做法（改模型）**：收集上萬筆對話去 Fine-tune 模型，費用高且可能破壞模型的通用語言能力。
    
- **Harness 做法（改外圍環境）**：
    
    - **Tool 擴充**：給 Agent 掛載一個計算機 API，只要遇到數學計算就強制呼叫計算機，解決算錯問題。
        
    - **Prompt 優化**：在 System Prompt 補上嚴格的 JSON 回覆範例。
        
    - **Guardrail 攔截**：在輸出端加一層腳本，若輸出的折扣小於 0 元則自動退回重寫。
        

## 觀點二：解釋給 Senior Engineer（系統架構與設計思維）

### 1. 控制平面與推理引擎解耦（Decoupled Architecture）

- **Inference Engine（Frozen Weights）**：將大語言模型定位為不可變（Immutable）、無狀態的基礎設施。
    
- **Agent Harness（Stateful Control Plane）**：維護動態狀態機（State Machine）、DAG 工作流編排、Context Window 壓縮與長短期記憶檢索、Tool Schema 註冊表及安全過濾器。
    

### 2. 優化空間轉移（Optimization Space Shift）

傳統的 LLM 優化是在連續向量空間利用梯度下降（Gradient Descent）更新權重 $\theta$；而 Harness 優化則是將目標轉移至**離散的符號與系統配置空間（Discrete Meta-System Design）**。工程師透過調整 Prompt Templates、Tool Selection Policies 與 Sub-agent Routing，在不更動模型權重的前提下達到系統效能突破。

## 文章分析：《Google 開源 Harness2！AI 代理不用重新訓練模型，也能自己改進工作流程》

這篇文章報導了 Google Cloud AI Research 最新開源的 **Harness2** 框架，其關鍵技術亮點與工程分析如下：

### 1. 核心突破：遞迴式外圍自省（Recursive Agent Harnessing）

- **凍結權重（Zero Weight Fine-Tuning）**：模型權重保持不動，Agent 僅調整外圍系統。
    
- **8 個開放修改維度**：Harness2 允許 Agent 自主改寫 8 大外圍模組，包括**系統提示詞、護欄、記憶、工具指引、子代理、技能、腳本與外掛**。
    

### 2. 三步自我改進機制：Propose-Probe-Compose

Harness2 的自動化升級流程不需要預先準備標準答案（Ground Truth），完全依賴執行行為的驗證：

|**步驟**|**階段名稱**|**運作邏輯**|
|---|---|---|
|**Step 1**|**Propose（提出）**|提出 2 項修改建議：1 項針對通用弱點，1 項針對特定任務瓶頸。|
|**Step 2**|**Probe（探測）**|在相同測試條件下並行執行修改版本與 Baseline，觀察真實行為差異。|
|**Step 3**|**Compose（組合）**|僅保留經數據與行為驗證有效改善表現的修改，記錄至「改進用 Harness」中。|

### 3. 實務效能與工程啟示

- **測試成績**：單輪改進在 WorkBuddy-Bench 提升 11.5 分、JobBench 提升 4.5 分、LAB 法律代理測試提升 11.8%。
    
- **遞迴遞減效應**：加入遞迴（Recursive）循環後可再分別獲得 +3.9 分、+4.2 分與 +6.7% 的額外提升。**工程啟示在於遞迴優化帶來的是「逐步收斂」的增幅，而非無限指數成長**。
    

### 4. 部署風險與隱憂提示

- **非官方支援產品**：屬於研究性質專案，專案依賴 Linux、Python 3.12 與 `uv`。
    
- **資安漏洞**：預設停用了 Codex 執行沙箱，且 Agent 擁有無限制的網路存取權限，實務部署時需特別防範 Benchmark Leakage（過擬合網路上的評測答案）及非預期的網絡行為。