

|                       |     |
| --------------------- | --- |
| [[#### 甚麼是LLM agent]] |     |
|                       |     |
|                       |     |
|                       |     |



#### 甚麼是LLM agent
```
請幫我詳細解釋甚麼是LLM agent? 所以他跟LLM model根據輸入text一個字一個字輸出text的關係是甚麼? 他可以操作component其他軟體的原理是甚麼? 先用具體舉例解釋給只有LLM一些知識的intern, 接著更詳細解釋技術細節
```
我們用一個很直觀的比喻來幫實習生理解，接著再進一步拆解底層的技術架構與運作原理。

## 第一部分：給 Intern 的觀念建立（生活化比喻）

### 1. 什麼是 LLM Agent？它跟單純的 LLM Model 有什麼關係？

簡單來說：

- **LLM Model（大語言模型）**：就像是「大腦」**或**「知識庫」**。你問它問題，它根據文字機率，一字一字（Token by Token）吐出文字回答你。但它本質上是一個**被動的文字生成器，沒有手、腳，也沒有辦法主動上網、操作電腦或記住長時間的事務。
    
- **LLM Agent（AI 代理人）**：就像是「給這個大腦裝上了手、腳、眼睛與工具箱」**。它是一個以 LLM 為核心的**完整軟體系統。
    

#### 具體舉例：

- **單純使用 LLM Model**：
    
    > **你**：「幫我訂明天下午 3 點台北到高雄的高鐵票。」
    > 
    > **LLM Model**：「好的，您可以開啟高鐵 App，選擇明天下午 3 點的班次，點擊預訂並輸入信用卡號完成付款...」（它只能吐出說明文字，無法真的幫你訂）。
    
- **使用 LLM Agent**：
    
    > **你**：「幫我訂明天下午 3 點台北到高雄的高鐵票。」
    > 
    > **LLM Agent 運作過程**：
    > 
    > 1. **思考 (Reasoning)**：大腦意識到「我需要知道明天的日期與車次狀況」。
    >     
    > 2. **調用工具 (Tool Use)**：Agent 呼叫高鐵 API（或開啟瀏覽器）查詢明天 15:00 是否有空位。
    >     
    > 3. **觀察結果 (Observation)**：系統回傳「15:15 有班車，尚有座位」。
    >     
    > 4. **執行動作 (Action)**：Agent 呼叫訂票 API 輸入你的個人資料，並發送付款請求。
    >     
    > 5. **最終回覆 (Final Response)**：回傳「已成功為您預訂明天 15:15 的高鐵票，訂位代碼為 XXXXX」。
    >     
    

### 2. Agent 是如何「操作其他軟體/元件」的？

LLM 雖然只會「輸出文字」，但**文字可以是程式碼或結構化的指令（例如 JSON）**。

#### 運作邏輯比喻：

想像 LLM 是一個只會寫信的指揮官（LLM Model），而其他軟體（如 API、資料庫、瀏覽器）是下面的兵。

1. **指揮官寫下指令（文字/JSON）**：「呼叫 API：`search_flights(date='2026-09-14')`」。
    
2. **中間層系統（Agent 框架）** 看懂了這行文字，幫忙執行這個 API。
    
3. **API 執行完畢** 後，把結果寫成文字回傳給指揮官：「找到 3 個航班...」。
    
4. **指揮官讀懂結果後**，再決定下一步要寫什麼指令。
    

## 第二部分：LLM Agent 的技術架構與細節解析

在工程實作上，LLM Agent 是利用 **ReAct (Reasoning + Acting) 模式**、**Function Calling (Function/Tool Call)** 以及外部環境膠水代碼（Glue Code）組合而成的閉環系統。

```
                    ┌────────────────────────┐
                    │      User Prompt       │
                    └───────────┬────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                           LLM Agent                             │
│                                                                 │
│   ┌────────────────┐     ┌───────────────┐    ┌─────────────┐   │
│   │   LLM Engine   │───► │ Execution/    │──► │ Environment │   │
│   │   (Reasoning)  │     │ Tool Handler  │    │  / Softwares│   │
│   └───────▲────────┘     └───────┬───────┘    └──────┬──────┘   │
│           │                      │                   │          │
│           │                      ▼                   │          │
│           │              ┌───────────────┐           │          │
│           └──────────────│  Observation  │◄──────────┘          │
│                          └───────────────┘                      │
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  Memory (Short-Term / Long-Term Vector DB / Context)    │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 1. LLM Model vs. LLM Agent 的架構差異

|**維度**|**LLM Model (如 GPT-4 / Claude / Llama)**|**LLM Agent**|
|---|---|---|
|**本質**|深度學習權重矩陣（Autoregressive Text Predictor）|包含 LLM、記憶體、規劃器與工具的整體軟體架構|
|**輸入與輸出**|Text In $\rightarrow$ Text Out (Token by Token)|Goal In $\rightarrow$ Action execution $\rightarrow$ Goal Achieved|
|**控制流程**|單向（Prompt $\rightarrow$ Output 隨即結束）|循環閉環（Loop: Think $\rightarrow$ Act $\rightarrow$ Observe $\rightarrow$ Repeat）|
|**外部互動**|無（僅能處理 Context Window 內的文字）|可透過 API、CLI、SDK 操作外部軟體與資料庫|

### 2. Agent 能夠操作軟體/元件的底層技術原理

Agent 能夠調用外部元件，關鍵在於以下四個技術環節：

#### 步驟一：Tool Schema 宣告（讓 LLM 知道有什麼工具可用）

在系統 Prompt (System Message) 中，開發者會透過 JSON Schema 定義好可用工具的名稱、功能敘述與參數格式。

_範例 JSON Schema 宣告：_

JSON

```
{
  "name": "query_database",
  "description": "查詢資料庫中的客戶訂單資訊",
  "parameters": {
    "type": "object",
    "properties": {
      "customer_id": {"type": "string", "description": "客戶 ID"}
    },
    "required": ["customer_id"]
  }
}
```

#### 步驟二：LLM 輸出結構化指令 (Function Calling / Structured Outputs)

LLM 經過微調（Fine-tuning）或 Prompt 引導後，如果發現無法僅憑自身知識回答問題，它不會輸出普通文字，而是輸出**符合約定格式的結構化字串**（如 JSON 或特定的 Markdown 標籤）：

JSON

```
{
  "thought": "用戶想查詢客戶 C12345 的訂單，我需要調用 query_database 工具。",
  "tool_call": {
    "name": "query_database",
    "arguments": { "customer_id": "C12345" }
  }
}
```

#### 3. 中間層解析與執行 (Parser & Runtime Coordinator)

Agent 框架（如 LangChain、LlamaIndex、AutoGPT 或自研 Python Runtime）攔截到 LLM 吐出的文字後：

1. **JSON Parser** 解析這段文字。
    
2. 比對工具庫，匹配到對應的 Python 函式或 API。
    
3. 替換參數並真正執行 code（例如發送 HTTP `POST` 請求到 SQL API 或執行 Docker 容器裡面的 Bash 腳本）。
    

#### 4. 觀察與回饋（Observation Loop）

工具執行的結果（例如 JSON 回應 `{"status": "success", "orders": [101, 102]}`）會被包裝成新的 User/System Message，重新丟回給 LLM 的 Context Window 中。LLM 讀取這個新輸入後，進行下一次思考與決定。

### 3. Agent 的四大核心模組 (Core Components)

一個完整的特化或通用型 Agent 系統通常由以下四大部分組成：

1. **Brain / Planning (規劃與推理)**
    
    - **Task Decomposition**：將複雜任務拆解成子任務（如 Chain-of-Thought, Tree-of-Thought）。
        
    - **Self-Reflection**：執行失敗時進行自我糾錯（如 ReAct, Reflexion 機制）。
        
2. **Memory (記憶體系統)**
    
    - **Short-term Memory**：Context Window 內的對話歷史。
        
    - **Long-term Memory**：利用 Vector Database (RAG) 或 KV Store 儲存跨會話的歷史紀錄或先驗知識。
        
3. **Tools / Actions (工具庫與執行介面)**
    
    - REST APIs, Web Browsing (Playwright/Selenium), Code Interpreter (Python Sandbox), Database Drivers (SQL/NoSQL)。
        
4. **Execution Engine (控制迴圈)**
    
    - 負責控制 `While loop` 流程，處理 Timeout、Retry、權限驗證與安全沙盒環境。
        

### 摘要總結

- **LLM Model** 是一個高智商但身處黑室內的文字推理引擎。
    
- **LLM Agent** 則是為該引擎配備了**感知（Memory）**、工具調用（Tools/APIs）**與**閉環決策（Control Loop）的代理系統。它藉由把 LLM 輸出的文字轉換為結構化的程式呼叫，完成跨系統與跨軟體的實際操作。