
|                                                 |     |
| ----------------------------------------------- | --- |
| [[#### OpenAI — Applied AI Engineer Codex職務流程]] |     |
|                                                 |     |
|                                                 |     |

#### OpenAI — Applied AI Engineer Codex職務流程
```
請深入詳細解釋譬如OpenAI — Applied AI Engineer, Codex Core Agent position, 以下的每一步驟是甚麼怎麼具體實現:
主要研究 Agent 在真實 Coding 任務中的 Task Success、Tool Usage、Context Construction、Regression Testing 及 Production Failures

```

# OpenAI — Applied AI Engineer, Codex Core Agent：完整技術解析與實作流程

這個職位的核心不是單純開發 Coding LLM，也不是使用 Codex 幫工程師寫程式，而是：

研究、設計、評估並改善 Coding Agent，讓 AI 能夠在真實、複雜的 Software Repository 中，可靠地完成程式修改、Debugging、Testing、Refactoring 與長時間的軟體工程任務。

我查閱了 OpenAI 目前公開的 [Applied AI Engineer, Codex Core Agent](https://openai.com/careers/applied-ai-engineer-codex-core-agent-san-francisco/) 職缺。官方特別強調：

- Agent 在真實 Coding Tasks 的 Solve Rate 與可靠性。
    
- Prompting、Tool-use Strategies、Context Construction。
    
- Evaluation、Regressions、Failure Modes 與 Edge Cases。
    
- Production Failure Analysis，以及將真實使用資料回饋給 Research。
    
- Token、Latency、Cost、Reliability 與使用者實際獲得的價值。
    

這些不是五個彼此獨立的研究題目，而是一個完整的 Agent Improvement Loop。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

+1

## 一、首先理解：Coding Agent 與 Coding LLM 有什麼不同？

假設工程師提出：

> 在一個有 2,000 個 Python files 的專案中，修復自動對焦功能。現在當感測器無法提供距離讀數時，程式可能會不斷重試。請修復這個問題、保留原本正常功能，並新增 Regression Tests。

### 傳統 Coding LLM

工程師把幾個 Python files 貼進 ChatGPT，模型分析後提供修改建議或程式碼。

但模型可能不知道：

- 其他 files 如何呼叫 Autofocus。
    
- Hardware API 的例外處理方式。
    
- 現有測試是否會失敗。
    
- 是否破壞其他相機的 Autofocus。
    
- 修改之後能不能真正執行。
    

### Coding Agent，例如 Codex

Agent 可以自行搜尋 Repository、讀取程式、使用 Shell、修改檔案、執行測試、觀察錯誤，再進一步修正。

Coding Agent 的核心運作迴圈（概念架構）

User Task

修復 Autofocus，新增 Regression Tests

Context Builder

Repository Search、Relevant Files、Dependency、Git History、Tests

LLM / Reasoning Model

理解任務、決定下一步、選擇 Tools

Search / Read

搜尋與閱讀程式

Edit

修改程式

Execute / Test

執行與驗證

Observation → Evaluate → Retry

收集 Tool Results、Test Results、Errors，再送回模型迭代

直到完成、需要人工介入，或達到執行限制

Verified Patch / PR + Test Evidence

可審查的修改、測試結果、未解決風險

這個架構通常稱為 Agent Harness + Model + Execution Environment。

- Model：負責推理、產生程式碼、決定行動。
    
- Harness：負責協調模型、Tools、Context、狀態、限制與停止條件。
    
- Execution Environment：提供 Repository、Sandbox、Shell、Tests、檔案系統。
    
- Evaluation System：從外部判斷完成的修改是否真正正確。
    

OpenAI 也在 2026 年 2 月的工程文章中公開說明，Codex 不同操作介面背後共用核心 Agent Harness，而其 App Server 提供與不同客戶端整合的介面。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

+1

Applied AI Engineer, Codex Core Agent 的主要工作，就是讓這個迴圈的完成率更高、失敗率更低、速度更快，而且能量化地證明改善有效。

接下來我用同一個完整範例，依照真實工程開發順序來解釋五個核心領域。

## 二、建立真實 Coding Task：Senior Engineer 首先做什麼？

我們以一個具體的 Python Hardware/AI 專案作為貫穿全文的例子。

假設它是一個類似 Moonlight / ImagingLibWatch 的自動化影像檢測系統，包含：

- Python + PyQt UI。
    
- Keyence Laser Distance Sensor。
    
- Zaber Motion Control。
    
- 多相機拍攝與 Autofocus。
    
- Image Processing 與 AI Inference。
    
- `pytest`、GitHub、CI/CD。
    

以下程式架構、測試資料與數字是為教學建立的假設範例，不代表我已讀取或驗證你的實際 Repository。

### Step 0.1 — 定義 Coding Task

使用者要求 Codex：

> Fix the Keyence-based autofocus process. When OUT1 returns an invalid or missing reading, the autofocus routine may retry indefinitely. Implement bounded retries, enforce the allowed Z-stage range, preserve existing successful autofocus behavior, and add regression tests.

中文意思：

目前自動對焦程式在 Keyence OUT1 沒有有效讀數時，可能一直嘗試移動 Zaber，造成工作流程卡住。

要求 Agent：

1. 找到 Autofocus 實作。
    
2. 找到 Keyence Sensor 的資料讀取介面。
    
3. 理解 Zaber Motion Control 的操作。
    
4. 修改 Retry 邏輯。
    
5. 確保任何移動都不超過 Hardware Safe Bounds。
    
6. 新增 Unit / Integration Tests。
    
7. 確保其他 Autofocus Modes 沒有壞掉。
    
8. 提供可以人工審查的程式修改與測試證據。
    

在面試中，一個 Senior Applied AI Engineer 不會直接回答「我會把 Prompt 改得更好」。

他會先問：如何定義這個 Agent Task 的成功、如何重現、如何評估，以及如何確定改善可推廣到其他 Coding Tasks？

這是以下所有工作的出發點。

# 三、Task Success：Agent 到底有沒有真正完成任務？

Task Success 不只是模型回答正確，也不是程式能通過 `pytest` 就算成功。

真正的問題是：

> Given a real repository and a software engineering request, can the agent deliver a correct, maintainable, verified change without introducing unintended behavior?

## Step 1.1 — 先制定 Acceptance Criteria

對這次 Autofocus Task，我會建立以下規格。

|驗證範圍|必須滿足的條件|
|---|---|
|Functional Correctness|有效 OUT1 讀數時 Autofocus 正常完成|
|Error Handling|無效 OUT1 讀數時有明確的 Retry 上限|
|Safety|Zaber Target Z 不得超過設定的安全範圍|
|Failure Recovery|達到上限時返回明確的失敗狀態，不得無限重試|
|Compatibility|其他 Autofocus Modes 行為不變|
|Testing|新增有效讀數、無效讀數、Boundary、Retry Exhaustion 測試|
|Maintainability|遵循專案架構，不在多個地方重複實作 Retry|
|Deliverable|提供 Git Diff、測試結果及未驗證的硬體風險|

其中最重要的是 Oracle。

Oracle 指的是用來判斷結果正確與否的可信標準。它可能是一組由人工設計的 Hidden Tests、Formal Invariants、參考輸出或真實硬體驗證結果。

Agent 自己寫的測試不能當作唯一 Oracle，因為 Agent 可能同時把實作與測試寫錯。

例如 Agent 寫出：

```
def autofocus():    for _ in range(10):        zaber.move_relative(3)        reading = keyence.read_out1()        if reading is not None:            return True    return False
```

程式看起來已經解決「無限重試」。

但它仍然存在問題：

- 不檢查 Z 的絕對安全邊界。
    
- 沒有處理非數值和感測器特殊錯誤值。
    
- `move_relative()` 發生例外時可能沒有安全停止。
    
- 即使讀到有效數值，也不代表已完成 Autofocus。
    
- 可能沒有保持原本成功路徑的移動與對焦邏輯。
    

因此，Task Success 必須驗證真正的使用者需求，而不是驗證 Agent 自己聲稱完成的工作。

## Step 1.2 — 建立可自動判定的 Success 指標

可以定義多個不同維度。

|Metric|定義|代表什麼|
|---|---|---|
|Task Solve Rate|完全通過任務 Oracle 的比例|核心成功率|
|First-attempt Solve Rate|不經重新啟動或額外人工提示即成功的比例|初次完成能力|
|Compilation / Import Pass Rate|修改後可成功建置或匯入的比例|基本程式有效性|
|Test Pass Rate|通過既有測試與指定測試的比例|自動化檢查結果|
|Regression-free Rate|沒有破壞既有行為的比例|修改安全性|
|Human Acceptance Rate|Reviewer 接受修改的比例|實際可用程度|
|Median / P95 Completion Time|Agent 從開始到完成的時間分布|效率及尾端延遲|
|Cost per Solved Task|總成本除以成功任務數|經濟效率|

例如：

假設有 200 個真實 Coding Tasks，固定 Agent 執行預算，每個 Task 各執行一次：

- 成功 130 個。
    
- 30 個修復部分問題。
    
- 25 個測試失敗。
    
- 15 個因環境、Tool 或 Timeout 而無法完成。
    

則：

\[ \text{Solve Rate}=\frac{130}{200}=65\% \]

但這個 65% 並沒有完整說明系統好不好。

假設另外兩個 Agent 設定的測試結果如下。

假設實驗：三種 Agent 設定的比較

0%20%40%60%80%Baseline AStrategy BStrategy C

示意數據，不是 OpenAI 官方測試結果

|   |   |   |   |
|---|---|---|---|
|指標|A|B|C|
|Solve Rate|65%|74%|76%|
|平均耗時|5 分鐘|6 分鐘|14 分鐘|
|平均每任務成本|$0.60|$0.72|$2.10|
|每成功任務成本|$0.92|$0.97|$2.76|

Strategy C 成功率最高，但成本及耗時都顯著增加。

如果目標是最小化每個完成任務的成本，A 可能較合適；如果希望提高成功任務比例、且能接受適度增加成本與延遲，B 可能是合理的選擇。

Senior Engineer 必須提供這類 Quality–Latency–Cost Tradeoff，而不只是宣布「新版 Prompt 提升了 11%」。

## Step 1.3 — 建立 Coding Agent Evaluation Dataset

單次 Autofocus Task 不足以判斷 Agent 是否真的變強。

必須建立多樣化的 Benchmark。

例如建立 500 個任務：

|類別|數量|例子|
|---|---|---|
|Bug Fixing|150|修復 Retry、Race Condition、Exception Handling|
|Feature Implementation|100|新增 API、UI 工作流程|
|Refactoring|80|解耦 Module、改善 Dependency|
|Regression Repair|70|修復更新造成的相容性問題|
|Multi-file / Long-horizon|70|跨數十個 Files 新增功能|
|Environment / Integration|30|修復 Build、Packaging、Dependency 問題|
|合計|500||

每個 Case 不只是存一段 Prompt，而是一個完整的 Evaluation Package。

```
eval_dataset/
  task_0001/
    task.json
    base_commit.txt
    environment.lock
    acceptance.md
    setup.sh

  task_0002/
    ...

private_oracles/
  task_0001/
    hidden_tests/
    safety_invariants.json
```

其中 `task.json` 的概念：

```

{
  "task_id": "AF-001",
  "repository": "watch-inspection",
  "base_commit": "fixed-evaluation-commit",
  "prompt": "Fix unbounded autofocus retries...",
  "category": "bug_fix",
  "difficulty": "medium",
  "allowed_tools": [
    "read_file",
    "search_code",
    "apply_patch",
    "run_tests"
  ],
  "budget": {
    "max_wall_seconds": 900,
    "max_model_tokens": 150000,
    "max_tool_calls": 100
  },
  "oracle": {
    "hidden_test_suite": "af_regression_v3",
    "require_original_tests": true,
    "require_safety_checks": true
  }
}
```

實務上 `base_commit` 應保存真正的 Commit SHA，Dependencies、測試環境及工具版本也要固定。

這是為了 Reproducibility（可重現性）。

同一個任務，要能讓 A 與 B 兩種 Agent 在完全相同的起點、環境與權限下執行。Hidden Tests 應由獨立評分服務保管，不能直接暴露給 Agent。

### Step 1.4 — 用科學實驗方式確認提升

假設你提出新的 Tool Selection Prompt，要確認是否有效。

實驗步驟：

1. Frozen Benchmark：固定測試集、Commit、Environment、Oracle。
    
2. Baseline A：舊版 Model + Prompt + Harness。
    
3. Candidate B：相同 Model + Harness，只改 Tool Strategy。
    
4. 在相同 Task、相同資源限制下執行。
    
5. 收集結果並比較 Task Success、Cost、Latency、Failure Types。
    
6. 對同一任務的成功/失敗變化做 Paired Analysis。
    
7. 用 Bootstrap Confidence Interval 或適當的配對統計檢定估計不確定性。
    
8. 在未參與開發調整的 Holdout Tasks 上再次驗證。
    

如果 Agent 結果具隨機性，可在固定評估預算下使用多個預先指定的隨機種子或重複執行，並清楚區分 Single-run Success 與 Pass@k。

面試重要觀念： 不能拿 200 個 Task 的 Baseline 與另外 200 個不同 Task 的 Candidate 直接比較，更不能只挑改善最好的 Case 當證據。

OpenAI 的 Agent Evaluation 文件也強調使用 Traces、Graders、Datasets 和可重複的 Eval Runs，而不是只檢查模型最後回答。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI API

+1

# 四、Tool Usage：Agent 怎麼選擇、呼叫並正確使用工具？

Coding Agent 跟一般 Chatbot 最大的技術差異之一，就是它能夠 Act on the Environment。

模型只知道它看過的資訊；工具則讓它有能力查詢、觀察和修改真實世界的狀態。

## Step 2.1 — 設計 Agent 的 Tool Interface

以 Autofocus 任務為例。

我們可以提供下列工具：

|Tool|輸入|輸出|
|---|---|---|
|`search_code`|Query / Symbol|相關檔案與行數|
|`read_file`|Path + Range|Source Code|
|`find_references`|Function Name|Call Sites|
|`git_diff`|Repo State|Changed Files / Diff|
|`apply_patch`|Structured Patch|Patch Result|
|`run_tests`|Test Selection|Pass / Fail / Logs|
|`run_static_analysis`|Scope|Lint / Type Errors|
|`get_repo_metadata`|Project|Environment / Dependency Info|

對 Coding Agent 而言，Tool Interface 設計非常重要。

例如兩個工具：

```
search_code(query="autofocus")
```

以及：

```
search_code(    query="def.*autofocus",    file_glob="*.py",    max_results=30)
```

後者可能比較容易讓 Agent 控制搜尋範圍，減少大量不相關結果。

但參數太多也會讓模型更容易選錯。因此要透過 Evals 判斷 Tool Schema 是否清晰，而不是單純增加選項。

## Step 2.2 — Agent 真正執行時的 Tool Calling Sequence

一個成功的 Agent 可能採取以下順序：

Autofocus 修復任務的示意執行紀錄

1. 理解 Repository
    
    `search_code("autofocus")`
    
    找到 `control/autofocus.py`、`hardware/keyence.py`、`hardware/zaber.py`。
    
2. 追蹤呼叫關係
    
    `find_references("run_autofocus")`
    
    找出哪些 UI、Camera Service 及 Capture Workflow 會呼叫它。
    
3. 閱讀現有邏輯
    
    `read_file("control/autofocus.py", 1, 260)`
    
    確認 Sensor Error、Retry、Motion Limits 與正常 Return Contract。
    
4. 建立失敗重現測試
    
    `run_tests("tests/test_autofocus.py")`
    
    先確認原本的失敗行為，再針對 Bug 建立新的測試。
    
5. 實作修復
    
    `apply_patch(...)`
    
    增加 Retry Bound、Sensor Validation、Safety Check 和明確 Failure Handling。
    
6. 驗證與修正
    
    `run_tests("tests/test_autofocus.py")`
    
    若測試失敗，分析輸出、修復程式、重新執行。
    
7. 檢查 Regression
    
    `run_tests("tests/")` + `git_diff()`
    
    確認其他模式沒有被破壞，最後產生 Patch 與 Test Evidence。
    

但並不是所有 Agent 都會採取這麼合理的流程。

有些 Agent 可能在第一步就直接修改 `autofocus.py`，沒有閱讀 Sensor Interface。

有些則反覆執行：

```
search autofocus
search autofocus retry
search autofocus Keyence
search Keyence autofocus
search autofocus again
```

結果消耗大量 Tokens，卻一直沒有進展。

這稱為 Tool Thrashing：Agent 不斷使用工具，但沒有有效增加資訊或接近任務目標。

## Step 2.3 — 怎麼改善 Tool Selection？

這正是 Codex Core Agent Applied AI Engineer 的工作。

可以設計三種 Experiment。

Experiment A：Minimal Tool Guidance

只告訴模型可以使用哪些 Tools。

Experiment B：Workflow Guidance

增加如下的工作原則：

```
Before modifying code:

1. Identify the relevant implementation.
2. Inspect dependent interfaces and call sites.
3. Reproduce the reported failure when feasible.
4. Make the smallest correct change.
5. Run targeted tests.
6. Run relevant regression tests.
7. Report what was and was not verified.
```

Experiment C：Adaptive Tool Strategy

根據任務情況動態引導。

例如：

- 只需修改小函式：優先直接閱讀少量檔案。
    
- 跨很多 Module：先建立 Dependency Map。
    
- 測試失敗：優先閱讀具體 Failure Trace。
    
- 相同搜尋連續沒有新資訊：改用不同 Symbol、Call Graph 或 Test Entry Point。
    
- 長時間無進展：重新評估假設，必要時停止並報告阻礙。
    

接著用相同 Benchmark 比較三者。

|指標|A|B|C|
|---|---|---|---|
|Task Solve Rate|65%|72%|75%|
|平均 Tool Calls|36|29|25|
|重複搜尋比例|18%|9%|5%|
|平均 Completion Time|8m|7m|6.5m|

以上均為假設結果，用於說明 Experiment 設計。

如果 C 可以同時提升 Solve Rate、減少 Tool Calls，這就是非常有價值的 Harness 改善。

但「呼叫越少越好」不是絕對原則。對安全攸關的程式修改，Agent 多執行一次關鍵 Regression Test 可能比省下一次 Tool Call 更重要。

## Step 2.4 — 實作可靠的 Tool Execution Layer

Senior Engineer 不能只假設每個 Tool Call 一定正常。

一個生產系統需要處理：

- Tool Timeout。
    
- Process Crash。
    
- Invalid Arguments。
    
- Permission Denied。
    
- Missing Dependency。
    
- Partial Output。
    
- 重複執行造成 Side Effects。
    
- Cancellation。
    
- Sandbox Isolation。
    

下面是一個簡化 Python 範例，展示如何包裝單一允許的測試命令：

```
import subprocessimport timedef run_test_tool(test_target: str) -> dict:    allowed = {        "autofocus": ["tests/test_autofocus.py"],        "all": ["tests/"],    }    if test_target not in allowed:        return {            "status": "invalid_argument",            "error": "Unknown test target"        }    command = [        "python", "-m", "pytest",        *allowed[test_target],        "-q"    ]    started = time.monotonic()    try:        result = subprocess.run(            command,            capture_output=True,            text=True,            timeout=120,            check=False,        )        return {            "status": (                "passed" if result.returncode == 0
```

這段程式只示範 Tool Result Contract，並不是完整的安全 Sandbox。

真正 Production 的執行程序應在隔離環境中，由執行層強制限制 Filesystem、Network、Process、CPU、Memory、Credentials 與允許的操作；不能把使用者可修改的 Shell 命令無限制地交給 Host 執行。

尤其在具有實體相機、Zaber Motion Stage 或其他硬體的系統中，Agent 應優先在模擬器及 Mock Environment 驗證，不應直接獲得可任意操作生產硬體的權限。

硬體移動限制也必須由可信的 Motion Controller / Safety Layer 強制執行，而非只靠 LLM Prompt。

OpenAI 官方 Sandbox 文件將隔離的 Filesystem、Commands、Packages、Ports 及受控 External Access 視為 Agent Environment 的重要能力。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI API

+1

# 五、Context Construction：Agent 如何在巨大 Repository 裡找到正確資訊？

這通常是 Coding Agent 最容易被低估、但影響極大的部分。

Context Construction 不只是 RAG，也不是把整個 Repository 塞進 LLM Context Window。

它的真正目的：

> At each reasoning step, provide the agent with the right information, at the right level of detail, within a limited context budget.

也就是讓 Agent 在每個決策點，都能看到目前最需要的資訊。

## Step 3.1 — 為什麼需要 Context Construction？

假設一個 Python Repository 有：

- 2,000 個 Files。
    
- 300,000 行程式碼。
    
- 總共約 300 萬 Tokens。
    
- 數百個 Functions、Classes 和 API。
    
- 大量 README、Configuration、Tests、Git History。
    

即使模型的 Context Window 很大，直接讀入完整 Repository 仍然可能造成：

1. Token Cost 過高。
    
2. 大量不相關資訊干擾推理。
    
3. 關鍵資訊被截斷。
    
4. 多輪 Tool Outputs 消耗 Context Budget。
    
5. 舊的程式碼或舊狀態與最新修改混淆。
    
6. 模型忽略位於大量 Context 中間的重要細節。
    

而且 Coding Task 的資訊需求會隨時間改變。

一開始可能需要知道：

`Where is autofocus implemented?`

修改過程中需要知道：

`What does read_out1() return on failure?`

最後需要知道：

`Which other tests might be affected by this change?`

所以 Context 必須是動態建立的。

## Step 3.2 — 實際架構：Hybrid Code Retrieval

在一個大型 Codebase，我會設計以下架構：

User Task + Current Agent State

Task、已讀 Files、現有假設、Test Failures

Query Understanding

辨認 Symbols、Modules、Errors、修改意圖

Lexical Search

ripgrep / BM25 / filename

Semantic Search

Embedding / Vector Search

Structural Search

AST / Call Graph / Imports

Historical Search

Git Diff / Git Blame / Tests

Candidate Merge + Re-ranking

Relevance、Dependency、Recency、Token Budget

Context Package for LLM

精選相關 Code、Contracts、Constraints、Tests、Observations

這是我建議的 Code Retrieval 設計示例，不代表 OpenAI Codex 的所有內部 Retrieval 細節均採取同一方式。

### A. Lexical Retrieval

直接搜尋 Function、Class、Variable、Error 字串。

例如：

```
rg -n "autofocus|OUT1|Keyence|Move Rel Error" .
```

優點是精確、速度快。

缺點是如果使用者說「Sensor Retry」，程式實際名稱卻是 `recover_laser_distance`，字串搜尋可能漏掉。

### B. Semantic Retrieval

將 Code Chunks 建立 Embeddings，使用 Vector Similarity 找到概念上相關的程式。

例如使用者說：

> Find the function that adjusts the camera height when distance readings fail.

即使程式沒有 `camera height` 這些字，也可能找出負責 Z Axis 調整的函式。

但 Semantic Retrieval 可能把很多概念相似、實際上不相關的函式排在前面。

### C. Structural Retrieval

利用 AST、LSP、Import Graph 或 Call Graph。

假設：

```
main.py
  └── capture_workflow.py
       └── autofocus_manager.py
            ├── keyence_controller.py
            └── zaber_controller.py
```

當 Agent 發現 `run_autofocus()` 時，可以進一步找：

- Callers：誰呼叫它？
    
- Callees：它會呼叫誰？
    
- Configuration：有哪些參數影響它？
    
- Tests：哪些測試直接覆蓋它？
    
- Interface Contracts：它回傳什麼資料？可能拋出哪些 Exception？
    

這些資訊對跨 File 修改尤其有價值。

### D. History Retrieval

利用 Git：

```
git log --oneline -- control/autofocus.py

git blame control/autofocus.py

git diff HEAD~1 HEAD -- control/autofocus.py
```

例如：

如果 Autofocus 原本可以正常運作，但某個 Commit 後開始無限重試，Git History 能幫 Agent 迅速縮小原因。

但 Git History 也是可能有用、也可能造成干擾的資訊，因此不應預設將大量 Commit Log 全部塞進 Context。

## Step 3.3 — 如何決定哪些 Files 應該進入 Context？

可以建立一個簡化 Ranking Score：

\[ S(d,q)= w_1L(d,q)+w_2E(d,q)+w_3G(d,q)+w_4T(d,q) \]

其中：

|符號|意義|
|---|---|
|\(L\)|Lexical Relevance|
|\(E\)|Embedding Similarity|
|\(G\)|Graph / Dependency Relevance|
|\(T\)|Test / Task Relevance|
|\(w_i\)|透過 Evaluation 調整的權重|

在實務中，不同 Score 必須先做合理的正規化；也可以利用 Rank Fusion 或 Learned Reranker，而不是手工加權。

例如 Autofocus Task 的候選檔案：

|File|Relevance Score（假設）|處理方式|
|---|---|---|
|`autofocus_manager.py`|0.98|讀取完整相關 Function|
|`keyence_controller.py`|0.92|讀取 OUT1 Interface|
|`zaber_controller.py`|0.87|讀取 Motion Bounds 與 Exception|
|`test_autofocus.py`|0.85|讀取重要 Fixtures 與 Tests|
|`camera_settings.yaml`|0.73|只讀 AF 相關參數|
|`image_stitching.py`|0.20|暫不讀取|
|`authentication_bayesian.py`|0.04|排除|

注意：不一定要讀取整個 File。

例如 `zaber_controller.py` 有 1,500 行，而 Task 只涉及 `move_absolute()` 和 `get_position()`，就優先讀取 Function Signature、相關 Implementation 及 Exception Handling。

## Step 3.4 — Context Window Budget 怎麼分配？

假設這個實驗允許使用 80,000 Tokens 的工作 Context Budget（這是本例自行設定的預算，不代表特定模型的 Context Window 上限）。

可以先設計一個分配策略：

History / State

Relevant Source Code

Reserved Headroom

Task & Constraints

Tool Results / Test Logs

示例工作預算：80,000 tokens

分配只是起點，並不是固定公式。

如果目前 Agent 正在分析大量 Test Failures，就可能需要更多空間保存 Stack Trace；如果已經完成修復，則只需要精簡保留測試摘要與修改狀態。

Senior Engineer 需要設計 Context Budget Manager：

```
def construct_context(    task,    repo_index,    agent_state,    token_budget):    candidates = hybrid_retrieve(        task=task,        index=repo_index    )    ranked = rerank_by_relevance(        candidates,        task,        agent_state    )    context = pack_with_budget(        task=task,        instructions=agent_state.instructions,        evidence=ranked,        recent_results=agent_state.tool_results,        budget=token_budget    )    return context
```

這只是介面層的 Pseudocode；`hybrid_retrieve`、`rerank_by_relevance` 與 `pack_with_budget` 仍需真正實作、測試及評估。

## Step 3.5 — Long-horizon Agent 如何管理 Context？

假設一個 Agent 連續工作兩個小時。

它可能執行：

- 70 次 Tool Calls。
    
- 讀取 50 個 Files。
    
- 修改 12 個 Files。
    
- 執行 30 次 Tests。
    
- 經歷多次失敗、Retry 與 Context Compaction。
    

不可能無限制保存所有原始資料。

因此需要維護 Structured Working State。

例如：

```

{
  "task": "Fix autofocus retry behavior",
  "phase": "regression_testing",
  "repo_base_commit": "abc123...",
  "working_tree_revision": "revision-07",
  "identified_root_cause": "Unbounded retry path",
  "modified_files": [
    "control/autofocus.py",
    "tests/test_autofocus.py"
  ],
  "verified_facts": [
    "OUT1 can return invalid sentinel values",
    "Z movement must respect configured bounds"
  ],
  "test_status": {
    "targeted_tests": "passed",
    "full_suite": "not_run"
  },
  "next_action": "Run full regression suite",
  "open_risks": [
    "Real hardware behavior not yet validated"
  ]
}
```

這個 State 應以具體 Artifacts、檔案版本和 Tool Results 作為依據。

不能讓 LLM 自己總結「所有測試已通過」，然後把它當成可信事實。

應由執行系統保留：

- 哪個 Commit / Working Tree Revision 被測試。
    
- 執行的真正 Test Command。
    
- Exit Code。
    
- Test Report。
    
- 成功與失敗的 Timestamp。
    

如果 Agent 再修改程式，先前 Test Pass 的證據就不能不經檢查直接套用到新版程式。

這是避免 Stale Context 的重要方法。

### Context Construction 的三種常見失敗

|問題|例子|解法|
|---|---|---|
|Missing Context|只看到 Autofocus，沒有看到 Zaber Safety Constraints|Dependency Expansion|
|Irrelevant Context|閱讀大量不相干 Camera / AI Modules|Reranking、Selective Reading|
|Stale Context|Agent 用修改前的測試結果宣告成功|Versioned State、重新執行 Tests|

此外，2026 年 9 月 OpenAI 發布的 Codex 工程文章也提醒，隨著 Coding Models 的能力提升，過度累積的 `AGENTS.md`、Skills 和任務指示可能造成 Context 膨脹，舊時代需要的冗長規則不一定還有幫助。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI Developers

### Step 3.6 — Context Construction 怎麼做 Evaluation？

假設要比較三個策略：

- A：Lexical Search Only。
    
- B：Lexical + Embedding。
    
- C：Lexical + Embedding + Dependency Graph + Adaptive Context。
    

用同一組 Tasks 測試：

|Metric|衡量問題|
|---|---|
|Relevant File Recall@k|正確檔案是否被放入前 k 個結果？|
|Relevant Symbol Recall@k|關鍵 Function 是否找到了？|
|Context Precision|提供的內容中，有多少真正相關？|
|Tokens per Solved Task|每完成任務消耗多少 Context Tokens？|
|Stale-context Error Rate|因舊資訊導致錯誤的比例？|
|End-to-end Solve Rate|最後是否真的比較容易完成任務？|

最終最重要的仍然是 End-to-end Solve Rate。

即使新的 Retrieval System 在 File Recall@10 從 85% 提高到 95%，如果 Agent 成功率完全沒有改善，仍然需要調查是否只是找到了更多檔案，卻沒有讓模型更容易做出正確決策。

# 六、Regression Testing：如何確認 Agent 改善沒有破壞既有功能？

這部分有兩個層次，面試時最好主動區分：

Level 1 — Software Regression

Agent 修改某個 Repository 後，是否破壞原本的程式？

Level 2 — Agent Behavioral Regression

你改善了 Agent 的 Prompt、Tool Strategy 或 Model 後，它是否在其他 Coding Tasks 上變差？

兩種 Regression 必須分開設計測試。

## Step 4.1 — Software Regression：先修復 Bug，再保護舊功能

回到 Autofocus 問題。

我們先把感測器讀數有效性檢查抽象出來。

以下是精簡教學實作，使用模擬 Motion Interface。它只負責取得有效 OUT1 讀數，不把「取得有效讀數」誤認為「自動對焦已成功完成」。

```
import mathfrom dataclasses import dataclassfrom typing import Callable@dataclassclass ProbeResult:    status: str    reading: float | None    attempts: intdef is_valid_reading(raw) -> bool:    if raw is None or raw == "-FFFFFF":        return False    try:        return math.isfinite(float(raw))    except (ValueError, TypeError):        return Falsedef acquire_valid_out1(    read_out1: Callable,    move_to_z: Callable,    start_z: float,    min_z: float,    max_z: float,    step_z: float = 3.0,    max_attempts: int = 3,) -> ProbeResult:    if max_attempts < 1 or step_z <= 0:        raise ValueError("Invalid probe configuration")    if not (min_z <= start_z <= max_z):        raise ValueError("Start Z outside safe bounds")
```

它實現了幾個核心 Invariants：

1. 不得超過 `max_attempts`。
    
2. 不得要求超過 `max_z` 的目標位置。
    
3. 遇到無效讀數不得誤報成功。
    
4. 達到限制要有明確狀態。
    
5. 讀到有效值後停止探測，交給後續 Autofocus Logic 處理。
    

這仍不是可以直接上線的真實 Hardware Autofocus 實作：現場還需要確認座標單位、實際位置回饋、設備故障停止、Motion Completion、感測器例外、取消操作及實體安全限制。

## Step 4.2 — 建立 Unit Tests

例如：

```
from unittest.mock import Mockdef test_valid_reading_requires_no_move():    read = Mock(return_value=25.4)    move = Mock()    result = acquire_valid_out1(        read, move,        start_z=10,        min_z=0,        max_z=20    )    assert result.status == "valid_reading"    assert result.reading == 25.4    assert result.attempts == 1    move.assert_not_called()def test_invalid_reading_stops_after_limit():    read = Mock(return_value="-FFFFFF")    move = Mock()    result = acquire_valid_out1(        read, move,        start_z=10,        min_z=0,        max_z=30,        max_attempts=3    )    assert result.status == "retry_exhausted"    assert result.attempts == 3    assert read.call_count == 3    assert move.call_count == 2
```

上面這些測試回答：

- 如果第一個 Reading 有效，Agent 是否避免不必要移動？
    
- 如果 Reading 永遠無效，是否真的會停止？
    
- 如果 Stage 接近安全邊界，是否拒絕繼續移動？
    

但這仍只是 Unit Tests。

## Step 4.3 — Integration Tests：驗證元件之間的行為

真實的 Autofocus 並不是一個獨立 Function。

可能涉及：

```
Capture Workflow
   ↓
Autofocus Manager
   ↓
Keyence Controller + Zaber Controller
   ↓
Motion Completed Signal
   ↓
Sensor Reading
   ↓
Autofocus Result
   ↓
Camera Capture
```

Integration Test 應確認：

- Motion Command 失敗時，Autofocus 是否正確返回 Failure？
    
- Sensor Timeout 時，是否能讓 UI 顯示錯誤並釋放流程資源？
    
- Cancellation 是否能中止未完成任務？
    
- 如果有其他 Motion Task 同時存在，是否會導致 Race Condition？
    
- Autofocus 失敗後，Capture Workflow 是否會安全停止或按照明確策略降級？
    

這些問題是 Unit Test 通過仍可能遺漏的。

## Step 4.4 — 建立多層 Regression Gates

Gate 1 — Static Validation

Lint、Type Check、Import、Format

Gate 2 — Unit Tests

新舊 Function 的局部正確性

Gate 3 — Integration Tests

Controller、Service、Workflow 互動

Gate 4 — End-to-End Tests

完整 User Journey、UI、Capture Workflow

Gate 5 — Independent Hidden Tests

防止只針對 Agent 自寫的測試過關

Gate 6 — Hardware-in-the-loop

受控實體設備測試與安全檢查

其中 Hardware-in-the-loop 可以配置成受控的專用驗證階段，並不意味每個 AI Agent Task 都需要連接真實設備。

在一般 SaaS Software Engineering 任務中，Gate 6 可能改成 Staging、Browser E2E 或 Production-like Integration Environment。

## Step 4.5 — Agent Behavioral Regression：確保新版 Agent 整體沒有變差

這是 Codex Core Agent 角色更接近的研究問題。

假設你改變 Context Construction，Autofocus 類任務成功率提高了。

但是：

- Python Bug Fix 提升 8%。
    
- TypeScript Feature Development 下降 12%。
    
- Multi-file Refactoring 下降 6%。
    
- Test Generation 沒有變化。
    

整體平均可能掩蓋重要 Regression。

因此，需要做 Slice Analysis。

示例：新版 Agent 相對 Baseline 的成功率變化

Baseline

Candidate

0%20%40%60%80%Python BugsTypeScriptRefactoringTest WritingMulti-file

假設的 Benchmark Slice 結果

這張圖的含義是：

Candidate 在某些任務上明顯變強，但也出現嚴重的局部退步。

Senior Engineer 不能只報告 Aggregate Solve Rate。

還要回答：

為什麼 TypeScript Task 下降？是 Retrieval、Tool Choice、Prompt、Context，還是測試環境引起？

### Step 4.6 — 使用 Ablation Study 找到原因

Ablation 是控制其他條件，只改變一個因素，判斷它真正造成多少改善。

例如：

|Experiment|Semantic Search|Graph Expansion|Context Compaction|
|---|---|---|---|
|E0|Off|Off|Off|
|E1|On|Off|Off|
|E2|On|On|Off|
|E3|On|On|On|

每個 Experiment 都在同一套 Tasks 下執行，分析 Solve Rate、Context Length、Latency 和錯誤類型。

如果 E2 比 E1 好，表示 Graph Expansion 在這個測試條件下可能有正面效果。

但如果 E3 比 E2 差，就代表新的 Context Compaction 可能刪除了重要資訊，或與其他策略存在交互作用。

必要時可以做完整 Factorial Experiment，而不是只逐步增加功能。

在 OpenAI 相近的 AI Systems Engineer 職缺中，官方也明確提到要跨 Model、System Prompt、Harness、Context Construction 和 Tool-use Strategies 做 Ablations，分析 Quality、Latency、Cost 與 Reliability。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

# 七、Production Failures：Agent 在真實環境失敗時，怎麼找出原因並修復？

這個領域非常接近 Senior / Staff AI Engineer 的日常工作。

因為你在 Benchmark 上看到的 Agent，和真實使用者在大型、混亂、持續變更的 Repository 中使用的 Agent，可能有很大差距。

## Step 5.1 — 首先建立 Production Failure Taxonomy

不能把所有失敗都分類成「LLM 不夠聰明」。

至少應區分以下類別。

|Failure Category|實際例子|主要診斷方向|
|---|---|---|
|Model Reasoning|推理錯誤，修復了錯誤的 Function|Reasoning Behavior、Prompt、Model|
|Context Failure|沒看到相關 Interface 或最新修改|Retrieval、Context Packing、State|
|Tool Selection|反覆 Search，沒有執行 Tests|Tool Strategy、Tool Description|
|Tool Execution|Shell Timeout、Process Crash|Tool Runtime、Sandbox|
|Harness / Orchestration|Tool Result 沒有正確回到模型|State Machine、Event Handling|
|Environment|Python Dependency 不存在、OS 不相容|Reproducibility、Environment Setup|
|Regression|修復 A 功能，破壞 B 功能|Test Coverage、Oracle|
|Long-horizon Failure|工作兩小時後忘記原本 Requirement|State、Compaction、Checkpoint|
|Security / Policy|Agent 嘗試讀取敏感檔案或越權操作|Sandbox、Permission、Approvals|
|Infrastructure|Inference Timeout、Rate Limit、Capacity|Serving、Retries、Capacity Planning|
|Product / UX|任務完成但結果沒有正確顯示給使用者|UI State、API Contract、Client|

這個分類決定問題要交給誰處理。

例如：

- Model Reasoning → Applied AI + Research。
    
- Harness Bug → Agent Systems / Core Engineering。
    
- GPU Serving / Capacity → Infrastructure。
    
- Tool Permission UX → Product + Security。
    
- Regression Test Coverage → Applied AI + Software Quality。
    

核心技能是 Root Cause Isolation：不要在還沒確認原因前，就靠修改 Prompt 猜測。

## Step 5.2 — 建立完整的 Agent Trace

真實 Production 需要 Observability。

每個 Task 需要能追蹤以下層次：

Task Run

run_id、repo revision、task class、model / harness version

Model Invocation

request_id、context tokens、output tokens、latency、error status

Tool Invocation

tool name、validated arguments、duration、exit code、artifact reference

Code / Environment State

working tree revision、patch hash、sandbox ID、environment version

Outcome / Evaluation

tests、acceptance、failure category、cost、user feedback

例如一筆簡化的 Trace Event：

```

{
  "run_id": "run-0182",
  "sequence": 14,
  "event": "tool_completed",
  "tool": "run_tests",
  "duration_ms": 18452,
  "exit_code": 1,
  "working_tree_revision": "rev-05",
  "test_summary": {
    "passed": 42,
    "failed": 2,
    "skipped": 1
  },
  "artifact_ref": "test-report-0182",
  "error_category": "test_failure"
}
```

需要注意的是，Production Logs 不應無限制保存完整 Source Code、Secrets、Private Repository 內容或未經授權的敏感資料。應依照資料政策做遮蔽、存取控制、保留期限與必要的分離儲存。

## Step 5.3 — 具體案例：Agent 修復了 Bug，卻沒有真的完成任務

假設使用者反映：

> Codex said the autofocus bug was fixed, but the application still freezes in some cases.

查看 Trace：

|時間|Agent / System Event|結果|
|---|---|---|
|00:00|User Task|修復 Autofocus Retry|
|00:10|Search Code|找到正確 Function|
|00:25|Read File|閱讀 Implementation|
|00:50|Apply Patch|加入 Retry Limit|
|01:20|Run Unit Tests|12 Tests Passed|
|01:35|Agent Final Message|宣告修復完成|
|之後|User Runs Application|仍然卡住|

首先不能直接說 LLM 判斷錯誤。

應提出幾個假設：

Hypothesis A — Missing Context

Agent 沒讀到真正造成 Deadlock 的 Motion Callback。

Hypothesis B — Incomplete Test Coverage

Unit Tests 全部使用 Mock，沒有模擬 Motion Completion 永遠不返回的情況。

Hypothesis C — Incorrect Completion Policy

Harness 容許 Agent 在只執行 Targeted Tests 後宣告完全成功。

Hypothesis D — Environment Difference

Agent 的測試環境與使用者實際運行環境不同。

接著做 Evidence-based Diagnosis。

例如讀取 Integration Trace，發現：

```
AutofocusManager.run()
  |
  +-- Zaber.move_absolute()
        |
        +-- wait_for_motion_complete()
              |
              +-- No completion event
              +-- No timeout
```

這表示真正問題可能是 Motion Completion Wait，而不是 Sensor Retry。

此時改善應包含：

1. 增加可重現該失敗的測試。
    
2. 實作 Motion Completion Timeout / Cancellation。
    
3. 明確定義失敗後的安全處理方式。
    
4. 更新 Agent 的 Context Retrieval / Test Selection。
    
5. 將該 Case 加入固定 Regression Suite。
    

這就是從 Production Failure 產生可量化 Agent Improvement 的流程。

## Step 5.4 — 另一種真實故障：Agent 的 Tool Call 根本沒有執行

例如使用者看到 Agent 長時間停在：

`Running command...`

可能不是模型在思考太久，而是：

- Model Stream 沒有完成 Tool Arguments。
    
- Harness 沒接收到完整 Tool Call。
    
- Tool Executor 卡住。
    
- App Server 沒送出進度事件。
    
- Client 沒收到 Completion / Error Event。
    

這不是單純 Prompt Engineering 可以解決的。

OpenAI Codex 的公開 GitHub Issue 中就有一個相關案例：2026 年 7 月有人回報 App Server 在部分 Custom Tool Calls 開始後，輸入資料長時間沒有完成，導致客戶端無法看到正常的完成事件。這是公開回報的具體 Failure Mode，不應直接推論為所有版本都有相同問題。

![](https://www.google.com/s2/favicons?domain=https://github.com&sz=32)

openai/codex

工程師應檢查：

```
Client
  ↓
App Server
  ↓
Model API / Streaming
  ↓
Tool Call Parsing
  ↓
Tool Dispatcher
  ↓
Sandbox Executor
  ↓
Tool Result
  ↓
Model Continuation
```

定位 Tool Call 卡在哪個階段。

不同位置需要不同修復：

|問題位置|可能修復|
|---|---|
|Model Stream|Idle Detection、Retry Policy、Transport Diagnostics|
|Tool Argument Parsing|Schema Validation、Incomplete Event Handling|
|Tool Dispatcher|State Transition、Idempotency|
|Executor|Process Timeout、Resource Limits、Cancellation|
|App Server|Heartbeat / Progress Signal、Terminal Error|
|Client|Liveness Feedback、Recovery UX|

這裡的 Recovery 設計要特別小心。

例如 Tool 已經成功修改 Files，但結果事件遺失，若 Harness 重試 `apply_patch`，就可能重複修改。

所以需要 Idempotent Operations、Operation IDs、Artifact Checksums，或者在重試之前先核對目前 Workspace State。

## Step 5.5 — Production Monitoring：監控哪些指標？

我會建立以下核心 Dashboard。

Verified Task Success

# 74.2%

+4.1 pp vs baseline

P95 Task Latency

# 12.4m

End-to-end completion

Tool Failure Rate

# 1.3%

Failing executions

Cost per Solved Task

# $1.12

Variable serving cost

以上為展示 Dashboard 的假設數據，並非 Codex 的實際內部指標。

除了這四個 Dashboard 指標，還需要：

|類型|指標|
|---|---|
|Reliability|Timeout Rate、Aborted Runs、Recovery Rate|
|Agent Behavior|Tool Calls per Task、Tool Thrashing、Context Compaction Frequency|
|Correctness|Regression Failure Rate、Verified Solve Rate|
|Efficiency|Input / Output Tokens、Inference Latency、Tool Latency|
|Safety|Permission Violations、Blocked Actions、Approval Escalations|
|User Outcome|Accepted Patches、Human Edits Required、User Reported Failures|

其中 Production Verified Solve Rate 特別難衡量，因為不是每個使用者都會執行完整 Tests 或提供人工 Feedback。

因此需要區分：

- Offline Verified Solve Rate：在具有 Oracle 的 Evaluation Tasks 上得到。
    
- Production Proxy Success：例如 PR 被接受、CI Passed、使用者採用 Patch。
    
- Production Confirmed Outcome：有足夠可信的外部證據確認任務完成。
    

不能將 PR 被 Merge 等同於程式完全正確，也不能將使用者沒有回報問題視為成功。

# 八、Feedback Loop：把 Production Failures 轉成真正的 Agent Improvement

這是 OpenAI 該職位另一個非常重要的工作。

官方特別提到要建立 Data Systems 和 Feedback Loops，讓真實使用案例可以回饋 Evaluation 和 Research。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

## Step 6.1 — 完整 Data Flywheel

1. Production Task / Trace

收集授權且經適當保護的失敗訊號及執行紀錄

2. Failure Detection

依測試、執行錯誤及使用者回饋判定失敗

3. Root Cause Analysis

區分 Model、Context、Tool、Harness、Infrastructure

4. Curated Evaluation Case

建立去識別化或合成且可重現的 Task + Oracle

5. Improvement Experiment

修改 Prompt / Context / Tools / Harness，必要時進行模型訓練

6. Regression and Ablation

以固定 Benchmark 量化 Solve Rate、Latency、Cost

7. Controlled Deployment

Canary、Monitoring、Rollback

8. New Production Evidence

用新版實際結果檢驗改進是否持續

持續迭代

這個流程的難點之一是 Production Data 不一定具有可信的 Ground Truth。

例如使用者只留下：

> Codex couldn't fix my bug.

這不等於可直接訓練的高品質樣本。

需要先取得可授權使用的、可重現且適當保護的問題資料，再建立可靠的預期行為及測試。

## Step 6.2 — Failure Clustering

如果每天有大量 Coding Tasks，不能靠工程師逐筆閱讀每個 Trace。

可以設計半自動 Failure Clustering。

例如先使用確定性特徵分類：

```
tool_timeout
test_failure
context_budget_exceeded
permission_denied
compile_error
task_aborted
```

再用 Embedding、Clustering 或 LLM-based Classification 整理細分類別：

```
test_failure
  ├── wrong_api_usage
  ├── incorrect_exception_handling
  ├── missing_dependency
  ├── wrong_assumption_about_contract
  └── incomplete_multi_file_change
```

但是 LLM 自動分類也會出錯。

可先抽樣給工程師建立 Gold Labels，再估計分類準確率及類別間的一致性。高風險問題如 Unauthorized Tool Access 應有獨立的確定性安全訊號，不依賴單純的語意分類。

## Step 6.3 — 如何選擇下一個要改善的問題？

假設一週中發現下列主要問題：

|Failure Mode|每週案例數|預估可挽回的失敗比例|改善成本|
|---|---|---|---|
|Missing Context|1,200|35%|Medium|
|Tool Timeout|500|70%|Low|
|Incorrect Code Reasoning|900|15%|High|
|Unnecessary Tool Calls|2,000|主要改善成本與延遲|Medium|

這時 Senior Engineer 應評估：

- 失敗發生頻率。
    
- 對使用者造成的影響。
    
- 是否能以較低成本修復。
    
- 預期改善幅度。
    
- 改善是否有 Regression Risk。
    
- 是否需要另一個 Team 的協助。
    

例如 Tool Timeout 可能先做，因為它既頻繁又能以相對低成本改善。

而 Model Reasoning 雖然重要，卻未必能只靠 Prompt 在短時間大幅提升。

這就牽涉 Research Prioritization 和 Engineering Leadership。

## Step 6.4 — 什麼時候需要 Fine-tuning / Post-training？

這個職位雖然接近 AI Research，但不是所有問題都應靠重新訓練模型解決。

|問題|優先考慮|
|---|---|
|搜尋不到正確 Function|Retrieval / Indexing|
|重複呼叫相同 Tool|Tool Strategy / Harness|
|長時間任務忘記限制條件|Context / State Management|
|Tool Result 丟失|Orchestration / Infrastructure|
|經常產生相同種類的錯誤程式|Model Behavior、資料或 Post-training|
|程式可執行但推理品質不足|更適合的模型、Post-training、Task-specific Feedback|
|Model 正確但測試環境失敗|Environment / Tooling|

如果已經確認問題來自模型行為，可以與 Research Team 合作：

1. 收集通過資料使用審查的高品質失敗案例。
    
2. 建立成功的 Tool-use Trajectories 或高品質解題範例。
    
3. 定義可以驗證的 Reward 或 Grader。
    
4. 評估是否適合 SFT、Preference Optimization 或其他 Post-training。
    
5. 在獨立 Tasks 上測試模型更新。
    
6. 再與新的 Harness 組合做 End-to-end Evaluation。
    

不能把單純的「Test Passed」無條件視為最好的 Reward，因為 Agent 可能利用不完整測試、修改測試、跳過測試，或採取不安全的捷徑。

必須保護 Oracle、限制評分環境的權限，並考慮 Reward Hacking。

# 九、真正上線：Agent 更新如何安全部署到 Production？

完成實驗並不等於可以直接讓所有使用者使用。

Coding Agent 的 Harness 更新，即使沒有換模型，也可能造成大規模 Regression。

例如改變 Context Compaction，可能使一些長時間 Tasks 忘記限制條件。

因此需要像一般 Production Software 一樣部署。

## Step 7.1 — 固定 Release Artifact

一次可重現的 Agent Release，至少應清楚記錄：

```
Agent Release: v2.4.0

Model: pinned-model-version
Harness: git-sha-abcdef
Prompt Bundle: prompt-v18
Tool Schema: tools-v7
Context Policy: context-v12
Sandbox Image: sha256:...
Evaluation Dataset: eval-v21
Evaluation Report: report-2026-10-11
```

實際存放形式可以不同，但關鍵是能追溯「到底哪一個版本造成成功或失敗」。

## Step 7.2 — Canary Deployment

例如：

Production Traffic Rollout

示例策略

Initial Canary

1%

Expanded Canary

5%

Limited Rollout

20%

Full Rollout

100%

實際升級條件要事先定義，不是單純每經過幾小時就自動擴大。

應比較新版與舊版在可比較的流量切片中的：

- Tool Failure / Timeout。
    
- Aborted Tasks。
    
- Verified Outcomes。
    
- Latency / Cost。
    
- User Complaints。
    
- Safety Violations。
    

並針對重大錯誤設定自動停止條件。

## Step 7.3 — Rollback 不能只恢復 Prompt

假設新版 Agent 的 Context Management 有問題。

回復到舊版 Prompt 可能不夠。

因為新版可能同時改變：

- Tool Schema。
    
- Session State Format。
    
- Context Compaction。
    
- Sandbox Image。
    
- Execution Policy。
    
- Intermediate Artifacts。
    

因此 Rollback 要設計版本相容性、Session Handling，以及是否能安全恢復尚未完成的 Tasks。

對有 Side Effects 的操作，更不能直接將整段任務重播。

這種問題正是 Long-horizon Stateful Agents 比一般 Stateless API 難維護的原因。

# 十、把所有步驟串起來：一個 Codex Core Agent Engineer 的完整 Research Project

假設你剛加入 OpenAI Codex Core Agent Team。

Manager 給你一個專案：

> Improve Codex's success rate on large-repository bug-fixing tasks, especially failures caused by missing or irrelevant context. Maintain acceptable latency and inference cost.

也就是：

改善 Codex 在大型 Repository 的 Bug Fixing 能力，重點處理 Context 不完整或不相關的問題，並控制延遲和成本。

你不是去替使用者修理某個 Autofocus Bug。

你要改善的是能夠替成千上萬使用者修復這類 Bug 的 Coding Agent 本身。

這個差異非常重要。

## Phase 1 — Baseline Evaluation

先選出 500 個 Coding Tasks，包含大型 Python、TypeScript、Rust 等 Repository。

固定：

- Model Version。
    
- Prompt Version。
    
- Harness Version。
    
- Tool Schema。
    
- Sandbox Environment。
    
- Evaluation Budget。
    
- Hidden Tests。
    

得到 Baseline：

|Metric|Baseline（假設）|
|---|---|
|Task Solve Rate|62%|
|Relevant Symbol Recall@10|82%|
|Average Context Tokens|45,000|
|Average Completion Time|8.5 min|
|Tool Calls per Task|38|
|Cost per Solved Task|$1.40|

## Phase 2 — Failure Analysis

分析失敗的 Tasks。

例如從有充分證據的失敗案例中，標記出：

- 30% 有 Missing Context。
    
- 20% 有 Incorrect Reasoning。
    
- 15% 有 Tool Execution 問題。
    
- 10% 有 Environment 問題。
    
- 25% 屬其他或多重原因。
    

此處百分比是示意且假設每個失敗案例只分配一個主要原因；真實 Failure Taxonomy 可能允許多個原因並存。

接著挑選 Missing Context 類別，詳細調查：

- Agent 是否找到核心 Function？
    
- 是否找到 Callers / Callees？
    
- 是否讀取相關 Tests？
    
- 是否知道 Repository 規則？
    
- 是否因 Context Compaction 丟失重要資訊？
    

## Phase 3 — Proposed Improvement

提出新的 Context Retrieval 策略：

```
Current:
User Task
    ↓
Lexical Search
    ↓
File Reading
    ↓
LLM

Proposed:
User Task
    ↓
Lexical + Semantic Retrieval
    ↓
Dependency Expansion
    ↓
Test-aware Reranking
    ↓
Selective Code Reading
    ↓
Adaptive Context Packing
    ↓
LLM
```

但要避免同時修改全部元件後無法定位效果。

因此分階段進行 Ablation。

## Phase 4 — Engineering Implementation

實際上需要寫的系統可能包括：

```
coding_agent_eval/
  datasets/
    tasks.jsonl

  retrieval/
    lexical.py
    semantic.py
    code_graph.py
    reranker.py

  context/
    builder.py
    budget_manager.py
    compaction.py

  tools/
    search_code.py
    read_file.py
    run_tests.py

  evaluation/
    runner.py
    grader.py
    metrics.py
    slices.py

  observability/
    traces.py
    failure_classifier.py

  experiments/
    baseline.yaml
    candidate.yaml
```

這是合理的專案拆分示例，而不是 OpenAI 內部實際檔案結構。

Evaluation Runner 的核心概念：

```
def evaluate_agent(    tasks,    agent_config,    sandbox_factory,    grader):    results = []    for task in tasks:        # Fresh isolated workspace at fixed commit        with sandbox_factory.create(            repo=task.repository,            commit=task.base_commit,            environment=task.environment,        ) as workspace:            # Run agent with pinned configuration            run = agent_config.run(                task_prompt=task.prompt,                workspace=workspace,                budget=task.budget,            )            # Independent evaluation, not agent self-rating            grade = grader.evaluate(                task_id=task.task_id,                workspace=workspace,                agent_run=run,            )            results.append({                "task_id": task.task_id,                "solved": grade.solved,                "failure_type": grade.failure_type,                "tool_calls": run.tool_calls,
```

這是 Pseudocode：真實實作還要處理 Agent Timeout、Sandbox Cleanup、Eval Infrastructure Failures、Task Retry Policy、Hidden Oracle Isolation 及結果的重現性。

## Phase 5 — Experiment Comparison

執行完成後，假設得到：

|Metric|Baseline|Candidate|
|---|---|---|
|Task Solve Rate|62%|71%|
|Relevant Symbol Recall@10|82%|94%|
|Average Context Tokens|45,000|37,000|
|Average Completion Time|8.5 min|7.2 min|
|Tool Calls per Task|38|29|
|Cost per Solved Task|$1.40|$1.12|

這代表 Candidate 同時：

- 提升 9 個百分點的 Task Solve Rate。
    
- 提高相關程式碼搜尋能力。
    
- 減少不必要的 Context。
    
- 減少 Tool Calls。
    
- 改善 Latency 和 Cost Efficiency。
    

但仍需確認變化的統計不確定性、各 Task Slice 的表現，以及候選版本是否對其他工作流程產生 Regression。

## Phase 6 — Holdout + Production Validation

在另一組獨立 Tasks 上驗證。

如果改善仍然成立，再透過 Canary Rollout 觀察真實使用行為。

這時要檢查：

- 大型 Repository 是否同樣有改善？
    
- 新的 Context Retrieval 是否會在特定語言或專案結構退步？
    
- 是否增加 Sandbox Runtime 的 CPU / Memory 消耗？
    
- 是否有新的 Permission / Security 問題？
    
- 是否有更多使用者實際完成 Tasks？
    

## Phase 7 — Research and Product Handoff

最後向 Research、Infrastructure、Product 團隊提交完整成果。

一份優秀的交付應包含：

|Deliverable|內容|
|---|---|
|Problem Definition|具體 Failure Mode 與 User Impact|
|Baseline Analysis|原本成功率、成本、延遲|
|Failure Taxonomy|Root Cause 分類與證據|
|Proposed Design|新策略架構與技術決策|
|Implementation|程式修改、Feature Flags、Tests|
|Evaluation Report|A/B、Ablation、Confidence Intervals|
|Regression Report|各 Task Slice 的退步風險|
|Rollout Plan|Canary、Monitoring、Rollback|
|Research Feedback|哪些問題應回饋到 Model Training|

這才是完整的 Applied AI Engineering Research → Production Delivery。

# 十一、Senior 面試可能追問什麼？應該如何回答？

這個職位尤其適合考 Agent System Design、Evaluation Methodology 和 Root Cause Analysis。

|面試題目|Senior Engineer 應展現的能力|
|---|---|
|How do you measure coding agent success?|定義 Oracle、End-to-end Success、Evaluation Set、Cost 與統計比較|
|How would you improve tool-use efficiency?|Tool Schema、Trace Analysis、Thrashing、Ablation|
|How do you construct context for million-line repositories?|Hybrid Retrieval、AST / Graph、Reranking、Budget、Compaction|
|How do you detect agent regressions?|Paired Benchmarks、Slice Analysis、Holdout、Hidden Tests|
|Why does an agent pass tests but still fail users?|Test Coverage、Environment Gap、Incorrect Oracle、Partial Success|
|What would you do if agents hang after 50 tool calls?|End-to-end Tracing、Stream Lifecycle、Tool Dispatcher、Recovery|
|How do you safely improve production agents?|Feature Flags、Canary、Telemetry、State Compatibility、Rollback|
|When would you fine-tune instead of changing the harness?|Root Cause Attribution、Model vs System Bottleneck、Training Data Quality|

以一個重要問題為例：

Interviewer: Your new prompt improved solve rate from 65% to 72%. How do you know it is a real improvement and not noise?

一個完整的回答應涵蓋：

> I would evaluate both versions on the same frozen task set, using identical environments, budgets and independently maintained test oracles. I would use paired comparisons and confidence intervals to quantify uncertainty, then break down performance by task category, language and difficulty.
> 
> I would also examine cost, latency, tool failures and regressions. Finally, I would validate the improvement on a holdout dataset and use a controlled production rollout to check whether offline gains translate into user outcomes.

中文重點就是：

不能只證明 Benchmark Score 上升，要證明這個提升是真實、可重現、沒有重大副作用，而且能改善 Production 使用體驗。

# 十二、這個職位和其他 LLM / Agent Engineer 最大差別

|職位|主要研究 / 工程對象|主要交付|
|---|---|---|
|Applied AI Engineer, Enterprise|利用 LLM 建立企業應用|RAG、Agents、Workflow、Enterprise Production|
|Applied AI Engineer, Codex Core Agent|改善 Coding Agent 在真實任務中的行為|Higher Solve Rate、Better Tools / Context、Evals、Production Reliability|
|AI Systems Engineer, Codex Agents|Core Harness、Sandbox、Orchestration、Inference / Runtime|Reliable Agent Execution Infrastructure|
|LLM Research Engineer|Model Training / Post-training|更強的 Model Capabilities|
|LLM Inference Engineer|Serving、GPU、KV Cache、Batching|更有效率的 Model Inference|

實際職責會有重疊。

例如 Applied AI Engineer 可能需要修改 Harness Code，AI Systems Engineer 也可能需要做 Evals。差別在主要負責的結果和技術深度重心。

在官方職缺中，Applied AI Engineer 更強調 Agent Behavior、真實任務成效、Research Collaboration 和 Feedback Loops；AI Systems Engineer 則更強調 Harness Runtime、Sandbox、Orchestration、系統效能與底層可靠性。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

+1

## 最後：我會如何建議你準備這個職位？

以你的 Python Computer Vision、Hardware Automation、複雜 Repository、測試與 AWS 系統開發經驗為起點，我會建議你做一個可以量化成果的 Coding Agent Evaluation & Reliability Platform，而不是再做一個普通的 Chatbot Demo。

尤其你可以利用熟悉的 Autofocus、Camera Capture、Motion Control、Image Processing、UI、Authentication 等工作流程，建立一組具真實複雜度的 Coding Tasks（使用可授權的內部程式或經過合成、去識別化的版本）。

專案最有價值的三個成果會是：

1. Agent Evaluation Harness：能在隔離 Repository 中執行多個版本的 Agent，使用獨立 Oracle 評分，輸出 Solve Rate、Latency、Cost、Regression Report。
    
2. Context + Tool Strategy Experiment：真正實作兩三種 Retrieval 與 Tool-use 策略，並透過 Ablation 證明差別。
    
3. Production Failure Analysis：能夠從 Traces 分辨 Model Failure、Context Failure、Tool Failure 與 Harness Failure，並把失敗案例轉成可重複的 Regression Tests。
    

這樣在 Senior 面試時，你就不是只說自己會使用 Codex 或 Claude Code，而是可以具體展示：

「我建立了一套 Coding Agent Evaluation System，在 500 個固定任務上分析失敗、改善 Context Retrieval 和 Tool Usage，透過 Regression Testing 驗證成功率及成本，最後以可控方式部署新版 Agent。」

這類完整的實驗與交付，比單純展示 Prompt Engineering 技巧更貼近這個職位的要求。

另外，OpenAI 在 [Harness engineering: leveraging Codex in an agent-first world](https://openai.com/index/harness-engineering/) 一文中，描述了如何將 Repository 的架構規則、測試、UI、Logs 與 Metrics 都變成 Agent 可直接使用、驗證的資源。這篇文章特別值得深入閱讀，因為它清楚展示了從「讓 AI 寫程式」走向「設計讓 AI 能可靠交付軟體的環境」的工程思維。

![](https://www.google.com/s2/favicons?domain=https://openai.com&sz=32)

OpenAI

這個職位最核心的能力，可以總結為：從真實 Coding Task 的失敗中找出可驗證的原因，建立改善假設，修改 Model-facing Systems，透過嚴謹 Evals 證明有效，最後讓改善安全地進入 Production。
