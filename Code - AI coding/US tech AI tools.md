

|                                                     |     |
| --------------------------------------------------- | --- |
| [[#### 2026 US big tech companies AI coding tools]] |     |
|                                                     |     |
|                                                     |     |
|                                                     |     |

#### 2026 US big tech companies AI coding tools
```
請幫我詳細解說2026或最新在美國大科技公司software engineer or AI engineer, 使用Codex or Clade等工具, 會如何做設定或如何讓結果能系統化更好更優化, 並具體舉例以及一步步教學
```
# 2026 美國大型科技公司 Software Engineer / AI Engineer 如何系統化使用 Codex、Claude Code 與 AI Coding Agents

2026 年 10 月最新實務整理｜Senior / Staff Software Engineer、Applied AI Engineer、ML Engineer 適用

2026 年，使用 Codex、Claude Code、GitHub Copilot 等 AI Coding Tools，真正重要的已經不只是「Prompt 寫得好不好」，而是如何建立一套可重複使用、可測試、可追蹤、可協作，而且不容易破壞 Production 的 AI Software Engineering System。

以美國大型科技公司的 Senior / Staff Engineer 工作標準來看，值得學習的是將 AI Coding Agent 整合進整個 Software Development Lifecycle（SDLC）。

傳統 AI Coding 與系統化 Agent Engineering 的差別

一般使用方式

Developer → Prompt → AI 產生 Code → Developer 自行檢查

2026 系統化工程流程

Requirements / GitHub Issue / Acceptance Criteria

Context Engineering / Repo Instructions / Skills

Codex Agent

Claude Agent

Copilot Agent

Implementation / Unit Tests / Integration Tests

AI Review + Human Review + Security Checks

CI/CD → Staging → Approval → Production → Monitoring

這是依照公開工程實務整理的建議架構，不代表每一家科技公司都採用相同流程。

## 一、2026 年大型科技公司的實際做法

先區分兩個概念：

公開可驗證的公司實務：公司工程團隊實際發表過的流程、案例和工具設定。

建議的企業級最佳實務：將不同公司的經驗結合軟體工程原則，建立適合自己團隊的制度；不能直接視為 Google、Meta、Microsoft、Amazon 等公司的內部標準。

### 1. OpenAI：把經常重複的工程流程轉成 Skills + CI

OpenAI 在 2026 年 3 月公開介紹，如何在 OpenAI Agents SDK 的 Python 與 TypeScript repositories 中，使用：

- `AGENTS.md`：Repository 的共同工程規則。
    
- `.agents/skills/`：可重複執行的專業工作流程。
    
- GitHub Actions：讓相同的檢查在 CI 中執行。
    
- 驗證與 PR Review：將檢查工作變成正式開發流程的一部分。
    

報告指出，兩個 repositories 在 2025 年 12 月至 2026 年 2 月合計合併 457 個 PR，而前一個三個月期間為 316 個。

這是開發產出的觀察值，不能直接把增加幅度全部歸因於 Codex，但它提供一個很具體的工程案例：把開發知識與驗證步驟保存在 Repository，而不是分散在個人的聊天記錄中。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI Developers

+1

### 2. Anthropic：讓 Claude Code 在不同任務採用不同自主程度

Anthropic 於 2025 年 7 月公開描述內部 Data Infrastructure、Inference、Security、Product Engineering、RL Engineering 等團隊如何使用 Claude Code。

它們的工作方式包括：

- 讓 Claude 先了解陌生 Codebase，再開始修改。
    
- 使用 `CLAUDE.md` 保存工程知識。
    
- 自動產生 Unit Tests、修復簡單 Bug。
    
- 對原型和低風險工作給予較高的 Agent 自主權。
    
- 對核心業務邏輯和安全敏感功能維持工程師密切監督。
    
- 透過 Git Checkpoints 和 Rollback 控制錯誤變更。
    

其中值得注意的是，它們不是所有任務都採用「完全自動」。任務風險越高，需要越多驗證和人工監督。

![](https://www.google.com/s2/favicons?domain=https://www.anthropic.com&sz=32)

Anthropic

+1

### 3. GitHub / Microsoft 生態系：Repository Instructions + PR Review

GitHub Copilot 官方文件支援 Repository-wide Instructions、Path-specific Instructions 與 Agent Instructions。

例如，可以讓修改 Python backend、React frontend、Infrastructure as Code 時，自動讀取不同的工程規則，再結合 Pull Request Code Review。

這代表一個團隊可以在 Git 裡保存 AI Coding 的共同規範，不必每位工程師每次重新輸入相同的 Prompt。

![](https://www.google.com/s2/favicons?domain=https://docs.github.com&sz=32)

GitHub Docs

+1

### 4. 2026 年的一個重要轉變：不要無限制增加 Prompt

OpenAI 於 2026 年 9 月的工程文章特別指出，隨著模型能力提升，過度冗長、過度規定步驟的 Skills 與 `AGENTS.md`，有時反而限制模型判斷、增加 Context 負擔。

因此，最新方向不是把幾百頁 Master Prompt 每次全部傳給 Agent，而是：

精簡的共同規則 + 依任務載入專用 Skills + 需要時讀取技術文件 + 可執行的測試。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI Developers

這個觀念對大型 Python、Computer Vision、LLM、AWS 專案尤其重要。

## 二、Codex、Claude Code、GitHub Copilot 應該如何分工？

先比較三種主要工具。

|工具|2026 年主要用途|我建議的定位|
|---|---|---|
|OpenAI Codex|理解 Repo、修改多個檔案、執行測試、CLI Automation、Code Review、Skills|主要 Implementation Agent|
|Anthropic Claude Code|大型 Codebase 分析、設計、Debug、Refactoring、Hooks、Subagents|Implementation + 獨立分析與 Review|
|GitHub Copilot|VS Code 程式協助、Repository Instructions、GitHub Coding Agent、PR Review|IDE 開發協助與 PR 流程整合|

不需要三個都買或都使用。若已經習慣 Codex，也可以只用 Codex 建立完整的工程流程；Claude 和 Copilot 是可選的第二層工具。

最重要的不是讓三個 AI 同時寫同一份 Code，而是讓它們共用工程規範，但工作彼此隔離。

### 建議的設定架構

GitHub Repository — Single Source of Truth

Source Code / Issues / Requirements / Tests / Documentation

共享 Context

`AGENTS.md` · Architecture Docs · ADRs · Skills · Test Commands

Codex

config.toml

.agents/skills/

Sandbox + Approvals

Claude Code

CLAUDE.md

.claude/skills/

Settings + Hooks

Independent Branches / Worktrees

GitHub PR → Deterministic CI → Review → Merge

其中有個 2026 年的重要細節：新版 Claude Code 已支援直接載入 `AGENTS.md`，但若專案同時有 `CLAUDE.md`，預設規則可能優先載入後者。為了跨工具相容，我仍建議在 `CLAUDE.md` 明確引用 `AGENTS.md`，避免兩份規範互相矛盾。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

## 三、一步步在 Windows 11 + VS Code 設定

以下使用 Windows 11、GitHub、Python Repository 作為範例。這套設定對 Software Engineer、Computer Vision Engineer、Applied AI Engineer 都適用。

### Step 1 — 準備軟體和 Git 環境

需要 Windows PowerShell、Git、VS Code，以及至少一種 Coding Agent。

如果使用 Claude Code CLI，可以在 PowerShell 執行官方的 Windows 安裝命令：

```
winget install Anthropic.ClaudeCode
claude --version
```

或依照 [Claude Code 官方 Quickstart](https://code.claude.com/docs/en/quickstart) 使用 Native Installer。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

如果使用 Codex Windows App，且已經可以開啟 Repository，可以直接沿用，不一定要額外安裝 CLI。想使用 CLI 自動化時，再依照 [Codex CLI 官方文件](https://learn.chatgpt.com/docs/codex/cli) 安裝。

確認基本環境：

```
git --version
python --version
git status
```

建議 VS Code 安裝 Python、Pylance、GitHub Pull Requests、GitHub Copilot（若使用），以及自己選用的 Codex / Claude Code Extension。

### Step 2 — 每次開發建立獨立 Branch

以下用一個已存在的 Python Repository 為例：

```
cd "D:\Projects\AIInspection"

git status
git switch dev
git pull --ff-only origin dev

git switch -c feature/improve-autofocus
```

假設要修改 Autofocus，就只在 `feature/improve-autofocus` 進行；不直接修改 `main` 或共用的 `dev`。

開始之前應確認 Working Tree 是乾淨的。如果 `git status` 顯示尚未提交的變更，先處理、備份或提交，不要直接切換和覆蓋。

### Step 3 — 建立 AI Agent 專用的 Repository 結構

```
AIInspection/
│
├── AGENTS.md
├── CLAUDE.md
├── pyproject.toml
├── src/
│   ├── acquisition/
│   ├── autofocus/
│   ├── image_processing/
│   └── inference/
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── hardware/
│
├── docs/
│   ├── architecture.md
│   ├── adr/
│   └── runbooks/
│
├── .agents/
│   └── skills/
│       └── safe-code-change/
│           └── SKILL.md
│
├── .claude/
│   ├── settings.json
│   └── skills/
│
├── .codex/
│   └── config.toml
│
└── .github/
    ├── copilot-instructions.md
    ├── PULL_REQUEST_TEMPLATE.md
    └── workflows/
        └── python-ci.yml
```

這裡每一種檔案有不同責任：

|檔案|用途|
|---|---|
|`AGENTS.md`|所有 AI Coding Agents 的共同工程指示|
|`CLAUDE.md`|Claude Code 特有的規則，並引用共同規範|
|`config.toml`|Codex 模型、Sandbox、Approval 等執行設定|
|`settings.json`|Claude Code 權限、Hooks 等設定|
|`SKILL.md`|專門任務的可重複流程|
|`architecture.md`|系統設計、模組關係和限制|
|`docs/adr/`|Architecture Decision Records，記錄技術選擇的原因|
|`python-ci.yml`|自動化且可重現的驗證程序|

重要： 上面是建議的專案結構，不代表工具安裝後就會自動建立所有檔案。可以逐項新增，並隨專案成熟程度擴充。

### Step 4 — 撰寫最重要的 `AGENTS.md`

這份檔案應該告訴 Agent 這個專案的技術背景、允許修改範圍、測試方式和風險邊界。

以下是可以直接改寫使用的範本。

檔案：Repository root / AGENTS.md

```
# Project Engineering Instructions

## Architecture
- Python application for automated visual inspection.
- Acquisition controls cameras and motion devices.
- Image processing is separated from inference.
- Production hardware is not available in CI.
- See docs/architecture.md for module boundaries.

## Development Rules
- Inspect relevant code and tests before modifying files.
- Prefer small, focused changes over broad rewrites.
- Preserve existing public APIs unless explicitly requested.
- Do not silently change camera calibration or motion units.
- Do not add production dependencies without justification.

## Testing
- Add or update regression tests for behavior changes.
- Run relevant unit tests before reporting completion.
- Use mocked devices for normal automated testing.
- Hardware integration tests require explicit approval.

## Security
- Never expose secrets, tokens, or customer data.
- Do not run production deployment commands.
- Do not modify AWS IAM, infrastructure, or database
  migrations without an approved change plan.

## Completion Report
Provide:
1. Summary of changes
2. Files changed
3. Tests executed and results
4. Known limitations and untested areas
5. Risks and rollback instructions
```

這份檔案有幾個設計上的關鍵。

首先，它只寫所有任務都需要遵守的規則，不把 Autofocus、HDR、UNet、AWS Deployment 的每個技術細節全部塞進去。

其次，讓 Agent 能夠自行閱讀 `docs/architecture.md`，而不是每次都把整個架構手動貼進 Prompt。

第三，它要求 Agent 回報「實際執行的測試」，而不是只說「程式應該正常」。

還必須理解：`AGENTS.md` 是 Agent 的行為指示，不是作業系統層級的安全限制。真正的執行權限仍需由 Sandbox、Access Control、CI 和人工核准實施。

### Step 5 — Claude Code 設定 `CLAUDE.md`

在 Repository 根目錄建立：

檔案：CLAUDE.md

```
@AGENTS.md

## Claude Code Specific Guidance

- For architecture-impacting changes, begin in Plan Mode.
- Identify relevant source files and tests first.
- Prefer small, reviewable patches.
- Do not claim tests passed unless executed.
- Ask for approval before any production side effect.
```

`@AGENTS.md` 是 Claude Code 文件支援的 Import 語法，並非所有工具通用的 Markdown 功能。Windows 環境使用這種 Import 方式比依賴 Symlink 更方便。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

### Step 6 — 設定 Codex 的 Sandbox、Approval 和 Reasoning

Codex 支援個人與 Repository 層級的設定。

|設定範圍|Windows 路徑|
|---|---|
|個人設定|`%USERPROFILE%\.codex\config.toml`|
|專案設定|`AIInspection\.codex\config.toml`|

先在個人設定中建立安全的預設值：

```
# ~/.codex/config.toml

approval_policy = "on-request"
sandbox_mode = "workspace-write"
model_reasoning_effort = "medium"
web_search = "cached"
```

設定解釋：

`approval_policy = "on-request"`：在需要額外權限時可要求核准；不是所有命令都一定跳出詢問。

`sandbox_mode = "workspace-write"`：限制 Agent 的檔案寫入範圍，降低對其他目錄的影響。

`model_reasoning_effort = "medium"`：一般 Coding 任務先使用中等推理強度，需要處理複雜架構或 Debugging 時再提高。

`web_search = "cached"`：優先使用快取搜尋結果，降低直接接觸不受信任外部內容的風險。

這些設定均有官方文件支持；實際隔離能力仍受 Windows Sandbox 實作、管理政策和外部工具權限影響。

![](https://www.google.com/s2/favicons?domain=https://learn.chatgpt.com&sz=32)

ChatGPT Learn

初期不建議開啟 `danger-full-access` 或未受監督的全權操作。模型能力很強，不代表在任何情況下都應有無限制的系統權限。

在 Codex CLI 內還可以使用：

```
/status
/permissions
/model
/review
```

分別檢查目前狀態、執行權限、模型與 Code Review。

### Step 7 — Claude Code 設定 Permissions

建立 Repository 層級設定：

檔案：.claude/settings.json

```
{
  "$schema": "https://json.schemastore.org/claude-code-settings.json",
  "permissions": {
    "defaultMode": "default",
    "deny": [
      "Read(./.env)",
      "Read(./.env.*)",
      "Bash(git push *)",
      "Bash(aws *)"
    ]
  }
}
```

這是刻意保守的起始範例。

它限制 Agent 讀取常見的環境機密檔案，並防止直接使用 `git push` 和 AWS CLI。若需要部署，應使用單獨核准的 Deployment Workflow，而不是直接取消限制。

但要注意：這些規則是工具層級的權限控制範例，不能單靠幾條命令比對就確保所有資料外洩途徑都被封鎖。企業環境仍需搭配 Credentials Isolation、Network Egress Control 和 Managed Policies。

Claude Code 官方也區分個人設定、共用 Project Settings 與 Organization Managed Settings。企業政策應放在管理者可強制執行的位置，而不是只依賴可由開發者編輯的 JSON。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

+1

在 Claude Code CLI 內可使用：

```
/init
/permissions
/status
/memory
/plan
```

其中 `/init` 可以協助產生 Project Instructions，但要由工程師檢查內容是否正確；不要直接把自動生成的規範當成已驗證的事實。

## 四、使用 Skills：把常做的工作變成標準流程

這是 2026 年我認為最值得學習的部分之一。

### 什麼是 Skill？

假設一個 Engineer 每週都要求 Codex：

「先分析程式 → 找到 Bug → 加入 Regression Test → 修改程式 → 執行相關測試 → Review Diff → 回報結果。」

如果每次都重新貼相同的長 Prompt，很容易遺漏步驟，也很難讓團隊使用同一套規範。

Skill 就是將這個流程保存成一個可重複使用的工作單元。

例如：

Bug Fix Skill

Reproduce → Test → Fix → Verify

Code Review Skill

Diff → Risks → Tests → Findings

ML Evaluation Skill

Model → Dataset → Metrics → Regression

Release Skill

Artifact → Staging → Canary → Rollback

### Step 8 — 建立第一個 Codex Skill

在 VS Code 建立：

`.agents/skills/safe-code-change/SKILL.md`

```
---
name: safe-code-change
description: Implement and verify a focused code change
  when fixing a bug or changing existing behavior.
---

# Safe Code Change

## Goal
Make a minimal, tested, reviewable code change.

## Workflow

1. Identify the affected modules and existing tests.
2. Explain the expected behavior and current failure.
3. Reproduce the bug with a regression test when feasible.
4. Implement the smallest compatible change.
5. Run relevant tests, lint, and static checks.
6. Review the Git diff for unintended modifications.
7. Summarize results, risks, and rollback procedure.

## Rules
- Never change production infrastructure as part
  of a routine code fix.
- Never modify unrelated modules unnecessarily.
- Do not weaken existing tests just to make them pass.
- Report tests that could not be executed.
- Stop and escalate if a change requires
  unavailable hardware or privileged credentials.
```

然後在 Codex 中輸入：

```
Use $safe-code-change to fix the camera
timeout bug.

First inspect the existing implementation and tests.
Reproduce the failure before modifying the code.
Do not change motion control or AWS configuration.
```

Codex 可以讀取這個 Skill，並按 Workflow 協助完成任務。Skill 也可以搭配 `scripts/`、`references/`、`assets/`，將穩定、可自動化的步驟交由程式執行。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI Developers

Claude Code 的類似機制是 `.claude/skills/<skill-name>/SKILL.md`。可以使用相同的工作流程設計，但需要依各自支援的路徑與觸發方式配置；不要假設所有 Agent 自動共用同一份 Skill。

### Step 9 — 不要只依賴 Skills，加入確定性的 Verification

這裡要區分：

AI 會遵循的流程指示與程式真正執行的驗證。

例如 Skill 裡寫了「Run unit tests」，不代表 Unit Tests 已通過。真正的驗證應使用 `pytest`、`ruff`、`mypy`、Build Scripts 等工具。

對 Python 專案，可以先建立一套標準驗證命令：

```
python -m pytest tests/unit -q
python -m ruff check src tests
```

這兩行是範例，假設專案有 `tests/unit`、`src`，且已安裝相關工具。

建議逐步加入：

|檢查|目的|能發現的問題|
|---|---|---|
|Ruff / Lint|程式碼靜態檢查|未使用變數、部分潛在 Bug|
|Type Checker|類型檢查|Interface、Type 不一致|
|Unit Tests|單一函式／模組|邏輯錯誤與 Regression|
|Integration Tests|模組之間互動|API、Database、Queue 整合問題|
|Security Scan|安全檢查|部分弱點、Secrets、依賴風險|
|Performance Benchmarks|效能驗證|Latency、Memory Regression|
|Domain Evals|AI 系統品質|Accuracy、Recall、Hallucination 等|

沒有任何單一檢查能證明程式一定正確。這些檢查應分層使用。

## 五、建立 GitHub CI/CD，讓 AI 產生的 Code 不能直接進 Production

### Step 10 — 建立 GitHub Actions

在 `.github/workflows/python-ci.yml` 放入以下範例：

```
name: Python CI

on:
  pull_request:
    branches: [dev, main]

permissions:
  contents: read

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[dev]"

      - name: Lint
        run: |
          python -m ruff check src tests

      - name: Unit tests
        run: |
          python -m pytest tests/unit -q
```

這份範例預設專案有可安裝的 `pyproject.toml`、`dev` dependencies、`src` 和 `tests/unit`；若既有 Repository 使用 Conda 或其他 Dependency Management，必須先修改相應命令。實務部署還應將 GitHub Actions 和重要 Dependencies 固定到經審核的版本或 Commit SHA。

當工程師或 AI Agent 建立 Pull Request，GitHub Actions 就會執行驗證。

#chatgpt-mermaid-_r_h3s_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_h3s_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_h3s_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_h3s_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_h3s_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_h3s_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_h3s_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_h3s_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_h3s_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_h3s_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h3s_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h3s_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_h3s_ p{margin:0;}#chatgpt-mermaid-_r_h3s_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_h3s_ .label text,#chatgpt-mermaid-_r_h3s_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .node rect,#chatgpt-mermaid-_r_h3s_ .node circle,#chatgpt-mermaid-_r_h3s_ .node ellipse,#chatgpt-mermaid-_r_h3s_ .node polygon,#chatgpt-mermaid-_r_h3s_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .rough-node .label text,#chatgpt-mermaid-_r_h3s_ .node .label text,#chatgpt-mermaid-_r_h3s_ .image-shape .label,#chatgpt-mermaid-_r_h3s_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_h3s_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .rough-node .label,#chatgpt-mermaid-_r_h3s_ .node .label,#chatgpt-mermaid-_r_h3s_ .image-shape .label,#chatgpt-mermaid-_r_h3s_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_h3s_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_h3s_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h3s_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h3s_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_h3s_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_h3s_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h3s_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h3s_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_h3s_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_h3s_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h3s_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_h3s_ .icon-shape,#chatgpt-mermaid-_r_h3s_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_h3s_ .icon-shape p,#chatgpt-mermaid-_r_h3s_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_h3s_ .icon-shape .label rect,#chatgpt-mermaid-_r_h3s_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h3s_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_h3s_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_h3s_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_h3s_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_h3s_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_h3s_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_h3s_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_h3s_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h3s_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_h3s_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h3s_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_h3s_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_h3s_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_h3s_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_h3s_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .node rect,#chatgpt-mermaid-_r_h3s_ .node circle,#chatgpt-mermaid-_r_h3s_ .node ellipse,#chatgpt-mermaid-_r_h3s_ .node polygon,#chatgpt-mermaid-_r_h3s_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_h3s_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_h3s_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_h3s_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_h3s_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h3s_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Agent changes codeFeature branchPull requestCI: Lint, unit tests, securitychecksAll required checks pass?Repair and rerunAI-assisted code reviewHuman approval /CODEOWNERSMerge to devStaging / integration validationRelease approved?Investigate / rollbackCanary deploymentProduction monitoringNoYesNoYes

在 GitHub Repository 的 Settings → Branches / Rulesets 中，設定 `dev` 和 `main` 的保護規則：

要求 PR 通過指定的 CI Checks，限制直接 Push，對重要檔案要求 Code Owner Review；Production Release 另外設置 Environment Approval。

要注意： 上面的 YAML 只執行 CI，不會自動建立所有 Branch Protection、Review 或 Production Deployment Gate。這些必須在 GitHub 和部署系統分別配置。

### Step 11 — AI Code Review 不等於 Human Approval

例如 Codex 開發某個新功能後，可讓 Claude Code 或 GitHub Copilot 執行第二次檢查。

Code Review Prompt：

```
Review the changes in this pull request.

Focus on:
- Functional correctness
- Concurrency and resource management
- Security and authorization
- Error handling and recovery
- Backward compatibility
- Missing edge-case tests
- Performance regressions

Do not modify any files.

Report only actionable findings.
For each finding provide:
- Severity
- File and line location
- Failure scenario
- Recommended fix
- Test that would catch the bug

If no substantial issues are found, say so explicitly.
Do not invent issues to fill the report.
```

由不同模型進行 Review，有機會發現第一個 Agent 忽略的問題，但兩者也可能犯相同錯誤。所以第二模型不能取代 Unit Tests、Security Checks 或人類對關鍵邏輯的理解。

Codex 官方支援透過 `openai/codex-action@v1` 將 Codex 納入 GitHub Actions，Claude Code 也有 GitHub Actions 整合。要避免把 API Keys 暴露給不受信任的 PR 程式碼，應依照官方建議做權限隔離。

![](https://www.google.com/s2/favicons?domain=https://learn.chatgpt.com&sz=32)

ChatGPT Learn

+1

## 六、具體案例 A：Senior Software Engineer 如何用 Agent 修復 Backend Bug

假設你在一家美國 SaaS 公司，負責一個擁有數千名企業客戶的 Python FastAPI 服務。

現在遇到的問題：

> 一個 API 在外部服務 Timeout 時，可能重複提交相同交易。

我們來看系統化使用 AI Coding Agent 的完整開發流程。

### Phase 1 — 需求和風險分類

Engineer 不會先要求「Fix the bug」。

他應該先確認：

|項目|定義|
|---|---|
|Expected Behavior|同一個 Idempotency Key 不應重複建立交易|
|Current Failure|Timeout 後重試，可能建立重複資料|
|Impact|重複交易、資料不一致|
|Risk|High — 財務與資料完整性|
|Acceptance Criteria|並發重試及 Timeout 下仍不能建立重複交易|
|Test Requirements|Unit + Database Integration + Concurrency|
|Rollback|回復前一版本；資料修復須有獨立方案|

這一步決定 AI 是否能自行處理，以及後續需要多少檢查。

### Phase 2 — 要求 Codex 先調查

```
Task: Investigate duplicate transaction creation
after an API timeout.

Read the transaction endpoint, service layer,
database constraints, and retry handling.

Do not modify code yet.

Return:
1. Relevant call graph
2. Root cause hypotheses
3. Existing protection against duplicate requests
4. Race conditions
5. Proposed regression tests
6. Minimal remediation options
7. Compatibility and migration risks
```

此時 Codex 的工作是閱讀現有 Code，建立 Control Flow 與 Root Cause Hypotheses。

Senior Engineer 需要審查診斷是否正確，例如是否真的需要 Database Unique Constraint，而不是只加一個 Python Lock。

### Phase 3 — 寫 Regression Tests

先測出原本會失敗的情境：

測試情境與預期結果

|   |   |
|---|---|
|同一 Key 連續兩次呼叫|只有一筆交易|
|同一 Key 並發請求|只有一筆交易|
|第一次請求 Timeout|重試不重複建立交易|
|同一 Key 搭配不同 Payload|回傳明確衝突結果|
|Database Transaction Failure|不留下不一致的狀態|

特別注意：只測「連續兩次 API 呼叫」通常不夠。真正危險的是並發以及 Network Failure 導致的未知提交狀態。

### Phase 4 — Implementation

Agent 依照已核准的設計修改：

API Endpoint、Service、Persistence Layer，以及必要的 Schema / Migration。

若需要 Database Migration，就必須另外評估部署順序、回滾和歷史資料；不能因為 AI 認為 Unique Constraint 比較安全，就直接更改 Production Database。

### Phase 5 — Verification 與 PR

要求 Codex 執行測試並生成 PR Summary：

```
Execute the approved implementation.

Run focused unit tests and database integration
tests in the test environment.

Inspect the complete diff.

Report:
- Actual test results
- Files changed
- Migration impact
- Failure recovery
- Unverified assumptions

Do not push, merge, or deploy.
```

AI 產生的變更只有在 CI、Code Review 和必要的實際測試通過後，才符合可以合併的條件。

這就是 Senior Engineer 與只用 AI 自動寫 Code 最大的差別：Senior Engineer 負責定義系統在各種失敗條件下應有的行為，AI 協助加速實作和驗證。

## 七、具體案例 B：Senior AI / Computer Vision Engineer 如何系統化使用 Codex

用一個更接近實際工業 AI 系統的案例。

假設你的 Repository 是 `ImagingLibWatch`，系統包含：

- 多相機 Acquisition。
    
- Zaber Motion Control。
    
- Keyence CL-3000 Autofocus。
    
- HDR Image Processing。
    
- UNet Segmentation。
    
- AWS Model Training、Deployment、Authentication Pipeline。
    

這種系統跟一般網站不同。AI 寫錯 Python Code，後果不只是 API 回傳錯誤，也可能涉及設備移動、拍攝失敗、錯誤影像資料，甚至 Production 分析結果不可靠。

因此，我會採用 Hardware-Aware Agent Development Workflow。

### Step 1 — 將專案拆成不同風險區域

|模組|風險|Agent 執行建議|
|---|---|---|
|Python Utility|Low|可在隔離環境自動開發、測試|
|HDR / Image Processing|Medium|Unit Tests + Golden Image Tests|
|UNet Inference|Medium|Dataset Evals + Performance Tests|
|Camera Acquisition|High|Mock Tests → Hardware Validation|
|Motion Control|Very High|模擬測試 + 人工設備操作核准|
|Authentication Policy|High|Regression + Expert Review|
|AWS IAM / Deployment|Very High|PR、CI、安全審查、部署核准|

其中 Motion Control 不能只依靠 `AGENTS.md` 裡面寫著「Do not move hardware」。

應該讓一般開發 Agent 執行環境根本沒有 Production Motion Device 權限，只有經核准的 Hardware Integration Test 才能取得對應權限。

### Step 2 — 建立子模組規範

例如在 `control/autofocus/` 下建立適用於該目錄的 `AGENTS.md`。

```
# Autofocus Engineering Rules

## Device Safety
- Never issue real motion commands in unit tests.
- Use simulated Keyence and Zaber interfaces.
- Preserve motion-axis and unit conventions.
- Treat invalid laser readings as invalid evidence.

## Algorithm Validation
- Test missing and out-of-range distance readings.
- Test search boundaries and maximum attempts.
- Verify position limits before movement.
- Preserve image and device traceability.

## Verification
- Run mocked autofocus tests first.
- Compare with known reference captures when available.
- Report changes in focus accuracy and runtime.
- Hardware validation requires explicit approval.
```

Codex 會依所在路徑載入適用的 `AGENTS.md`；Claude Code 也有階層化的專案規範機制，但需依前述版本與設定確認載入方式。

### Step 3 — 實際要求：改善 Keyence Autofocus

假設系統目前遇到問題：

當 Keyence OUT1 沒有有效讀值時，Autofocus 可能無法找到合適的 Z Position。

傳統 Prompt：

> Please fix autofocus when Keyence OUT1 is missing.

這樣太模糊。Agent 可能隨意修改馬達移動策略、Fallback 位置或其他相關程式。

更好的 Prompt：

```
Task: Improve autofocus recovery when Keyence
OUT1 does not return a valid measurement.

Repository: ImagingLibWatch

Phase 1: Investigation only.
Do not edit code.

Analyze:
- Keyence reading and validity checks
- Current autofocus control flow
- Existing search limits
- Motion safety constraints
- Unit conversion
- Existing tests

Phase 2: Propose design.

Requirements:
- Invalid readings must never be treated as valid.
- Recovery attempts must be bounded.
- No hardware access during automated unit tests.
- Existing successful autofocus behavior must
  remain backward compatible.

Phase 3: Implementation after design approval.

Phase 4: Verification.
Use mocked Keyence and Zaber interfaces.
Include boundary and failure-path tests.
Report focus-error changes against references.

Never execute real motion without explicit approval.
```

這個 Prompt 的好處是把幾個不同責任分開：

Agent 負責調查與實作，工程師核准設備移動和演算法變更，測試程式驗證行為，實際硬體測試確認模擬是否符合真實設備。

### Step 4 — 建立 Autofocus Evaluation Dataset

這一步是 AI Engineer 比一般 Software Engineer 更需要注意的部分。

除了 Unit Tests，我會保留一個小型、可重現的 Autofocus Benchmark。

例如：

|測試類型|驗證內容|
|---|---|
|Valid Reading|正常讀值時能得到正確位置|
|Missing OUT1|無有效讀值時正確執行 Recovery|
|Out of Range|超出量測範圍不被誤認為有效|
|Boundary Search|不超出允許的馬達行程|
|Noise|讀值抖動時演算法是否穩定|
|Hardware Disconnect|逾時或中斷後安全終止|
|Ground Truth|結果與人工最佳焦點的差距|
|Runtime|搜尋時間和擷取影像數量|

假設要衡量 Autofocus Position Error：

\[ e_i=|z_{\text{predicted},i}-z_{\text{ground truth},i}| \]

衡量重點不只 Mean Error，還包括 P95 Error、成功率、Failure Recovery Rate 和搜尋時間。若量測的是不同坐標軸，必須先使用正確坐標轉換，不能直接把不同軸的位置相減。

此後每次 Codex 修改 Autofocus Algorithm，都在相同的 Benchmark Cases 上比較新舊版本。

範例：Autofocus 版本比較

以下數字純屬示意，不是真實測試結果。

|   |   |   |
|---|---|---|
|Metric|Baseline|Candidate|
|Success Rate|92%|97%|
|Median Error|0.18 mm|0.11 mm|
|P95 Error|0.55 mm|0.32 mm|
|Median Runtime|2.2 s|2.5 s|

在這個例子裡，Candidate 提高成功率和位置準確度，但速度變慢。是否值得部署，需要依系統的真實產能、安全需求和驗收標準決定。

這就是 AI-assisted AI Engineering：AI 加速演算法開發，但品質由可重現的數據判斷。

### Step 5 — Production 部署不能讓 Agent 自行決定

#chatgpt-mermaid-_r_h5o_{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;fill:rgb(13, 13, 13);}@keyframes edge-animation-frame{from{stroke-dashoffset:0;}}@keyframes dash{to{stroke-dashoffset:0;}}#chatgpt-mermaid-_r_h5o_ .edge-animation-slow{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 50s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_h5o_ .edge-animation-fast{stroke-dasharray:9,5!important;stroke-dashoffset:900;animation:dash 20s linear infinite;stroke-linecap:round;}#chatgpt-mermaid-_r_h5o_ .error-icon{fill:rgb(243, 243, 243);}#chatgpt-mermaid-_r_h5o_ .error-text{fill:rgb(13, 13, 13);stroke:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .edge-thickness-normal{stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .edge-thickness-thick{stroke-width:3.5px;}#chatgpt-mermaid-_r_h5o_ .edge-pattern-solid{stroke-dasharray:0;}#chatgpt-mermaid-_r_h5o_ .edge-thickness-invisible{stroke-width:0;fill:none;}#chatgpt-mermaid-_r_h5o_ .edge-pattern-dashed{stroke-dasharray:3;}#chatgpt-mermaid-_r_h5o_ .edge-pattern-dotted{stroke-dasharray:2;}#chatgpt-mermaid-_r_h5o_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h5o_ .marker.cross{stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h5o_ svg{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:16px;}#chatgpt-mermaid-_r_h5o_ p{margin:0;}#chatgpt-mermaid-_r_h5o_ .label{font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .cluster-label text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .cluster-label span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .cluster-label span p{background-color:transparent;}#chatgpt-mermaid-_r_h5o_ .label text,#chatgpt-mermaid-_r_h5o_ span{fill:rgb(13, 13, 13);color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .node rect,#chatgpt-mermaid-_r_h5o_ .node circle,#chatgpt-mermaid-_r_h5o_ .node ellipse,#chatgpt-mermaid-_r_h5o_ .node polygon,#chatgpt-mermaid-_r_h5o_ .node path{fill:rgb(222, 234, 251);stroke:rgb(83, 154, 248);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .rough-node .label text,#chatgpt-mermaid-_r_h5o_ .node .label text,#chatgpt-mermaid-_r_h5o_ .image-shape .label,#chatgpt-mermaid-_r_h5o_ .icon-shape .label{text-anchor:middle;}#chatgpt-mermaid-_r_h5o_ .node .katex path{fill:#000;stroke:#000;stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .rough-node .label,#chatgpt-mermaid-_r_h5o_ .node .label,#chatgpt-mermaid-_r_h5o_ .image-shape .label,#chatgpt-mermaid-_r_h5o_ .icon-shape .label{text-align:center;}#chatgpt-mermaid-_r_h5o_ .node.clickable{cursor:pointer;}#chatgpt-mermaid-_r_h5o_ .root .anchor path{fill:rgb(143, 143, 143)!important;stroke-width:0;stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h5o_ .arrowheadPath{fill:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h5o_ .edgePath .path{stroke:rgb(143, 143, 143);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .flowchart-link{stroke:rgb(143, 143, 143);fill:none;}#chatgpt-mermaid-_r_h5o_ .edgeLabel{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_h5o_ .edgeLabel p{background-color:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h5o_ .edgeLabel rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h5o_ .labelBkg{background-color:rgba(252, 252, 252, 0.5);}#chatgpt-mermaid-_r_h5o_ .cluster rect{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .cluster text{fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ .cluster span{color:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ div.mermaidTooltip{position:absolute;text-align:center;max-width:200px;padding:2px;font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";font-size:12px;background:rgb(243, 243, 243);border:1px solid rgba(0, 0, 0, 0.1);border-radius:2px;pointer-events:none;z-index:100;}#chatgpt-mermaid-_r_h5o_ .flowchartTitleText{text-anchor:middle;font-size:18px;fill:rgb(13, 13, 13);}#chatgpt-mermaid-_r_h5o_ rect.text{fill:none;stroke-width:0;}#chatgpt-mermaid-_r_h5o_ .icon-shape,#chatgpt-mermaid-_r_h5o_ .image-shape{background-color:rgb(252, 252, 252);text-align:center;}#chatgpt-mermaid-_r_h5o_ .icon-shape p,#chatgpt-mermaid-_r_h5o_ .image-shape p{background-color:rgb(252, 252, 252);padding:2px;}#chatgpt-mermaid-_r_h5o_ .icon-shape .label rect,#chatgpt-mermaid-_r_h5o_ .image-shape .label rect{opacity:0.5;background-color:rgb(252, 252, 252);fill:rgb(252, 252, 252);}#chatgpt-mermaid-_r_h5o_ .label-icon{display:inline-block;height:1em;overflow:visible;vertical-align:-0.125em;}#chatgpt-mermaid-_r_h5o_ .node .label-icon path{fill:currentColor;stroke:revert;stroke-width:revert;}#chatgpt-mermaid-_r_h5o_ .node .neo-node{stroke:rgb(83, 154, 248);}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node rect,#chatgpt-mermaid-_r_h5o_ [data-look="neo"].cluster rect,#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node polygon{stroke:url(#chatgpt-mermaid-_r_h5o_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].swimlane.cluster rect{filter:none;}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node path{stroke:url(#chatgpt-mermaid-_r_h5o_-gradient);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node .outer-path{filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node .neo-line path{stroke:rgb(83, 154, 248);filter:none;}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node circle{stroke:url(#chatgpt-mermaid-_r_h5o_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].node circle .state-start{fill:#000000;}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].icon-shape .icon{fill:url(#chatgpt-mermaid-_r_h5o_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h5o_ [data-look="neo"].icon-shape .icon-neo path{stroke:url(#chatgpt-mermaid-_r_h5o_-gradient);filter:drop-shadow( 1px 2px 2px rgba(185,185,185,1));}#chatgpt-mermaid-_r_h5o_ .node text{font-size:14px;font-weight:600;letter-spacing:normal;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_h5o_ .edgeLabels text{font-size:13px;font-weight:600;letter-spacing:-0.08px;fill:rgb(65, 65, 65);}#chatgpt-mermaid-_r_h5o_ .node tspan[font-weight="normal"],#chatgpt-mermaid-_r_h5o_ .edgeLabels tspan[font-weight="normal"]{font-weight:600;}#chatgpt-mermaid-_r_h5o_ .edgeLabel .label rect{opacity:1;rx:13px;ry:13px;fill:rgb(252, 252, 252);stroke:rgb(219, 219, 219);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .node rect,#chatgpt-mermaid-_r_h5o_ .node circle,#chatgpt-mermaid-_r_h5o_ .node ellipse,#chatgpt-mermaid-_r_h5o_ .node polygon,#chatgpt-mermaid-_r_h5o_ .node path{fill:rgb(243, 243, 243);stroke:rgba(0, 0, 0, 0.1);stroke-width:1px;}#chatgpt-mermaid-_r_h5o_ .node rect{rx:16px;ry:16px;}#chatgpt-mermaid-_r_h5o_ .node.mermaid-decision .label-container{fill:rgb(249, 249, 249);stroke:rgb(219, 219, 219);stroke-dasharray:2,2;}#chatgpt-mermaid-_r_h5o_ .edgePaths .flowchart-link{stroke:rgb(143, 143, 143);stroke-width:1px;stroke-linecap:round;stroke-linejoin:round;}#chatgpt-mermaid-_r_h5o_ .marker{fill:rgb(143, 143, 143);stroke:rgb(143, 143, 143);}#chatgpt-mermaid-_r_h5o_ :root{--mermaid-font-family:-apple-system-body,ui-sans-serif,-apple-system,system-ui,"Segoe UI",Helvetica,"Apple Color Emoji",Arial,sans-serif,"Segoe UI Emoji","Segoe UI Symbol";}Codex implements changeUnit tests with mock devicesPass?Fix implementationOffline benchmark evaluationEngineer code reviewControlled hardware validationSafety and quality gates pass?Reject or reviseRelease approvalCanary / staged rolloutProduction monitoringNoYesNoYes

這裡區分兩種能力：

Coding Agent 可以自動化的部分：Code Analysis、Implementation、Mock Unit Tests、Offline Benchmark、PR Summary。

仍應由經授權的人員或受控系統決定的部分：真實設備移動、敏感 AWS 權限、資料庫變更、Production Release、安全驗收。

## 八、具體案例 C：Applied AI Engineer 開發 RAG / LLM Agent

再看另一個很常見的 2026 年職務：Senior Applied AI Engineer。

假設任務是設計企業內部文件助手，能使用 RAG 回答問題，也能呼叫 Business Tools。

你希望 Codex 或 Claude 幫忙新增一個功能：

> 使用者詢問內部文件時，系統必須根據使用者的文件權限過濾 Retrieval Results，避免讀取未授權內容。

這裡不能只要求 Agent「實作 ACL Filter」。

### 系統開發應分成五個層次

|層次|Agent 要完成的工作|如何驗證|
|---|---|---|
|Design|定義 Authorization Boundary|Architecture / Security Review|
|Retrieval|加入 ACL-aware Filtering|Authorized / Unauthorized Cases|
|Generation|僅使用允許的 Retrieval Evidence|Groundedness、Citation Tests|
|Tool Calling|Enforcement of Tool Permissions|Unauthorized Tool Tests|
|Production|Latency、Cost、Tracing|P95、Cost、Security Monitoring|

例如建立 `tests/security/test_document_acl.py`，至少包含：

使用者 A 只能讀取 Department A 文件；Department B 文件即使在 Vector Search 最相似，也不能被返回。若文件權限更新，舊 Cache 不能繼續洩漏已撤銷權限的文件。Agent 的 Tool Calls 也不能繞過這個授權檢查。

特別重要的是：權限檢查必須由可信任的 Backend / Data Access Layer 強制執行，而不是交給 LLM 自己判斷哪些文件可以使用。

接著讓 Codex 產生 Integration Tests，再用另一個獨立 Agent Review Authorization Logic。最後使用 Prompt Injection Dataset 測試惡意文件能否誘導系統洩漏資料。

這就是 2026 年 Applied AI Engineer 所需要的 Agent Engineering 與 Security Engineering 交集。

## 九、如何讓 Codex / Claude 的結果持續變好？建立 Agent Evals

這一部分是從個人開發進一步提升到 Staff Engineer 或 Engineering Team Lead 的關鍵。

很多人會憑感覺比較：

「這次 Claude 好像比較厲害。」

「Codex 修改的檔案比較少，應該比較好。」

這不夠可靠。

你需要建立 Coding Agent Evaluation Benchmark。

OpenAI 在 2026 年 1 月公開了針對 Agent Skills 的系統化 Evaluation 方法：將 Prompt、Execution Trace、Artifacts 和 Checks 組成一組可量化的測試流程。

![](https://www.google.com/s2/favicons?domain=https://developers.openai.com&sz=32)

OpenAI Developers

### Step 1 — 建立代表性的 Golden Task Set

例如收集 30 個你們團隊真實遇過、且已有正確解決方案或清楚驗收條件的工作。

示例：30 個 Coding Agent Golden Tasks

0369Bug FixFeaturesRefactoringTest GenerationAI/MLSecurity

示範資料集配置，不代表實際企業統計

所有候選 Agent 都在相同的固定任務、相同的初始 Commit、相近的執行環境和相同驗收條件下測試。

### Step 2 — 設計 Evaluation Metrics

|Metric|定義|為什麼重要|
|---|---|---|
|Task Success Rate|完成且通過驗收的任務比例|最直接的正確性|
|Test Pass Rate|通過自動化測試比例|基本品質|
|First-pass Success|不需人工修正即通過比例|效率|
|Regression Rate|引入既有功能退化的比例|可靠性|
|Security Violation Rate|違反安全規則的比例|企業風險|
|Median / P95 Completion Time|任務完成時間|開發速度|
|Cost per Accepted Task|完成有效任務的總成本|經濟效益|
|Human Review Minutes|人工審查與修正時間|真正的團隊負擔|

Task Success Rate 應看真正的 Acceptance Tests，不是看 Agent 自己宣稱有沒有完成。

### Step 3 — 比較 Codex 與 Claude

假設得到以下結果：

純示範數值，並非 Codex 或 Claude 的實際 Benchmark

0%25%50%75%100%Codex + basic rulesCodex + skills + CIClaude + rules + CI

範例結果；成功率與時間應搭配信賴區間及任務分類分析

這只是展示 Evaluation 的形式；30 個任務的樣本較小，實際選型需要檢查失敗案例、統計不確定性、成本與任務類型。

重點不是追求某個 AI 的總分最高，而是確認你做了哪些設定，真的讓結果有所改善。

例如，你可以做 A/B Comparison：

同一模型、同一批 Tasks、同一種執行環境，A 組只有基本 Instructions，B 組加入新 Skill，然後比較任務成功率、執行成本、工具呼叫次數、測試覆蓋以及 Review 時間。

這樣才知道某個 Skill 真的有價值，還是只是增加了 Context 和 Token Cost。

## 十、進階最佳化：Context、Token、Multi-Agent、Hooks、MCP

### 1. Context Engineering：只載入完成任務需要的資訊

不應每次把所有程式、Documentation、歷史聊天記錄和 Master Specification 全部塞進 Context Window。

建議把 Context 分四層：

|Layer|內容|載入方式|
|---|---|---|
|Global|一般編碼安全與工程原則|User / Organization Config|
|Repository|Architecture、Build、Coding Rules|`AGENTS.md`|
|Task|Issue、Acceptance Criteria、相關檔案|每次任務 Prompt|
|Workflow|專業操作流程與工具|Skill 按需載入|

例如，修改 UNet Segmentation，不必每次都給 Agent 所有 AWS Infrastructure 細節；除非它確實會影響 Training Artifact 或 Model Deployment。

這就是 Progressive Disclosure：只在需要的時候載入詳細資訊。

### 2. Model Routing：不同任務使用不同推理成本

可以建立團隊內部策略：

|Task|建議推理程度|
|---|---|
|Rename / Simple Documentation|Low|
|小型 Bug Fix|Low / Medium|
|一般 API Feature|Medium|
|複雜跨模組 Refactoring|High|
|Distributed System Design|High|
|Security-sensitive Logic|High + Independent Review|
|Production Failure Analysis|High + Logs / Tracing|

這不是指 High Reasoning 一定比較好，而是不要讓所有簡單工作都支付最昂貴的推理成本。

真正的 Model Selection 應透過自己的 Evals 來確認。

### 3. Multi-Agent：不要讓多個 Agent 同時修改同一工作目錄

假設你想並行處理 Autofocus 和 RAG。

可以使用 Git Worktrees，讓兩個 Agent 有獨立的 Working Directory：

```
# Run from a clean repository checkout on dev
git worktree add -b feature/af-fallback `
  ..\AIInspection-af dev

git worktree add -b feature/rag-eval `
  ..\AIInspection-rag dev

git worktree list
```

第一個 Agent 在 `AIInspection-af` 修改 Autofocus。

第二個 Agent 在 `AIInspection-rag` 修改 RAG Evaluation。

兩者共享同一個 Git Repository 的歷史資料，但各自有自己的工作目錄與 Branch，可以避免直接覆蓋對方尚未提交的檔案。最終仍需經過 PR 和 Integration Tests 處理變更衝突。

對 Windows + VS Code 多台電腦開發尤其實用。

### 4. Hooks：把一部分檢查變成事件觸發

Claude Code 支援 Hooks，例如：

|Hook|用途|
|---|---|
|PreToolUse|在工具執行前檢查或阻止操作|
|PostToolUse|在工具執行後進行額外處理|
|SessionStart|初始化開發 Session|
|Stop|Agent 完成回應時執行指定檢查邏輯|

例如，可以設計 Hook 在 Python 檔案修改後安排 Formatter / Linter，或在特定 Tool Call 前進行安全檢查。

但不建議一開始把完整測試都放進每一次 File Edit Hook。大型專案可能因此不斷重跑測試，讓開發速度變慢。比較合理的是使用快速、精確的 Local Hooks，加上正式 CI 的完整驗證。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

### 5. MCP：給 Agent 受控的外部工具

MCP（Model Context Protocol）可以把 GitHub、Database、Issue Tracker、Cloud 或其他 Tool 暴露給 Agent。

但在企業環境中，我建議按風險劃分：

|工具類型|建議|
|---|---|
|Read GitHub Issues|可以給唯讀授權|
|Search Documentation|可以給唯讀授權|
|Read Test Logs|可以給唯讀授權|
|Create Development PR|使用受限身分與審核|
|Modify Production Database|不給一般 Agent 直接權限|
|Change AWS IAM|需獨立權限與人工核准|
|Release to Production|經部署平台的 Approval Gate|

MCP 是 Tool Integration Protocol，不是 Authorization System。不能因為工具經由 MCP 呼叫，就假設操作一定安全。

## 十一、每天實際使用的標準 Prompt

以下這個 Prompt 可以作為 Codex 或 Claude Code 日常工程任務的起點。

Reusable Engineering Task Template

Copy-and-use task specification

Copy template

```
{template}
```

此範本是完整模式。對簡單的小型 Bug Fix，可以縮短 Scope 和流程；不需要每次填寫所有欄位。

## 十二、如果要在自己的團隊真正導入，我建議的順序

不要一開始就讓 Codex 或 Claude 自動控制完整 AWS、Hardware、CI 和 GitHub。

可以採用以下四個階段：

第 1 階段：建立共同工程規範

建立精簡 `AGENTS.md`、`CLAUDE.md`、Architecture Documentation。先讓 Codex 和 Claude 能正確理解 Codebase，並確認修改前有清楚的 Git Baseline。

第 2 階段：建立可驗證的開發流程

加入 Unit Tests、Mock Hardware、Regression Dataset、Lint、PR Template 和 GitHub CI。先用低風險 Bug Fix 驗證流程能正常運作。

第 3 階段：建立 Skills 與多 Agent 分工

將 Bug Fix、Code Review、ML Evaluation、Release Preparation 等常見任務轉成 Skills。使用 Worktrees 隔離並行工作，讓 AI Review 與人類 Review 各有明確責任。

第 4 階段：量化最佳化與受控部署

建立 Golden Task Set，比較模型、Instructions 和 Skills。確認成功率、成本、Review 時間及 Regression 有改善後，再擴大自動化範圍。

對你的 `ImagingLibWatch` / Moonlight 這類同時包含 Computer Vision、設備控制、Authentication 與 AWS 的專案，我會優先導入 Repository-level Instructions → Skills → Mock/Regression Tests → CI/PR Gates，再逐步讓 Codex 參與受控的 AWS Pipeline 驗證。

這比先建立一個可以自主修改整個 Repository 的 Super Agent 更容易維護，也比較容易讓新的 Junior Engineer 加入。

## 十三、Senior / Staff Engineer 應該掌握的核心觀念

2026 年的大型科技公司 AI Coding 工作流程，最值得學習的不是某個特定模型、某個隱藏 Prompt，或某個神奇設定。

而是以下這個分工：

|層次|人類 Engineer 的責任|Coding Agent 的責任|
|---|---|---|
|Requirements|定義問題和成功條件|協助分析需求|
|Architecture|決定技術取捨|提出設計選項|
|Implementation|審查關鍵設計|撰寫與修改程式|
|Testing|決定驗收標準|產生並執行測試|
|Security|設計並強制權限邊界|協助找出風險|
|Deployment|核准高風險操作|在授權範圍內執行|
|Evaluation|決定 Metrics 和品質門檻|蒐集與分析結果|
|Ownership|承擔系統可靠性責任|提供工程協助|

最終建議：把 AI Coding 從「聊天式程式生成」升級成「有規範、有工具、有測試、有權限、有驗收標準的工程系統」。

如果是為 2026 年美國 Senior / Staff Software Engineer 或 Applied AI Engineer 職位做準備，能夠實際展示這套系統如何提高交付速度、減少 Regression、控制 Production Risk，會比只展示自己能用 Codex 或 Claude 很快完成一個功能，更能反映資深工程能力。

官方延伸閱讀

[OpenAI — Using Skills to Accelerate OSS Maintenance](https://developers.openai.com/blog/skills-agents-sdk) · [OpenAI — Rethinking Skills and Prompts (2026)](https://developers.openai.com/blog/rethinking-skills-and-prompts-for-gpt-6-astra) · [Claude Code — Project Memory and Instructions](https://code.claude.com/docs/en/memory) · [Claude Code — Settings and Permissions](https://code.claude.com/docs/en/settings) · [GitHub — Repository Custom Instructions](https://docs.github.com/en/copilot/how-tos/copilot-on-github/customize-copilot/add-custom-instructions/add-repository-instructions)
