
|                                                                    |     |
| ------------------------------------------------------------------ | --- |
| [[#### Anthropic Code Review]]                                     |     |
| [[#### 詳細解釋Anthropic (how the claude code team does code review)]] |     |
| [[#### 核實文章論點]]                                                    |     |

#### Anthropic Code Review
```
Anthropic Code Review
```
當 AI 寫了 90% 的程式碼，Anthropic 團隊 Code Review 怎麼做？

先說我的觀點：

一、這篇讓我最有共鳴的，是 Anthropic 用 Revert Rate / Incident Rate 作為品質護欄。不追求零錯誤，而是重視快速發現、快速復原，並透過數據驗證流程調整是否真的有效。

二、「You land it, you own it」的文化，搭配 Revert、On-call Rotation、Incident Review 等機制，才讓降低人工 Review 比重這件事變得可控。

以下整理幾個我覺得值得參考的重點。完整的七層審查架構放在圖中，原文連結放在第一則留言。
Anthropic 的 monorepo，每天合併的程式碼變更，從 5 月約 2,500 個，成長到 9 月超過 30,000 個。


1. 重新定義 Review 的目標

與其問「有沒有人看過這段 Code」，不如問「這個變更可能出什麼錯？哪一道檢查最適合抓到它？」​他們追蹤每個 PR 的人工 Approve 數量、等待時間，同時以 [#RevertRate](https://www.facebook.com/hashtag/revertrate?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、[#IncidentRate](https://www.facebook.com/hashtag/incidentrate?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R) 作為品質護欄。結果從 6 月到 9 月，Claude Code 的 Merged PR 數量增加超過一倍，需要 Revert 的比例反而下降超過 40%。

2. Review 不再只是 PR 的最後關卡

許多檢查在 PR 建立之前就開始了。Claude 會先執行測試、Build、截圖比對，再由另一個使用全新 Context 的 Subagent 審查 Diff。讓問題盡可能在開發階段就被發現，而不是全部留到 PR 才處理。

3. 多層防護，而不是依賴單一關卡

[#CI](https://www.facebook.com/hashtag/ci?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、AI Code Review、Security Review、人工審核、[#MergeQueue](https://www.facebook.com/hashtag/mergequeue?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)、[#FeatureFlag](https://www.facebook.com/hashtag/featureflag?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)，以及 [#Rollback](https://www.facebook.com/hashtag/rollback?__cft__[0]=AZiS-grtY58IS3mcgN-_0oJ1dRrbAseCE67u7n8fJXXT_H-qRIIuvgY5MKHk6kb3y7nYqgKfx0nBe6KWPfrGvqgqRLOrMVogrt6hAMxRzrnyfVxcggjPNGQlNumIH6B95u-wKWHW9IR3Zb4XgTVXSj_sSGPKJsLTyHyhtReYe7F9YxJM7dDbdD8QcK5qCpo_VqE&__tn__=*NK-R)，共同構成防護機制。每一層都可能漏掉問題，但透過不同機制交叉檢查，可以降低問題一路進入正式環境的機率。

4. AI Review 需要降噪
多個 Agent 平行找 Bug，再由另一組 Agent 交叉驗證，過濾掉不夠可靠的發現與純風格類意見。留下的 Review 意見都需要處理：修正問題，或說明理由後關閉。否則 AI Reviewer 產生大量低價值意見，反而會增加工程師的負擔。

5. 根據風險決定人工審查程度

不是每個 PR 都需要相同數量的 Reviewer。透過分類器依照變更內容判斷風險，低風險變更可以不需要額外人工 Approve，高風險變更則需要更多人審查。同時也有明確限制：AI 的 Approve 不能取代人的 Approve；新 Commit 進來後，原本的核准也會失效。

6. 避免 AI Review 無限循環

曾經有 PR 被 AI 作者與 AI Reviewer 來回修改超過 50 輪。後來透過限制 Review 輪數、只審查新的 Diff、避免重複提出相同問題，讓平均 Review 輪數從 2.08 降到 1.32。

7. 把 Incident 變成 Evaluation Dataset

將過去造成事故或需要 Revert 的 PR 整理成評測集，每天驗證 AI Reviewer 的能力。未來不論是調整 Prompt、工具或模型，都可以透過相同的評測標準，確認是否真的改善。

8. 人應該專注在更需要判斷力的事情

例如需求是否合理、架構方向、高風險變更、證據判讀，以及最終的責任歸屬。這些事情，不會因為 AI 能寫出更多程式碼就自動解決。

![[852661240_10234812014070512_5672273076250322684_n.jpg]]





#### 詳細解釋Anthropic (how the claude code team does code review)
```
請幫我詳細解釋Anthropic (how the claude code team does code review)是如何做多層程式碼審查架構, 並具體舉例說明譬如我做了一個code changes, 但還沒git push一直到完整完成code review的每一個流程實際的步驟.
```
# Anthropic 的多層 Code Review 架構：從 Local Code Changes 到 Production 的完整實作流程

根據 Anthropic 在 2026 年 10 月 15 日正式發表的工程文章 [How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/?utm_source=chatgpt.com)，Claude Code 團隊已經建立一套以 AI Agents、Automated Verification、Security Review、Risk-Based Human Approval 與 Rollback 為核心的多層程式碼審查架構。

這套方法不只是讓 Claude 幫忙看程式碼，而是重新設計整個 Software Development Lifecycle（SDLC）的品質控制流程。

我會分成四個部分說明：

1. Anthropic 官方架構：每一層的任務、技術實現及判斷標準。
    
2. 完整實際案例：你修改了一段 Python 程式碼，還沒有 `git commit`、`git push`，如何一步步完成所有 Code Review。
    
3. 實際設定方式：如何設定 Claude Code、`CLAUDE.md`、`REVIEW.md`、GitHub Actions、CODEOWNERS 與 Branch Protection。
    
4. 2026 年的最佳實務與常見誤解：哪些是 Anthropic 真正採用的流程，哪些需要額外開發，哪些不應該讓 AI 自行決定。
    

以下根據 Anthropic 官方文章、Claude Code 文件，以及 GitHub 官方文件整理，而不是把第三方推測當成 Anthropic 的內部事實。

## 一、Anthropic 為什麼需要重新設計 Code Review？

Anthropic 公開指出，Claude 目前撰寫了約 90% 的 Claude Code 專案程式碼。在 2026 年 6 月至 9 月期間，其 Claude Code 程式庫的已合併 PR 數量增加了一倍以上，但被 Revert 的比例反而下降超過 40%。

這些是 Anthropic 公布的內部結果，並不代表其他團隊導入後一定會得到相同改善。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

+1

傳統開發流程通常是：

Engineer writes code

Commit / Push

Open PR

Human Review

Merge

傳統流程：Code Review 主要發生在 PR 建立以後。

但是如果 AI 可以在很短時間產生數千行 Code，Human Reviewer 可能變成整條 Pipeline 的瓶頸。

因此 Anthropic 改變了審查策略：

讓機器負責大量、反覆、可驗證的檢查；讓人類負責高風險決策、架構設計、產品方向與最終責任。

這不等於完全取消 Human Review，也不等於讓 Claude 自己寫完程式碼之後自己宣布通過。

## 二、Anthropic 官方的多層 Code Review Architecture

官方文章以六層檢查加上最後的快速偵測與 Rollback 描述整體架構。若把 Rollback 也算作一層，可以將它整理為七個階段。

Developer + Claude Code

Local working directory / Uncommitted changes

Layer 1 — Pre-PR Verification

Local

Unit Test · Build · Fresh-context AI Review · Local Security Review

Commit → Push → Pull Request

Layer 2 — AI Code Review

Parallel bug-hunting agents + independent verification

Layer 3 — Security Review

Authorization · Injection · Data Leakage

Layer 4 — Required CI

Lint · Types · Tests · Build · Platform checks

Layer 5 — Human Approval

0 / 1 / 2 人，依變更風險決定

Layer 6 — Merge Queue + Gradual Release

重新執行 CI → Merge → Feature Flag → Canary / Staged Release

Layer 7 — Production Monitoring + Rollback

Telemetry → Incident Detection → Disable / Revert → New Regression Tests

根據 Anthropic 2026/10/15 官方工程文章重繪。Layer 2–4 可以平行執行，不一定在時間上逐一等待前一層結束。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

### 各層的主要任務

|Layer|檢查機制|主要目的|是否需要 Git Push|
|---|---|---|---|
|1|Local Test + Fresh-context AI|提前找出功能與設計錯誤|不需要|
|2|Multi-Agent Code Review|找出 Logic Bugs、Regression|一般需要 PR|
|3|Security Review|找出安全漏洞及越權風險|Local 不需要；PR Review 需要|
|4|CI Pipeline|執行確定性的自動測試|GitHub CI 需要|
|5|Human Approval|確認架構、風險與責任|通常需要 PR|
|6|Merge Queue|驗證合併後的整體程式碼|需要|
|7|Monitoring + Rollback|發現並控制漏網的問題|部署後|

最重要的區別是：Layer 1 可以在完全不 Push Code 的情況下進行；但 Layer 2–6 的完整 GitHub PR 流程，仍然需要把程式碼提交到遠端 Branch。

所以你的目標如果是「在 git push 之前就完成所有審查」，可以把大量 AI Review 與測試提前到本機，但不可能在尚未建立遠端 PR 時就完成所有 GitHub PR Gate、正式 Human Approval 和 Merge Queue。

### Anthropic 的 Human Approval 規則

官方文章對 Human Approval 的設計相當特別：

|Risk Level|變更範例|Anthropic 內部要求|
|---|---|---|
|Low|Documentation、文案、受 Feature Flag 控制的低風險新增功能|可不要求額外人類核准|
|Medium|一般產品功能、Cache、Auto-updater、System Prompt|1 人|
|High|最高風險程式碼、特定敏感路徑|2 人|

風險等級由變更內容決定，而不是單純依據作者職級。另外，較輕量的核准路徑只開放給 Trusted Authors；非 Trusted Author 仍需其他人員核准。

Anthropic 也將指揮 AI Agent 寫程式的人視為需要承擔審查責任的人，AI 本身不能算作一名 Human Approver。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

還有一個非常重要的差別：

公開的 Claude Code Review 產品只發現並張貼問題，不會直接替你的 PR 按 Approve 或自動阻止合併。 Anthropic 內部的「未處理審查問題不得合併」，必須透過額外的 Repository Workflow 與 Merge Policy 實現。

也就是說，安裝 Claude Code Review 並不代表你的 GitHub 專案自動擁有 Anthropic 全部七層保護。

![](https://www.google.com/s2/favicons?domain=https://support.claude.com&sz=32)

Claude Help Center

## 三、實際案例：從尚未 Git Push 的 Code Changes 開始

以下使用一個接近你開發的 `ImagingLibWatch` Authentication System 的例子。

假設你在 Windows 11 的 VS Code 中，修改了 Python Authentication Pipeline 的程式碼，這次的需求是：

> Authentication Pipeline 必須正確處理 Missing Evidence。如果某個 Watch 缺少部分影像，系統可以繼續分析現有證據，但不能因為所有證據都缺失，就誤判成 Authentic。

你已經修改程式碼，但目前：

- 尚未 `git commit`。
    
- 尚未 `git push`。
    
- 尚未建立 GitHub PR。
    
- 想先用 Claude Code 進行多層審查。
    

### Step 0 — 確認 Git 狀態與建立 Feature Branch

首先，在 VS Code 的 PowerShell Terminal 執行：

```
cd 'D:\Provenance Laboratories projects\ImagingLibWatch'

git status
git branch --show-current
git diff --stat
git diff
```

假設你看到：

```
On branch dev

Changes not staged for commit:
    modified: authentication/decision.py
    modified: authentication/api.py

Untracked files:
    helper/tests/test_authentication_decision.py
```

這代表你的修改目前只存在本機。

注意：`git diff` 不會顯示 Untracked Files，所以新建的測試檔也必須另外檢查。

如果你目前在 `dev`，建議先建立 Feature Branch：

```
git fetch origin dev
git switch -c feature/auth-missing-evidence
```

`git fetch` 只更新遠端追蹤資訊，不會直接合併遠端程式碼，也不會清除你的修改。

`git switch -c` 則是以目前 Branch 為基礎建立新的 Branch。正常情況下，尚未 Commit 的修改會跟著保留。

如果原本已經在正確的 Feature Branch，則不用再次建立。

本案例使用 `dev` 當 PR 的 Target Branch，而不是 `main`。實際命令仍應以你的 Repo 工作流程為準。

### Step 1 — 明確定義正確的 Expected Behavior

假設你寫了以下程式：

```
def authenticate_watch(evidence):    usable = [        item for item in evidence        if item.get("score") is not None    ]    if not usable:        return {            "decision": "Original",            "confidence": 1.0        }    # Other statistical / Bayesian calculations...
```

這裡存在一個可能非常嚴重的 Bug。

當所有影像都缺失時，`usable` 是 Empty List，但程式卻回傳：

```
{    "decision": "Original",    "confidence": 1.0}
```

也就是：

完全沒有證據，卻給出最高信心的 Original 結論。

假設這次產品需求已經明確規定「無證據不得通過 Authentication」，正確的行為應該類似：

```
if not usable:    return {        "decision": "InsufficientEvidence",        "confidence": None    }
```

這裡的 `InsufficientEvidence` 是示範用的 Watch-level 決策狀態，不是要取代你原本的八種 Component Expert Labels。實際應該依你的 Authentication Policy 設計。

這個例子可以看出 AI Code Review 必須同時理解：

- Requirement：沒有足夠證據不能認證。
    
- Implementation：目前的 Python Code 實際做了什麼。
    
- Business Logic：Component Label 和 Watch-level Decision 的差異。
    
- Failure Impact：錯誤結果可能造成假錶被認定為真錶。
    

單純檢查 Python Syntax、Typing 或 Formatting，完全找不到這類錯誤。

### Step 2 — Layer 1A：執行本機 Deterministic Verification

在呼叫 Claude Code Reviewer 前，先執行可以得到明確 Pass/Fail 結果的檢查。

首先：

```
git diff --check
```

這可以抓出部分 Whitespace Error 等 Diff 問題。

接著執行你現有的 Python Test Suite：

```
python -m pytest -q helper/tests/test_authentication_decision.py
```

上述測試路徑為案例示範，請使用 Repo 真正存在的測試檔案與 Python Environment。

新增一個 Regression Test：

```
def test_missing_evidence_never_authenticates():    result = authenticate_watch([])    assert result["decision"] == "InsufficientEvidence"    assert result["confidence"] is None
```

對前面的錯誤實作執行測試，預期會出現：

```
FAILED test_missing_evidence_never_authenticates

Expected: InsufficientEvidence
Actual:   Original
```

這就是一個有用的驗證結果：它不是 AI 說「應該沒問題」，而是實際測試證明行為不符合需求。

如果修改涉及相關程式，也應執行：

```
python -m pytest -q
```

以及 Repo 已設定的 Lint、Type Check、Build。

針對你的 Imaging System，我還建議測試四類條件：

|Test Category|範例|
|---|---|
|Normal Case|所有預期影像與 Features 均存在|
|Partial Missing|部分 Macro / Micro Images 缺失|
|Complete Missing|完全沒有可用 Evidence|
|Failure Handling|S3 存取失敗、DB Timeout、Invalid Feature|

這些測試應由 Requirements 決定正確結果，而不是讓 Claude 根據既有實作自行猜測。

### Step 3 — Layer 1B：執行 Claude Code Local Review

現在進入 Claude Code：

```
claude
```

接下來的 `/...` 指令，是輸入在 Claude Code 的互動介面內，不是直接當成 PowerShell Command 執行。

先輸入：

```
/diff
```

你可以看到當前 Working Tree 的改動。

接著輸入：

```
/code-review high
```

根據目前的 [Claude Code Local Review 官方文件](https://code.claude.com/docs/en/code-review-local?utm_source=chatgpt.com)，`/code-review` 可以檢查目前 Branch 的程式碼變更，包括尚未 Commit 的修改。

它會啟動具有獨立 Context 的 Review Subagent，而不只是讓原本負責寫 Code 的 Agent 自己重新讀一次答案。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

主要指令的差別如下：

|Claude Code 指令|功能|
|---|---|
|`/diff`|查看尚未提交的修改|
|`/code-review`|本機 Correctness Code Review|
|`/code-review high`|使用較深入的審查層級|
|`/code-review --fix`|審查後嘗試直接修改程式碼|
|`/security-review`|專門檢查 Security 問題|
|`/simplify`|尋找可簡化、重用、改善的程式碼|
|`/verify`|嘗試執行程式，驗證真實行為|
|`/code-review ultra`|進一步使用 Cloud Multi-Agent Deep Review，視帳號和環境而定|

指令可用性和行為會隨 Claude Code 版本調整。`/review` 在目前的新版文件中也是 `/code-review` 的別名。

建議第一輪不要加 `--fix`。 先讓 Reviewer 只提出問題，再由你決定哪些問題合理，才不會讓自動修正意外擴大修改範圍。

### Step 4 — Layer 1C：使用 Fresh-context Reviewer 檢查 Requirements

這是我認為最值得加入你現有開發流程的一步。

即使已經執行 `/code-review high`，你仍然可以額外要求一個獨立 Reviewer 對照原始需求。

在 Claude Code 輸入以下 Prompt：

Independent Requirement Review Prompt

Copy

Use an independent fresh-context subagent to review my current uncommitted and staged changes against origin/dev. Do not modify files, commit, push, or create a PR. Review: - Correctness and original requirements - Missing-evidence behavior - Bayesian decision semantics - Error handling and regression risks - Backwards compatibility - Relevant callers, consumers, and existing tests For every finding, provide: 1. Severity 2. File and line number 3. Concrete failure scenario 4. Proposed regression test Distinguish newly introduced bugs from pre-existing issues. Do not report speculative issues as confirmed bugs. Explicitly state what could not be verified.

這樣的 Review 比單純輸入：

`Review my code.`

更有價值，因為你明確定義了 Reviewer 的工作範圍、驗證標準和輸出格式。

Anthropic 也在官方 Best Practices 中推薦 Writer/Reviewer 分離，避免同一個 Agent 因為了解自己原本的實作意圖，而忽略實際結果的問題。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

### Step 5 — 模擬第一次 Code Review 的結果

假設 Reviewer 找到以下問題（以下是示範結果，並非實際執行你的 Repo 後的發現）：

Finding 1 — Missing Evidence Returns False Authentication

`authentication/decision.py` · Critical

When all evidence is missing, the function returns Original with confidence 1.0.

Impact: A watch without valid evidence may receive an incorrect positive authentication decision.

Recommendation: Use an explicit insufficient-evidence outcome and add a regression test.

Finding 2 — Missing Confidence Value May Break Existing Callers

`authentication/api.py` · Major

An existing caller may assume every authentication result has a numeric confidence value.

Impact: Introducing `None` without updating response schemas or consumers may cause a TypeError or serialization failure.

Recommendation: Review the API schema, UI consumers, and downstream persistence logic.

這兩個 Findings 很適合說明 Multi-layer Review 的價值。

第一個 Reviewer 從 Domain Logic 找出問題；第二個 Reviewer 從 System Integration 找出問題。

你可以把第一個 Bug 修好，卻因為新的 `confidence=None` 破壞另一個模組。

這也是為什麼 AI Code Review 應檢查整個相關 Codebase，而不是只看 `git diff` 的幾行程式。

### Step 6 — Layer 1D：Local Security Review

如果本次修改也新增 Authentication Result 查詢 API，假設你寫了：

```
def get_analysis(watch_id, current_user):    return repository.get_by_watch_id(watch_id)
```

這段程式碼可能有重大安全問題。

雖然 `current_user` 已經登入，但是查詢沒有使用其 Tenant 或 Organization 權限。

如果系統允許不同客戶查詢自己的 Watch 資料，就可能發生：

```
User A → Watch A → Allowed

User B → Watch A → Also allowed (BUG)
```

你可以在 Claude Code 執行：

```
/security-review
```

並額外要求：

```
Review the modified authentication API for
broken object-level authorization, tenant
isolation, S3 object access, sensitive data
exposure, and privilege escalation.

Do not change code.
Provide a concrete reproduction test.
```

修正的方向應該是：

```
def get_analysis(watch_id, current_user):    return repository.get_by_watch_and_tenant(        watch_id=watch_id,        tenant_id=current_user.tenant_id    )
```

但還需要確保 Tenant 身分來自可信任的 Authentication Context，而不是客戶端自己提交的參數，並處理不存在或未授權資料的回應方式。

這類問題不能只靠 Unit Test 的 Happy Path 驗證，還需要 Negative Authorization Tests。

有一個與你的 Git Workflow 特別相關的細節：目前 `/security-review` 文件描述它會以 `origin` 的 Default Branch 為比較基準，因此如果你的實際 PR 是 `feature → dev`，而 GitHub Default Branch 是 `main`，就應另外明確檢查相對 `origin/dev` 的 Diff，並確認所有尚未 Commit 的檔案也確實被涵蓋。不要假設預設 Security Review 一定會完全符合你的 PR 比較範圍。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

+1

### Step 7 — 修正後重新驗證，但避免無限 AI Review

完成修正後，再執行：

```
git diff --check
python -m pytest -q helper/tests/test_authentication_decision.py
```

接著重新執行：

```
/code-review high
```

第二輪應該聚焦於：

- 上次發現的問題是否已修復。
    
- 是否因修正引入新的 Regression。
    
- 原本失敗的 Regression Test 現在是否通過。
    

Anthropic 特別提到一種稱為 Review Convergence 的問題：AI 作者與 AI Reviewer 可能持續互相修改，一個內部 PR 曾經在一晚超過 50 輪審查。

團隊後來採用三個重要規則：

1. 超過約三輪後，不再盲目重試，而是重新檢查設計方向。
    
2. 後續審查逐步改成只檢查新的 Diff。
    
3. 同一類問題應一次修正所有實例，並避免重複報告已提出的 Findings。
    

這不是要降低審查標準，而是避免把時間花在沒有實際新資訊的審查循環。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

### Step 8 — 最後檢查，然後 Git Commit

到目前為止，我們仍然沒有執行任何 `git push`。

當本機測試、AI Correctness Review 與 Local Security Review 都完成之後：

```
git status
git diff --check
git diff
```

確認新檔案、Modified Files、測試結果以及可能的 Secrets 都正確之後，進行 Staging：

```
git add -p
```

或者逐一指定檔案：

```
git add authentication/decision.py
git add authentication/api.py
git add helper/tests/test_authentication_decision.py
```

檢查真正即將 Commit 的內容：

```
git diff --cached
git diff --cached --check
```

然後：

```
git commit -m "Fix authentication missing-evidence handling"
```

此時 Code 已經 Commit 到本機，但還沒有 Push 到 GitHub。

值得注意的是，`git commit` 不是 Code Review 的結束，而是下一階段 PR Review 的開始。

## 四、從 Git Push 到 PR Merge：真正的 Multi-Agent Review

### Step 9 — Push Feature Branch

當本機審查已達標準：

```
git push -u origin feature/auth-missing-evidence
```

這只會 Push 你的 Feature Branch，不會直接修改遠端 `dev` 或 `main`。

### Step 10 — 建立 Pull Request

可以在 GitHub 網站操作，也可以透過 GitHub CLI：

```
gh auth status
```

確認 GitHub CLI 已安裝且完成登入後：

```
gh pr create `
  --base dev `
  --head feature/auth-missing-evidence `
  --title "Fix authentication missing-evidence handling" `
  --body "Prevent false positive authentication decisions when evidence is missing. Includes regression tests and rollback notes."
```

現在 GitHub 上就會有一個：

```
PR #123 (示範編號)

Base: dev
Head: feature/auth-missing-evidence
```

建議 PR Description 至少包含：

|欄位|內容|
|---|---|
|Summary|本次修改的目標|
|Requirements|預期行為與限制|
|Tests|實際執行的測試及結果|
|Risk|是否涉及安全、資料正確性、API 相容性|
|Deployment|是否需要 Migration、Feature Flag、特殊部署|
|Rollback|如果失敗，如何恢復|
|Limitations|哪些測試因缺乏硬體或資料尚未執行|

### Step 11 — Layer 2：GitHub Multi-Agent Code Review

假設你的 Team 已經啟用 Anthropic 的 Code Review for Claude Code。

在 GitHub 建立 PR 後，系統可以自動啟動 AI Review。

根據 Anthropic 的公開說明，Review System 的核心是並行尋找問題，再由其他 Agents 驗證問題是否真實。

PR Diff + Repository Context + PR Intent

Reviewer A

Logic / Correctness

Reviewer B

Edge Cases / Regression

Reviewer C

System Context

Independent Finding Verification

Other agents verify / challenge candidate findings

Deduplicate + Severity Ranking

Filter unsupported or redundant findings

GitHub Inline Review Comments

這是根據官方描述繪製的概念架構。Reviewer A/B/C 的角色標籤是說明用途，並非 Anthropic 公開的實際內部 Agent 名稱或完整 Prompt。

Anthropic 表示，多個 Agents 會平行找 Bugs，另一組 Agents 對候選 Findings 進行驗證和投票。只有通過足夠驗證的結果才會張貼，純粹格式或 Cleanup 類問題可能被過濾。

這降低了 False Positives，也減少 Human Reviewer 需要閱讀的低價值評論。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

在你的 PR 案例中，可能會產生以下分類：

|Finding|結果|
|---|---|
|無證據時回傳 Original|Important — 必須修復|
|`confidence=None` 可能破壞 UI|Important — 需驗證|
|舊版函式命名不一致|Pre-existing 或 Nit|
|建議把一個簡單函式拆成三個 Class|可能只是主觀建議，未必需要修正|

其中 Pre-existing 特別重要，它代表問題原本就存在，不一定是這次 PR 引入的。

### Step 12 — Layer 3：PR Security Review

這一層獨立於一般的 Bug Review，目的是檢查：

- Authentication / Authorization。
    
- Injection、敏感資料洩漏與 Insecure API。
    
- 錯誤的 AWS IAM 或 Resource Access。
    
- 不安全的 Credentials 處理。
    
- Privilege Escalation、跨 Tenant 存取。
    

Anthropic 內部的流程要求 Bug Review 與 Security Review 的未解決 Findings 都要處理，才允許合併。

你可以選擇修正問題，或在確定是 False Positive 時提出合理解釋並 Resolve Thread；修正後重新審查也可能自動關閉已解決的問題。

這種阻擋能力要靠 GitHub Review Thread Resolution 與相關自動化規則實現，不是 `/security-review` 指令本身會自動對 GitHub 施加 Merge Block。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

+1

### Step 13 — Layer 4：GitHub Actions CI

現在 GitHub CI 應該執行確定性的測試，而非單靠 LLM 判斷正確性。

例如：

```
PR #123
│
├── Python Lint                  PASS
├── Python Type Check            PASS
├── Unit Tests                   PASS
├── Integration Tests            PASS
├── Authentication Regression    PASS
├── Security Scan                PASS
└── Package / Build              PASS
```

你的 `ImagingLibWatch` 還包含 Windows Hardware、相機、Zaber Motion、Keyence Sensors，因此建議另外區分：

Software CI 可以在 GitHub Hosted Runner 上執行；Hardware-in-the-loop Test 則必須在具備實際設備的機器上執行。

不要把未執行 Hardware Test 的 PR 標示成全部驗證完成。

### Step 14 — Layer 5：Human Review

接下來才由 Risk Policy 決定是否需要額外 Human Approval。

假設這個 PR 只是修改 UI 顯示文字，可能是 Low Risk。

但這次修改直接影響：

```
Raw Evidence
    ↓
Statistical Evidence
    ↓
Bayesian Authentication
    ↓
Watch-level Decision
```

因此我會將它歸類為 High Risk，建議至少由 Authentication Domain Owner 和 Backend/System Owner 兩個不同角色審查。

這是對你專案的建議分類，不是 Anthropic 對你的專案做出的官方判斷。

Human Reviewer 應該關心的是：No Evidence 是否真的不會變成 Authentic？新結果格式是否破壞 API？Bayesian Priors、Thresholds 和 Expert Policy 是否仍符合設計？測試證據能否支持這些結論？

如果你自己是唯一一位 Senior Engineer，也可以先由你負責完整 Domain Review，但不能把同一個 Claude Agent 的兩次同意當成兩名獨立人類核准。

### Step 15 — 修正 PR Findings、Push 新 Commit

假設 GitHub Review 找到新問題。

你在本機：

```
# Modify code and tests

python -m pytest -q
git diff --check

git add authentication/decision.py
git add helper/tests/test_authentication_decision.py

git commit -m "Address authentication review findings"

git push
```

因為你之前已經設定過：

```
git push -u origin feature/auth-missing-evidence
```

所以現在直接 `git push` 就可以更新同一個遠端 Feature Branch。

GitHub PR 也會自動更新，不需要重新開一個 PR。

新的 Commit 可能觸發：

- GitHub Actions 重新測試。
    
- Claude Code Review 重新審查。
    
- 原有 Findings 自動標記為已解決。
    
- Human Approval 失效並需要重新批准（視 Repo Policy）。
    

Anthropic 的內部流程會對核准綁定特定版本的變更；如果新增未經認可的修改，就可能要求重新批准。GitHub 可以透過 Dismiss stale pull request approvals when new commits are pushed 實作相近的機制。

![](https://www.google.com/s2/favicons?domain=https://docs.github.com&sz=32)

GitHub Docs

### Step 16 — Layer 6：Merge Queue

假設有兩個 PR：

```
PR #123: Change authentication decision schema
PR #124: Update API response handling
```

兩個 PR 單獨測試時都通過。

但如果其中一個改變了欄位名稱，另一個還在讀取舊欄位，最終合併可能產生 Regression。

Merge Queue 要解決的就是這個問題。

```

Current dev:
    commit A

PR #123:
    commit B

PR #124:
    commit C

Merge Queue:
    A + B       → CI PASS
    A + B + C   → CI FAIL

Result:
    PR #124 cannot merge in this combination
    until the failure is resolved.
```

GitHub Merge Queue 會在最新的 Target Branch 加上候選 PR 變更之後，重新執行 Required Status Checks。

這樣就不只是驗證某個 PR 在過去某個時間點是否正常，而是驗證它與其他已經進入 Queue 的修改能不能共同運作。

![](https://www.google.com/s2/favicons?domain=https://docs.github.com&sz=32)

GitHub Docs

### Step 17 — Layer 7：合併後驗證、監控與 Rollback

假設 PR #123 合併到 `dev` 後，部署到測試環境。

如果你的 Authentication Pipeline 有 Feature Flag，可以採用：

```
Develop / Internal Testing
          ↓
Staging
          ↓
Canary Deployment
          ↓
Production Rollout
          ↓
Monitoring
```

應觀察的指標包括 Authentication Error Rate、`InsufficientEvidence` 比例、API Exception Rate、P95 Latency、資料讀取錯誤與 Decision Distribution Drift。

如果發生重大 Regression：

1. 先停止擴大部署，必要時停用 Feature Flag。
    
2. Revert 問題 Commit，或回退到已驗證版本。
    
3. 分析是哪一層沒有捕捉到錯誤。
    
4. 增加能重現錯誤的 Regression Test。
    
5. 把真實事件加入未來 AI Reviewer 的 Evaluation Dataset。
    

Anthropic 也是使用過去造成 Incident 的 PR、Fix-forward 和 Revert Case，持續評估 Reviewer 的能力，而不是只靠主觀感覺判斷 AI Review 是否改善。

這裡的核心概念是：Code Review 並不是 Merge 前的一次性活動，而是一個可以持續學習的工程品質系統。

## 五、如何在自己的 Repository 真正建立這套架構？

前面是完整流程。現在進一步說明，第一次導入時應該如何設定。

我建議分成四個部分：Local Rules、Review Rules、CI/Security Gates、GitHub Approval Policy。

### 1. 設定 `CLAUDE.md`

在 Repository Root 建立：

```
ImagingLibWatch/
├── CLAUDE.md
├── REVIEW.md
├── authentication/
├── control/
├── helper/
├── App/
└── .github/
```

`CLAUDE.md` 是提供 Claude Code 理解專案的主要指示檔案，包含 Architecture、Conventions、Build/Test Commands、重要限制。

例如：

```

# ImagingLibWatch Engineering Guidelines

## Architecture
- Python application for automated watch imaging,
  analysis, and authentication.
- Hardware integrations include cameras, motion stages,
  illumination, and laser sensors.
- Authentication uses statistical and Bayesian evidence.

## Development
- Use feature branches based on dev.
- Keep changes small and focused.
- Preserve existing public APIs unless explicitly approved.

## Verification
- Run affected unit tests after every code change.
- Run regression tests before committing.
- Do not claim hardware validation without real test output.
- Report tests that could not be executed.

## Authentication Constraints
- Missing evidence must not imply authenticity.
- Never silently convert missing data into a positive score.
- Preserve the distinction between unavailable evidence
  and an actual negative observation.

## Safety
- Never commit credentials or secrets.
- Do not deploy infrastructure without explicit approval.
- Do not push or merge without user authorization.
```

這份文件應該保持精簡。

大量細節可以放到其他架構文件中，然後由 `CLAUDE.md` 引用，而不是把全部 Repository Documentation 都塞進同一個檔案。

### 2. 設定 `REVIEW.md`

`REVIEW.md` 主要用來告訴 Managed Code Reviewer 哪些地方值得審查，以及哪些問題應該被判定為 Bug。

例如：

```

# Code Review Policy

## Primary Objective
Find newly introduced correctness and security bugs.

## Important Review Areas

### Authentication
- Missing or invalid evidence handling
- Incorrect Bayesian aggregation
- Unintended changes to decision thresholds
- Incorrect mapping of expert labels
- Cross-watch or cross-tenant data access
- Backwards-incompatible result schemas

### Imaging
- Camera acquisition failures
- Incorrect image-to-watch association
- Frame synchronization and concurrency bugs
- Silent loss of image data

### AWS
- Incorrect S3 object access
- Missing authorization checks
- Non-idempotent processing and retries
- Unsafe deployment or rollback behavior

## Evidence Requirements
For each finding:
- Provide the exact file and line
- Explain the failure condition
- Identify the affected behavior
- Propose a reproducible test

## Review Noise
- Prioritize correctness over formatting.
- Do not flag hypothetical bugs without evidence.
- Clearly label pre-existing issues.
- Avoid repeat findings already resolved.
```

但有一個容易忽略的細節：

`CLAUDE.md` 和 `REVIEW.md` 並不是在所有 Claude Review 模式下都會被相同方式使用。

|功能|`CLAUDE.md`|`REVIEW.md`|
|---|---|---|
|Claude Code 日常開發|使用|不保證自動使用|
|Local `/code-review`|使用|不自動讀取|
|Managed GitHub Code Review|使用|使用|
|自訂 Reviewer Agent|取決於設定|取決於設定|

目前 Anthropic 的 Local Review 文件明確指出，`/code-review` 不會自動讀取 `REVIEW.md`；它主要用於 GitHub Managed Code Review。若希望 Local Reviewer 同樣遵循這些規則，可以透過 Prompt 明確要求讀取，或者將關鍵規則寫入 `CLAUDE.md`。

![](https://www.google.com/s2/favicons?domain=https://code.claude.com&sz=32)

Claude Code Docs

### 3. 在 GitHub 啟用 Managed Claude Code Review

如果使用的是 Anthropic Team 或 Enterprise Plan，而且具備相應的組織管理權限，可以依照官方流程設定：

1. 前往 [Claude](https://claude.ai?utm_source=chatgpt.com) 的 Organization Settings → Claude Code。
    
2. 找到 Code Review，選擇 Configure。
    
3. 安裝 Claude GitHub App，核准所需 Repository 權限。
    
4. 選擇要啟用的 Repository。
    
5. 設定 Review Trigger，並以一個測試 PR 驗證。
    

Managed Code Review 的主要觸發模式如下：

|Review Trigger|適用情況|
|---|---|
|Once after PR creation|控制成本、PR 相對穩定|
|After every push|希望每次修改都自動重新審查|
|Manual|只在需要時要求深入 Review|

手動模式可以在 PR 的頂層 Comment 輸入：

```
@claude review
```

依目前官方規則，這會啟動審查，並使該 PR 後續 Push 也可以觸發 Review。PR 必須是 Open 且非 Draft，發出指令的人也需要具有適當的 Repository 權限。

Managed Code Review 現在已經是 Team / Enterprise 可使用的產品，但會另外依 Token Usage 收費，不包含在原本 Claude Code 的一般使用額度裡。

Anthropic 目前公布每次 Review 平均約 US$15–25，實際依 PR 大小、複雜度及驗證工作量而改變。

![](https://www.google.com/s2/favicons?domain=https://support.claude.com&sz=32)

Claude Help Center

如果你主要是一個人開發，先使用本機 `/code-review`，再挑重要 PR 使用較深度的 Review，通常比讓每個小 Commit 都重新跑高成本的 Managed Review 更合理。

### 4. 設定 GitHub Actions CI

以下是簡化的 Python CI 示範，檔案可以放在：

`.github/workflows/authentication-ci.yml`

```

name: Authentication CI

on:
  pull_request:
    branches: [dev]
  merge_group:
    types: [checks_requested]

jobs:
  authentication-tests:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install -r requirements-dev.txt

      - name: Run authentication tests
        run: |
          python -m pytest -q \
            helper/tests/test_authentication_decision.py
```

這是 CI 範本，不是保證可直接在你的 Repo 執行的設定。其中 `requirements-dev.txt`、測試路徑和 Python 版本都要依實際 Repository 修改，並確保硬體依賴能在 Runner 上正確隔離。正式環境也建議固定 GitHub Actions 的可信版本或 Commit SHA。

其中：

```
on:
  pull_request:
  merge_group:
```

特別重要。

如果你啟用了 GitHub Merge Queue，Required CI Workflow 必須支援 `merge_group` 事件，否則 Queue 建立的新 Merge Group 可能無法收到 Required Check Result，導致 Merge 失敗。

![](https://www.google.com/s2/favicons?domain=https://docs.github.com&sz=32)

GitHub Docs

### 5. 設定 CODEOWNERS

新增：

`.github/CODEOWNERS`

```

# Example owner mappings
/authentication/       @YOUR_ORG/auth-owners
/.github/workflows/    @YOUR_ORG/platform-owners
/control/              @YOUR_ORG/hardware-owners
```

這裡的 `YOUR_ORG` 和 Team Names 必須改成實際 GitHub Organization 中存在、有權限的團隊。

這樣就可以根據修改檔案，自動要求適當的 Code Owner 審查。

不過 CODEOWNERS 不等於 Anthropic 的 Dynamic Risk Classifier。例如一個路徑列了兩個 Owner，不代表 GitHub 自動要求兩個不同團隊分別核准。若要真正實現 High Risk 必須由兩名不同角色核准，仍需要配合 Rulesets、Required Reviewers 或其他 Approval Automation。

### 6. 設定 GitHub Branch Protection

針對 `dev` 和 `main`，可以進入：

GitHub Repository → Settings → Branches / Rulesets。

我會建議啟用以下機制：

|GitHub 設定|用途|
|---|---|
|Require a pull request before merging|禁止任意直接修改受保護 Branch|
|Require status checks|CI 沒通過不得 Merge|
|Require conversation resolution|Review Threads 未處理不得 Merge|
|Require code owner reviews|關鍵檔案需要 Owner Review|
|Dismiss stale approvals|新 Commit 後重新核准|
|Require merge queue|合併前驗證最新程式碼組合|

這些都是 GitHub 原生提供的保護能力。

![](https://www.google.com/s2/favicons?domain=https://docs.github.com&sz=32)

GitHub Docs

對於你的專案，我不建議一開始就模仿 Anthropic，讓 Low-risk Code 可以零額外人類核准直接合併。

理由是你正在建置涉及硬體控制、雲端資料與真偽判定的系統，而目前還沒有足夠的歷史 Reviewer Evaluation、Production Incident Statistics 與完整自動化驗證證據，來支持放寬審查門檻。

可以先要求每個 PR 至少由一名有責任的人檢查；Authentication Policy、AWS IAM、Deployment 和 Hardware Safety 相關變更再增加獨立 Reviewer。

## 六、Anthropic 架構中最值得學習的三個進階設計

### A. Reviewer 不只要能找 Bug，還要能證明 Bug

一般 AI Code Review 最常見的問題是：

> AI 說這裡可能有 Race Condition，但沒有提供實際可發生的執行情境。

成熟的 Reviewer 應該盡可能提供：

```
Finding:
    Concurrent updates may overwrite results.

Evidence:
    Two workers can write the same result key.

Reproduction:
    Launch two jobs for the same watch_id.

Expected:
    One canonical result, no lost update.

Observed:
    Last writer overwrites prior state.

Suggested Fix:
    Add idempotency control and conditional writes.
```

但上面的 `Observed` 只有在 Reviewer 真的執行並觀察到失敗時，才能稱為 Observed。若只是分析推測，就應該標示為 Hypothesis 或 Potential Failure。

這種區分能減少 False Positives，避免 AI 把合理猜測誤報成已經確認的 Production Bug。

### B. Code Review 本身也應該有 Evaluation Pipeline

Anthropic 並不只評估開發中的程式碼，也評估 Code Reviewer 的表現。

對你的 Repo，可以建立專門的 Reviewer Evaluation Dataset：

|Test Case|已知真實問題|Reviewer 應該如何表現|
|---|---|---|
|AUTH-001|無證據仍通過驗證|必須找到|
|AUTH-002|Tenant 越權查詢|必須找到|
|AUTH-003|Bayesian Score 計算錯誤|必須找到|
|IMG-001|相機 Capture Failure 未處理|必須找到|
|AWS-001|重試導致重複寫入|必須找到|
|CLEAN-001|完全正確的 Code|不應產生重大誤報|

然後衡量：

\[ \text{Recall}=\frac{\text{Detected True Bugs}}{\text{All Known Bugs}} \]

\[ \text{Precision}=\frac{\text{Correct Findings}}{\text{All Reported Findings}} \]

另外再追蹤 False Positive Rate、Review Cost、Review Time，以及 Review Rounds to Pass。

這樣之後如果你修改 `REVIEW.md`、更換 Claude Model、增加 Reviewer Agents，就能使用同一批 Golden Cases 測試有沒有真的改善。

### C. 按風險分配 Reviewer，而不是對每段程式碼使用相同成本

這裡提供一個互動式的風險分級範例，幫助你理解如何把 Anthropic 的觀念轉成 Repository Policy。

試選這次 Code Changes 的類型

Documentation / Text

General UI / Application Code

Camera / Motion / Hardware

Authentication / AI Decision

AWS Infrastructure / Security

Suggested Risk Tier

## High

可能影響真偽判定、證據完整性或敏感資料

Required Checks

Domain Tests、Statistical Tests、Security Review、Deep Review

Human Review

2 位建議：Authentication + System Owner

此處是我建議給你 Repository 的風險分級示範，不是已經部署的 Anthropic Classifier。

## 七、最後核實：哪些理解正確？哪些容易誤解？

根據截至 2026 年 10 月 20 日可查證的官方文件，整理如下：

|敘述|判斷|正確解釋|
|---|---|---|
|Claude Code 可以在 Git Push 之前做 Code Review|正確|Local `/code-review` 支援未提交修改|
|Anthropic 使用多個平行 AI Agents 審查 PR|正確|包含候選 Bug 尋找及後續驗證|
|Claude 自己寫完 Code，自己 Approve 就能 Merge|不正確|AI Approval 不能取代要求的人類核准|
|每一個 PR 都必須有兩名 Human Reviewers|不正確|Anthropic 依變更風險分級|
|Managed Claude Code Review 會自動禁止有 Bug 的 PR Merge|不完全正確|必須配合適當的 GitHub Merge Gate|
|Local `/code-review` 會自動使用 `REVIEW.md`|不正確|官方目前說明 Local Review 不會自動讀取它|
|只要把 `CLAUDE.md` 寫好，就能保證無 Bug|不正確|仍需要測試、Reviewer Verification 與 Human Judgment|
|Reviewer 可以無限重跑直到沒有任何意見|不建議|Anthropic 明確設計了 Review Convergence 機制|
|可以完全不 Push 就完成正式 GitHub PR Review|不正確|正式 PR、GitHub CI、Merge Queue 需要遠端程式碼|
|Code Review 完成後就不需要再監控 Production|不正確|Rollback 與 Incident Learning 是整個架構的一部分|

## 八、針對你現有開發流程，我最推薦的落地版本

如果你同時使用 Codex、Claude Code、VS Code 與 GitHub，並採用 `dev → main` 的開發模式，我建議先建立以下流程。

本機階段：不需要 Git Push

1. Engineer / Codex / Claude 編寫 Code

明確定義需求與驗收條件，修改 Feature Branch

2. Deterministic Verification

Unit Tests → Integration Tests → Build / Lint

3. Independent Claude Code Review

`/code-review high` → Fresh-context Requirements Review → `/security-review`

4. Review Convergence + Commit

Fix → Test → Verify → Commit

GitHub 階段：正式 PR 審查

5. Push Feature Branch → Open PR to dev

6. Managed Code Review + Security + CI

7. Risk-Based Human Approval

8. Merge Queue → dev

部署階段：品質保護

9. Staging / Hardware / AWS End-to-End Tests

10. Release Approval → main / Production

11. Monitor → Rollback → Feed Incident into Evals

其中，正式上線 `main` 應視你的發布政策另設 Approval Gate，而不是認為 `dev` 合併成功就可以直接 Production Deployment。

### 總結

Anthropic 的重點不是「用更多 AI Agents 取代 Reviewer」，而是建立一條可驗證的品質鏈：

Automated Verification → Independent AI Review → Verified Findings → Security Gates → Risk-Based Human Approval → Merge Validation → Production Feedback。

我會特別建議你優先完成 Local Independent Review、強制 Regression Tests、GitHub Required Checks、Security Review，以及 Production Rollback。這五項通常比一開始就投入大量時間建立複雜的 Multi-Agent Orchestration，更容易得到可量化的實際改善。





#### 核實文章論點
```
也請檢查以下是否正確: 一、這篇讓我最有共鳴的，是 Anthropic 用 Revert Rate / Incident Rate 作為品質護欄。不追求零錯誤，而是重視快速發現、快速復原，並透過數據驗證流程調整是否真的有效。

二、「You land it, you own it」的文化，搭配 Revert、On-call Rotation、Incident Review 等機制，才讓降低人工 Review 比重這件事變得可控。

以下整理幾個我覺得值得參考的重點。完整的七層審查架構放在圖中，原文連結放在第一則留言。  
Anthropic 的 monorepo，每天合併的程式碼變更，從 5 月約 2,500 個，成長到 9 月超過 30,000 個。

1. 重新定義 Review 的目標

與其問「有沒有人看過這段 Code」，不如問「這個變更可能出什麼錯？哪一道檢查最適合抓到它？」​他們追蹤每個 PR 的人工 Approve 數量、等待時間，同時以 #RevertRate、#IncidentRate 作為品質護欄。結果從 6 月到 9 月，Claude Code 的 Merged PR 數量增加超過一倍，需要 Revert 的比例反而下降超過 40%。

2. Review 不再只是 PR 的最後關卡

許多檢查在 PR 建立之前就開始了。Claude 會先執行測試、Build、截圖比對，再由另一個使用全新 Context 的 Subagent 審查 Diff。讓問題盡可能在開發階段就被發現，而不是全部留到 PR 才處理。

3. 多層防護，而不是依賴單一關卡

#CI、AI Code Review、Security Review、人工審核、#MergeQueue、#FeatureFlag，以及 #Rollback，共同構成防護機制。每一層都可能漏掉問題，但透過不同機制交叉檢查，可以降低問題一路進入正式環境的機率。

4. AI Review 需要降噪  
    多個 Agent 平行找 Bug，再由另一組 Agent 交叉驗證，過濾掉不夠可靠的發現與純風格類意見。留下的 Review 意見都需要處理：修正問題，或說明理由後關閉。否則 AI Reviewer 產生大量低價值意見，反而會增加工程師的負擔。
5. 根據風險決定人工審查程度

不是每個 PR 都需要相同數量的 Reviewer。透過分類器依照變更內容判斷風險，低風險變更可以不需要額外人工 Approve，高風險變更則需要更多人審查。同時也有明確限制：AI 的 Approve 不能取代人的 Approve；新 Commit 進來後，原本的核准也會失效。

6. 避免 AI Review 無限循環

曾經有 PR 被 AI 作者與 AI Reviewer 來回修改超過 50 輪。後來透過限制 Review 輪數、只審查新的 Diff、避免重複提出相同問題，讓平均 Review 輪數從 2.08 降到 1.32。

7. 把 Incident 變成 Evaluation Dataset

將過去造成事故或需要 Revert 的 PR 整理成評測集，每天驗證 AI Reviewer 的能力。未來不論是調整 Prompt、工具或模型，都可以透過相同的評測標準，確認是否真的改善。

8. 人應該專注在更需要判斷力的事情

例如需求是否合理、架構方向、高風險變更、證據判讀，以及最終的責任歸屬。這些事情，不會因為 AI 能寫出更多程式碼就自動解決。
```
我已經逐項對照 Anthropic 在 2026 年 10 月 15 日發表的官方文章 [How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/?utm_source=chatgpt.com)，以及另一篇官方工程文章 [Redesigning source control at Anthropic](https://www.anthropic.com/engineering/redesigning-source-control?utm_source=chatgpt.com)。

結論：你的整理整體相當準確，8 個重點都有原文依據，可以作為 LinkedIn 技術文章發表。不過有 5 個細節值得修正或補充，尤其是數據的適用範圍、Human Approval 條件，以及 Review 輪數的解釋。

## 一、最重要的 5 個核實結果

### 1. 每日 2,500 → 30,000 個 Changes：正確，但不是 Claude Code Repository 的數字

Anthropic 官方確實公布：

2026 年 5 月

# ~2,500

Merged changes / day

2026 年 9 月

# 30,000+

Merged changes / day

但必須區分兩個範圍：

- Anthropic 整體 Monorepo：每天合併的 Code Changes，從約 2,500 增加至超過 30,000。
    
- Claude Code 專案 Repository：2026 年 6 月至 9 月，Merged PR 數量增加超過一倍，Revert 比例相對下降超過 40%。
    

Claude Code 使用自己的獨立 Repository，並不屬於上述 Monorepo。

你原本的句子已明確寫「Anthropic 的 monorepo」，因此本身沒有錯誤。只是不要在後文把這組數字當成 Claude Code 團隊的每日 PR 數量。

![](https://www.google.com/s2/favicons?domain=https://www.anthropic.com&sz=32)

Anthropic

+1

### 2. Revert 比例下降超過 40%：正確，但不是減少 40 個百分點

Anthropic 的說法是 Revert Share 的相對降幅超過 40%，並未公布這段期間的確切起始與結束比例。

舉例來說，假設原本 10% 的 Merged PR 需要 Revert，後來下降到 5.9%，就是相對下降 41%，而不是下降 40 個百分點。

你使用的「需要 Revert 的比例反而下降超過 40%」正確。

但我建議加上「相對」，避免讀者誤解成絕對降幅。

另外，Anthropic 並沒有在這篇文章公布同期間 Incident Rate 下降多少。Incident Rate 是品質監控指標，不應把 Revert 的改善直接視為 Incident 也改善了相同比例。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

### 3. 「You land it, you own it」：完全正確，但責任並非全部壓在作者身上

這是 Anthropic 官方文章的正式章節標題。

原文強調，小型或中型變更如果出了問題，原本負責讓變更合併的工程師，通常會負責修復。

但同時強調：

- Ownership 也是團隊共同承擔的。
    
- Incident Review 是檢討系統，而不是責怪個人。
    
- 高 Blast Radius 的修改需要更嚴格的事前審查。
    
- 透過 Feature Flags、Telemetry、Staged Release、Rollback 降低事故影響。
    

你將它與 On-call Rotation、Incident Review 結合，是正確且很重要的觀察。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

### 4. Human Approval：需要補充兩個 Classifier 和 Trusted Author

你寫的低風險零核准、中風險一人、高風險多人，方向正確。

但 Anthropic 並不是只有一個簡單的 Risk Classifier：

1. 第一個 Classifier 根據變更內容判斷 Low / Medium / High。
    
2. 第二個 Classifier 判斷，即使風險分級允許較少核准，這次變更是否仍應由另一個人檢查。
    

此外，輕量核准路徑只開放給 Trusted Authors；Security-sensitive Paths、指定敏感區域及 CI Workflow Files 還有額外要求。

「新 Commit 進來後，原本核准會失效」也需要更精確：當新增的 Commit 無法被現有核准機制確認涵蓋時，才需要重新核准，不是任何 Commit 都一律撤銷核准。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

### 5. Review 輪數從 2.08 → 1.32：正確，但三輪上限並非強制停止所有 Review

你引用的數字與官方文章 Figure D 相符。

平均完成審查所需輪數

調整後

平均 Review 輪數1.32

00.551.11.652.2調整前調整後

Anthropic 內部 PR；Review Convergence 調整前後

更完整的比較是：

|Metric|調整前|調整後|
|---|---|---|
|一輪 Review + Fix 即通過|65%|78%|
|需要五輪以上|8.9%|1.5%|
|平均 Review 輪數|2.08|1.32|

所謂三輪上限，是指三輪之後 Reviewer 應要求作者重新思考問題及設計，而不是無限進行同樣的修正；不表示所有高風險問題到了第三輪就可以無條件跳過。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

+1

## 二、逐項核實你列出的 8 個重點

|項目|核實結果|我的建議|
|---|---|---|
|1. 重新定義 Review 目標|正確|建議補充 Human Review Wait Time 的 P50 / P95，以及 Revert Rate 是相對降幅|
|2. PR 建立前就開始審查|正確|Fresh-context Subagent 是關鍵設計，保留|
|3. 多層防護|正確|官方 Figure B 是「六層 Checks，加上 Rollback」；稱七層可以，但要說明|
|4. Multi-Agent Review 降噪|正確|補充 Pre-existing Bugs 分類，以及 Findings 必須有證據|
|5. Risk-based Human Approval|需補充|加上第二個 Classifier、Trusted Authors 與 Commit Approval 規則|
|6. 避免 AI Review 無限循環|正確|補充 59% 的重複發現涉及第一輪已存在的 Code|
|7. Incident 變成 Evaluation Dataset|正確|補充由熟悉程式碼的工程師確認 Ground Truth，並量測 False Positives|
|8. 人專注於高判斷力工作|正確|建議特別保留「決定某個功能是否值得做」與「對結果負責」|

你最值得強調的觀點是第 1、6、7 點的關聯：Review 不只是找 Bug，還要量測 Review 的品質、成本、收斂能力，以及部署後的實際結果。

這一點確實比單純介紹 Multi-Agent 架構，更接近 Senior / Staff Engineer 會關注的工程管理問題。

## 三、我建議的修正版（可直接用於 LinkedIn）

以下保留你原本的語氣、觀點和八個重點，但調整數據的精確性，以及幾個容易被挑戰的技術描述。

Anthropic 最新分享了 **How the Claude Code team does code review**。比起使用多少 AI Agents，我覺得更值得關注的是：**當 AI 讓程式碼產出速度大幅提高，團隊該如何重新設計 Code Review，才能兼顧開發效率與品質？**

這篇讓我最有共鳴的，是 Anthropic 用 **Revert Rate / Incident Rate 作為品質護欄（Quality Guardrails）**。

不是追求零錯誤，而是讓問題能更早被發現、影響範圍能被控制、錯誤能快速復原，並透過數據驗證流程調整是否真的有效。

另一方面，「**You land it, you own it**」的文化，搭配 Feature Flags、Rollback、On-call Rotation 與 Incident Review，才讓降低人工 Review 比重這件事變得更可控。重點不是讓 AI 取代人的責任，而是讓人的注意力集中在最有價值的決策上。

以下整理八個我覺得值得參考的重點。完整的多層審查架構放在附圖，原文連結放在第一則留言。

**背景數據**

2026 年 5 月至 9 月，Anthropic 整體 Monorepo 每天合併的 Code Changes，從約 2,500 個成長到超過 30,000 個。

而在獨立的 Claude Code Repository 中，6 月至 9 月的 Merged PR 數量增加超過一倍，但需要 Revert 的 PR 比例反而相對下降超過 40%。

**1. 重新定義 Code Review 的目標**

與其問「有沒有人看過這段 Code」，不如問：

「這個變更可能出什麼錯？哪一道自動化或人工檢查最適合抓到它？」

Anthropic 追蹤每個 PR 的人工 Approvals、Review Comments 和 Human Waiting Time，同時以 Revert Rate / Incident Rate 作為品質護欄。

降低 Human Review Time 不代表成功。如果節省下來的時間最後都花在修復 Production Incidents，就只是把成本轉移到另一個階段。

**2. Review 不再只是 PR 的最後關卡**

許多檢查在 PR 建立之前就已經開始。

Claude 先透過 Tests、Build、Screenshot Comparison 等方式驗證實際行為，再由另一個具有 Fresh Context 的 Subagent 審查 Diff。

重點不只是讓 AI 說「完成了」，而是要求它提供可驗證的執行結果，盡可能在開發階段就發現問題。

**3. 多層防護，而不是依賴單一關卡**

Pre-PR Verification、AI Code Review、Security Review、Blocking CI、Risk-based Human Approval、Merge Queue / Gradual Release，再加上 Production Monitoring 與 Fast Rollback，共同形成完整的品質防線。

每一層都有可能漏掉問題，但不同機制交叉驗證，可以降低錯誤一路進入正式環境的風險。

**4. AI Review 需要降噪（Noise Reduction）**

多個 Agents 平行尋找 Bugs，再由另一組 Agents 交叉驗證候選 Findings，過濾缺乏足夠證據的問題與低價值的 Style Comments。

Review Findings 也會區分 Important、Nit 與 Pre-existing。

留下的意見必須被處理：修正問題，或提出理由後關閉 Review Thread。

因為 AI Reviewer 如果產生大量 False Positives，不只增加工程師負擔，還可能讓 Coding Agent 不斷修改原本沒有問題的程式碼。

**5. 根據風險決定 Human Review 程度**

不是每個 PR 都需要相同數量的 Reviewer。

Anthropic 使用 Classifier 根據實際變更內容決定風險等級：

- Low Risk：符合條件時可以不需要額外 Human Approval。
    
- Medium Risk：一名 Human Approver。
    
- High Risk：兩名 Human Approvers。
    

此外還有第二個 Classifier、Sensitive Paths 與 Trusted Author 機制，避免單純靠風險分級就放行不適合自動合併的變更。

AI 的 Approval 不能取代人的 Approval。對於無法被既有核准涵蓋的新 Commit，也必須重新確認。

**6. 避免 AI Review 無限循環**

Anthropic 曾有 PR 被 AI 作者與 AI Reviewer 來回修改超過 50 輪。

後來透過三輪後要求重新檢視方向、只重新審查新 Diff、避免重複 Findings，以及要求一次修正同類型 Bugs，改善 Review Convergence。

調整後：

- 一輪 Review / Fix 就通過的 PR：65% → 78%。
    
- 需要五輪以上的 PR：8.9% → 1.5%。
    
- 平均 Review 輪數：2.08 → 1.32。
    

這提醒我們，更多輪 AI Review 不一定代表更高品質；Review 本身也需要衡量效率與邊際效益。

**7. 把 Incident 變成 Evaluation Dataset**

將過去造成 Production Incidents 或需要 Revert 的 PR，連同 Fix-forward 與 Revert 資料整理成 Evaluation Dataset。

由熟悉程式碼的工程師確認 Bug 與 Ground Truth，再每天執行評測。

未來不論修改 Reviewer Prompt、Tools 或 Model，都能用相同標準評估是否真的改善。同時也追蹤 False Positives，避免為了提高 Bug Recall 而產生過多無效意見。

**8. 人應該專注在更需要判斷力的事情**

例如：某個需求是否值得做、架構方向是否合理、高風險變更的取捨、測試證據是否充分，以及最終的責任歸屬。

這些問題不會因為 AI 能寫出更多程式碼就自動解決。

**我的最大 Takeaway：**

未來 Software Engineering 的瓶頸，可能不再只是 Code Generation，而是如何建立一套能隨著 AI 開發速度擴展的 Verification、Review、Release 和 Recovery System。

真正重要的，不是有多少程式碼經過 AI Review，而是我們能不能用數據證明：**在節省工程師時間的同時，沒有把更多風險轉移到 Production。**

## 四、還有兩個值得保留的技術提醒

第一，這些都是 Anthropic 自行公布的內部觀測數據。 它們支持「這套流程在其團隊的觀測期間取得改善」，但不等於有控制實驗證明每個改進都由 AI Reviewer 單獨造成。不同的測試、部署機制、變更組成與團隊行為也可能影響結果。

第二，Quality Guardrails 不代表已經有完整的自動回饋控制。 Anthropic 明確承認：目前還沒有當 Revert / Incident 指標惡化時，自動提高 Approval 要求的 Circuit Breaker。現階段仍依賴 Author、On-call 工程師與監控儀表板進行判斷。

![](https://www.google.com/s2/favicons?domain=https://claude.dev&sz=32)

claude.dev Blog

這個限制很值得留意。如果未來要把這種架構應用在涉及 AWS Production、硬體控制或 AI Authentication Decision 的系統，不能因為自動化 Review 的表現改善，就直接假設零額外人工核准已經安全。

整體而言，你原本的文章重點是成立的。 如果準備公開發布，我認為最重要的是清楚區分 Monorepo 和 Claude Code Repository 的數據、精確描述 Human Approval 的例外條件，以及避免把觀測到的改善直接當成因果證明。這三點修正後，文章會更嚴謹。

主要參考來源：

- [Anthropic — How the Claude Code team does code review](https://claude.dev/blog/solving-code-review-at-anthropic/?utm_source=chatgpt.com)（2026/10/15）
    
- [Anthropic — Redesigning source control at Anthropic](https://www.anthropic.com/engineering/redesigning-source-control?utm_source=chatgpt.com)（2026/10/13，10/15 更新）

