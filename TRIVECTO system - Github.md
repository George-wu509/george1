
|                                                      |     |
| ---------------------------------------------------- | --- |
| [[#### Github公司帳號跟權限]]                               |     |
| [[#### Merge github dev branch to main branch]]      |     |
| [[#### Create Trivecto ImagingLibWatch github repo]] |     |
| [[#### 標準的 PR 描述模板 (Template)]]                      |     |
|                                                      |     |
|                                                      |     |


#### Github公司帳號跟權限
```
公司CEO create 公司github account也把我加入公司github並說給我很高的權限, 不過因為CEO對軟體管理不熟, 而且我是負責軟體開發的tech lead. 要怎麼知道我的權限以及查公司github的權限? 所以github account也有不同的權限嗎? 如果我要把我的repo從我的github搬到公司github, 要怎麼確定這個github的這個repo其他人是看不到因為是公司產品, 包括在公司github的其他人
```

可以。GitHub 在「公司 Organization」層級、以及「單一 repo」層級都有不同權限；而且**“給很高權限”**可能是：

- 你是 **Organization owner**（最高）
    
- 你是 **Organization member** 但被加到某些 **Team**（repo 權限很高）
    
- 你是 **Outside collaborator**（只對特定 repo 有權限，不算 org 成員）
    

下面我用「怎麼查你自己的權限 → 怎麼查公司整體權限設定 → 怎麼確保搬 repo 後只有指定人能看」三段，一步一步給你。

---

## 1) 怎麼知道你在公司 GitHub 的權限是什麼

### A. 先確認：你是 Org owner / member / outside collaborator

1. 打開公司 GitHub Organization 頁面
    
2. 右上角 **Your organizations** → 點公司 org
    
3. 進去後找 **People**（人員）
    
    - 你會看到自己的角色通常會標示：**Owner / Member / Billing manager**（或類似）
        

> 如果你看不到 **People**，代表你不是 owner（或你沒有管理成員的權限），那你要靠 repo 權限來判斷你能做什麼。

### B. 在單一 repo 內確認你對該 repo 的權限

到該 repo → **Settings**（若看得到表示你至少有 admin 權限；看不到通常不是 admin）

- **Settings → Collaborators and teams**  
    這裡會列出：
    
    - 哪些 Team 有權限（Read/Triage/Write/Maintain/Admin）
        
    - 哪些個人 collaborator 有權限
        
- 你也可以在 repo 的頁面（有時在右側/insights）看到你的 role，但最準還是 Settings 內的列表。
    

---

## 2) GitHub 真的有不同權限嗎？（你需要懂的權限模型）

### A. Organization 層級（公司層）

- **Owner**：幾乎全權（管理人、管理 repo、刪 repo、設定安全政策、看到稽核 log）
    
- **Member**：是 org 成員，但不一定能看到所有 private repo（視設定）
    
- **Outside collaborator**：不是成員，只被授權某些 repo
    

### B. Repo 層級（專案層）

常見權限（由低到高）：

- **Read**：看、clone、開 issue
    
- **Triage**：管理 issue/PR（不改 code）
    
- **Write**：push 分支、開 PR
    
- **Maintain**：管理 repo 設定（部分）
    
- **Admin**：最高（可管理權限、刪 repo、改 visibility 等）
    

### C. 影響「公司其他人能不能看到 repo」的關鍵：Base permissions + Repo access

- **Organization 設定裡有個 “Base permissions”**  
    例如：Base = Read，代表**所有 org 成員**預設至少能讀你們 org 裡的 repo（視 repo 設定與例外而定）。
    
- 正確做法通常是：**Base permissions 設為 “No permission”**，然後只用 Team/Collaborator 明確授權。
    

---

## 3) 我把 repo 搬到公司 GitHub，怎麼確保公司其他人也看不到？

你要同時控制 **(1) repo visibility** + **(2) 誰被授權** + **(3) org 的預設權限/政策**。

### Step 0：先盤點公司 org 的「預設權限」與「成員可見性」

如果你是 org owner（或有管理權限）：

1. 進公司 org → **Settings**
    
2. 找 **Member privileges / Repository permissions / Base permissions**
    
3. 確認 **Base permissions** 是不是 **No permission**
    
4. 檢查是否允許成員建立 repo、邀外部 collaborator、允許 fork private repo（這些都會影響外流風險）
    

> 如果你不是 owner：你至少可以確保「repo 自己的權限」設得嚴，並要求 CEO 幫你把 org 的 Base permissions 改掉（不然 org 內的人可能預設能讀）。

---

### Step 1：把 repo 變成 Private（必要但不夠）

搬到公司 org 後，確認 repo 是 **Private**：

- repo → **Settings → General → Danger Zone → Change repository visibility** → **Private**
    

但注意：Private ≠ 只有你能看。Private 的意思是「不是公開」，**仍然是給被授權的人看**，而 org 的預設權限可能讓很多人自動有 Read。

---

### Step 2：把 repo access 變成「白名單」：只讓指定 team/人看

到 repo → **Settings → Collaborators and teams**

- **移除**不需要的 team / 個人
    
- 建立最小 team，例如：
    
    - `software-core`（Write 或 Maintain）
        
    - `release-admin`（Admin，通常只有你/少數人）
        
- 其他人不在 team 就看不到（前提：org base permissions 不是 Read）
    

---

### Step 3：檢查是否有 “Outside collaborators” 或不該存在的 team access

在 org 層（若你是 owner）：

- org → **People → Outside collaborators**
    
    - 把不必要的外部人移除
        
- org → **Teams**
    
    - 檢查哪些 team 被授權到哪些 repo
        

---

### Step 4：防止「不小心外流」的常見洞

建議你把以下幾個也設定起來（尤其是公司產品）：

1. **禁止或限制 private repo 的 fork**
    
    - org settings 有「Allow forking of private repositories」之類的選項
        
    - 最安全：關掉（或只允許特定情況）
        
2. **Actions 權限**
    
    - repo/settings → Actions：限制哪些 workflow 可跑、是否允許外部 PR 執行 secrets
        
3. **Secrets 管理**
    
    - 不要把 AWS keys 放 repo
        
    - 用 org/repo secrets
        
4. **Branch protection**
    
    - main/master：禁止直接 push
        
    - 必須 PR + review + CI pass
        
5. **Audit log（稽核）**
    
    - org owner 才看得到
        
    - 用來追查誰存取/改權限/建立 token
        

---

## 4) 把你的 repo 從個人 GitHub 搬到公司 GitHub 的安全搬移流程

有兩條路：**Transfer（保留歷史/issue/PR）** 或 **重新推一份（更可控）**。

### 方案 A：直接 Transfer（最方便）

在你的個人 repo：

1. repo → **Settings → General → Danger Zone → Transfer ownership**
    
2. 目標選公司 org（你必須有權限轉入）
    
3. Transfer 後立刻做：
    
    - 立刻確認 repo 是 **Private**
        
    - 立刻到 **Collaborators and teams** 做白名單
        
    - 立刻檢查 org 的 Base permissions（若是 Read，要快修）
        

⚠️ Transfer 會把 repo 從你個人名下移走（URL 會 redirect），而且如果 org 有預設 Read，可能「瞬間」就讓很多人有讀權限。

---

### 方案 B：新建公司 repo → 你手動推上去（最安全、最可控）

1. 在公司 org 建一個 **Private repo**
    
2. 在你本地 repo：
    
    - `git remote add company <company_repo_url>`
        
    - `git push company --all`
        
    - `git push company --tags`
        
3. 再把個人 repo 改成 archive 或刪掉（視公司政策）
    

好處：你在「建立公司 repo 的第一秒」就能先設權限，不容易被 org 預設權限陰到。

---

## 5) 一個快速「你現在就能做」的檢查清單

到公司 org（或 repo）做完這 8 個點，你就能大幅降低外洩風險：

1. ✅ repo 是 **Private**
    
2. ✅ repo → Collaborators and teams：只有你要的人/Team
    
3. ✅ org → Base permissions：**No permission**
    
4. ✅ org → Outside collaborators：清乾淨
    
5. ✅ org/repo：限制 private repo fork
    
6. ✅ main 分支 protection + 必須 PR + CI
    
7. ✅ Actions：外部 PR 不可拿到 secrets
    
8. ✅ Audit log 可用（至少 owner 可查）



#### Merge github dev branch to main branch
```
我的github有個private repo目前有main branch跟dev branch. 現在新開發都在dev branch有很多的new commits跟update都在dev branch. 請中文詳細解釋從dev branch merge到main branch的每一步走流程. 並要指出所有可能的問題跟怎麼避免
```

這是一個非常標準且關鍵的 Git 工作流程。由於你的 `dev` branch 累積了大量的 commits，為了確保 `main` (生產環境/穩定版本) 的安全與整潔，建議根據你的團隊規模或個人習慣，選擇 **GitHub Pull Request (PR)** 或 **Command Line (CLI)** 兩種方式之一。

以下是詳細的中文步驟流程、潛在問題與預防措施。

---

### 方法一：透過 GitHub Pull Request (最推薦，最安全)

這是業界標準做法，即使是個人專案也強烈建議，因為它提供了一個**「緩衝區」**來檢查程式碼，並能視覺化地處理衝突。

**詳細流程：**

1. **Push `dev` 分支：** 確保你本地 (Local) 的 `dev` 所有的改動都已經 push 到 GitHub 上的 `dev`。
    
    Bash
    
    ```
    git checkout dev
    git push origin dev
    ```
    
2. **開啟 PR 介面：**
    
    - 登入 GitHub，進入你的 Repo 頁面。
        
    - 通常 GitHub 會偵測到 `dev` 有新改動，上方會出現一個黃色的 **"Compare & pull request"** 按鈕。
        
    - 如果沒有，點擊 **"Pull requests"** 分頁 -> **"New pull request"**。
        
3. **選擇分支方向：**
    
    - **base:** 選擇 `main` (箭頭指向的目標)。
        
    - **compare:** 選擇 `dev` (來源)。
        
4. **檢查與建立：**
    
    - 往下滑查看 **Diff** (程式碼差異)，確認這些就是你想合併的變更。
        
    - 輸入標題 (例如：`Release v1.0: Add watch authentication feature`) 與描述。
        
    - 點擊 **"Create pull request"**。
        
5. **Merge (合併)：**
    
    - 如果顯示 "This branch has no conflicts with the base branch"，點擊綠色的 **"Merge pull request"**。
        
    - **進階選項：** 點擊綠色按鈕旁的小箭頭，你會看到 **"Squash and merge"**。如果 `dev` 上有很多瑣碎的 commits (如 "fix typo", "debug")，選這個可以把它們壓縮成一個乾淨的 commit 到 `main`。
        

---

### 方法二：透過 Command Line (CLI) 直接合併 (快速)

如果你是單人開發且不想開網頁，這是最直接的方式。

**詳細流程：**

1. **切換到目標分支 (`main`)：**
    
    Bash
    
    ```
    git checkout main
    ```
    
2. **確保 `main` 是最新的：** 這步極為重要！防止遠端 `main` 有別人改動過而你不知道。
    
    Bash
    
    ```
    git pull origin main
    ```
    
3. **執行合併：** 將 `dev` 的內容合進來。
    
    Bash
    
    ```
    git merge dev
    ```
    
    - _註：如果希望保留 `dev` 的分支歷史線圖 (不被快轉 Fast-forward)，可使用 `git merge --no-ff dev`。_
        
4. **解決衝突 (如有)：** 如果沒有衝突，Git 會自動跳出編輯器要你輸入 Merge Commit 訊息，儲存離開即可。如果有衝突，請見下方「潛在問題」章節。
    
5. **推送到遠端 (`main`)：** 現在你的本地 `main` 已經更新了，需要同步到 GitHub。
    
    Bash
    
    ```
    git push origin main
    ```
    

---

### 可能遇到的問題與解決方案

#### 1. Merge Conflicts (合併衝突)

這是最常見的問題。當 `main` 和 `dev` 修改了同一行程式碼，Git 不知道該聽誰的。

- **現象：** 執行 merge 時出現 `CONFLICT (content): Merge conflict in filename.py`。
    
- **解決：**
    
    1. 開啟衝突的檔案。
        
    2. 尋找 `<<<<<<< HEAD` (目前 main 的內容) 和 `>>>>>>> dev` (新進來的內容)。
        
    3. 手動修改程式碼，決定保留哪一段，並刪除那些特殊符號 (`<<<`, `===`, `>>>`)。
        
    4. 執行 `git add <filename>` 標記為已解決。
        
    5. 執行 `git commit` 完成合併。
        

#### 2. 環境設定檔被覆蓋

`dev` 環境的設定 (如資料庫連線字串、API Key) 不小心覆蓋了 `main` 的正式環境設定。

- **解決：** 使用 `.gitignore` 排除設定檔 (如 `.env`)，或在程式碼中依賴環境變數 (Environment Variables) 而非寫死的數值。
    

#### 3. 邏輯錯誤 (Logic Errors)

程式碼語法沒錯，merge 也成功了，但因為 `dev` 的新功能破壞了 `main` 原有的邏輯，導致系統崩潰。

- **解決：** 必須依靠單元測試 (Unit Tests) 或在 Merge 前進行完整的人工測試。
    

---

### 如何避免問題 (Best Practices)

為了讓流程更順暢，建議採取以下預防措施：

#### 1. 經常性同步 (Sync Often)

不要等 `dev` 開發了一個月才 merge 回 `main`。

- **做法：** 經常做「反向合併」。當你在 `dev` 開發時，如果 `main` 有更新，定期執行：
    
    Bash
    
    ```
    git checkout dev
    git merge main
    ```
    
    這能讓你在開發過程中就解決衝突，而不是在最後上線前才面對「衝突地獄」。
    

#### 2. 使用 "Squash and Merge"

你提到 `dev` 有「很多的 new commits」。如果在 `dev` 上有 `Update readme`, `Fix bug`, `Fix again` 這種瑣碎紀錄，直接 merge 會汙染 `main` 的歷史。

- **做法：** 在 GitHub PR 介面選擇 **Squash and merge**。這樣 `main` 只會多出一個乾淨的 commit：「Feature X 完成」，方便回溯。
    

#### 3. 設定 Branch Protection Rules (分支保護)

既然是 GitHub Private Repo，你可以設定保護機制。

- **做法：** 到 Repo 的 **Settings** -> **Branches** -> **Add rule**。
    
- 設定 `main` 為受保護分支，勾選 **"Require pull request reviews before merging"**。這樣就強制你必須走 PR 流程，防止自己手滑在 CLI 直接把壞掉的程式碼 push 到 `main`。
    

**接下來這一步我可以為您做：** 您提到現在有很多 new commits，如果您想嘗試用 **GitHub PR 的方式**進行，我可以為您提供一份**標準的 PR 描述模板 (Template)**，幫助您整理這次合併的更動內容，方便未來回顧或給協作者看。需要嗎？




#### Create Trivecto ImagingLibWatch github repo
```
echo "# ImagingLibWatch" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Trivecto/ImagingLibWatch.git
git push -u origin main


### …or push an existing repository from the command line

git remote add origin https://github.com/Trivecto/ImagingLibWatch.git
git branch -M main
git push -u origin main
```





#### 標準的 PR 描述模板 (Template)
```
提供一份標準的 PR 描述模板 (Template)，幫助您整理這次合併的更動內容，方便未來回顧或給協作者看. 另外我的目前dev branch還不是Release version. 進度大概是prototype product剛完成v2版本的interface, PR標題應該怎麼下? 並幫我整理常用的PR標題或不同階段, 有release, 還有甚麼
```
這是一份非常實用的資源整理。針對您的 **Prototype V2 Interface** 剛完成的情況，以及 **Private Repo** 的管理需求，我為您準備了以下三個部分的建議：

### 1. 針對您目前進度的 PR 標題建議

由於您提到這還不是正式的 Release Version，但完成了 Prototype V2 的介面，這屬於 **「里程碑式的可交付成果 (Milestone Deliverable)」**。合併到 `main` 代表這是一個相對穩定的階段性版本。

建議使用以下幾種格式（依推薦程度排序）：

- **選項 A（推薦 - 語意化且清晰）：** `feat(ui): Complete Prototype V2 Interface Implementation` _(解釋：`feat` 代表新功能，`(ui)` 標註範圍，後面清楚說明完成了 V2 介面實作)_
    
- **選項 B（強調版本推進）：** `Milestone: Prototype V2 Interface Update (Pre-release)` _(解釋：強調這是一個里程碑，並括號註明這是預發布版本)_
    
- **選項 C（簡單直白）：** `Merge dev: V2 Interface & New Features`
    

---

### 2. 標準 Pull Request (PR) 描述模板

這份模板使用了 Markdown 語法，您可以直接複製貼上到 GitHub 的 PR 描述框中。對於包含 UI 變更的專案（如您的 V2 介面），**截圖 (Screenshots)** 區塊特別重要。

Markdown

```
## 📝 變更摘要 (Summary)
本 PR 將 `dev` 分支合併至 `main`，主要完成了 Prototype V2 版本的介面重構，並整合了新的開發功能。

## 🚀 主要變更內容 (Changes)
- [New] 實作 V2 使用者介面 (包含新的配色與佈局)
- [New] 新增功能 A (例如：圖片上傳功能)
- [Update] 優化了資料庫連線邏輯
- [Fix] 修正了舊版 V1 介面在手機版跑版的問題

## 📸 截圖或錄影 (Screenshots / Demo)
| 修改前 (Before) | 修改後 (After - V2) |
| :---: | :---: |
| (貼上舊版截圖) | (貼上新版截圖) |

## 🧪 測試說明 (Testing)
- [ ] 已在本地環境 (Localhost) 執行並且編譯成功
- [ ] 測試了 V2 介面的主要按鈕與跳轉流程
- [ ] 確認沒有破壞現有的 API 功能

## ⚠️ 注意事項 (Notes)
- 合併後請執行 `pip install -r requirements.txt` 安裝新依賴。
- 此版本尚未包含金流功能，預計在 V2.1 加入。
```

---

### 3. 常用的 PR 標題前綴與階段分類

在軟體工程中，我們通常會使用 **Conventional Commits** (約定式提交) 的規範來命名 PR，這樣一眼就能看出這個 PR 的「性質」。除了 `Release` 之外，常見的還有以下幾種：

#### A. 開發階段 (Development Phase)

這是最常出現的類別，用於日常開發：

- **`feat: ...` (Feature / 新功能)**
    
    - 用途：新增功能。
        
    - 範例：`feat: add user login page` (新增使用者登入頁)
        
- **`fix: ...` (Bug Fix / 修復)**
    
    - 用途：修正錯誤。
        
    - 範例：`fix: resolve navigation bar alignment issue` (修復導覽列對齊問題)
        
- **`refactor: ...` (Refactoring / 重構)**
    
    - 用途：修改程式碼結構但不改變功能（例如優化效能、整理雜亂的程式碼）。
        
    - 範例：`refactor: optimize image processing algorithm` (優化影像處理演算法)
        
- **`style: ...` (Style / 格式)**
    
    - 用途：不影響程式邏輯的變更（改縮排、改變數命名、分號）。
        
    - 範例：`style: formatting code with pylint`
        

#### B. 維運與雜項 (Operations & Chores)

- **`chore: ...` (Chore / 雜務)**
    
    - 用途：不更動原始碼的修改（更新依賴套件、修改 `.gitignore`、設定檔）。
        
    - 範例：`chore: update numpy version to 1.21`
        
- **`docs: ...` (Documentation / 文件)**
    
    - 用途：只修改了文件（README, Wiki）。
        
    - 範例：`docs: update installation guide in README`
        

#### C. 特殊緊急情況

- **`hotfix: ...` (Hotfix / 緊急修復)**
    
    - 用途：通常指已經在 `main` 或生產環境上發現嚴重 Bug，需要立即修復並插隊合併。
        
    - 範例：`hotfix: fix critical crash on startup`
        

#### D. 發布階段 (Release Phase)

當累積了一定程度的 `feat` 和 `fix` 後，就會進行 Release。

- **`release: ...`**
    
    - 範例：`release: v1.0.0` (正式發布)
        
    - 範例：`release: v0.2.0-beta` (測試版發布)
        
- **`build: ...`**
    
    - 用途：影響建置系統或外部依賴的更動。
        
    - 範例：`build: upgrade docker image version`
        

### 總結您的情況

既然您的 `main` 目前累積了許多來自 `dev` 的更動，且包含介面更新，我建議您的 PR 標題使用： **`feat(ui): Prototype V2 Interface update and core features merge`** 這樣既專業，又能清楚交代這次合併的重點。
