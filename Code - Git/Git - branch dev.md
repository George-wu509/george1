
你現在已經有一個典型的團隊開發結構：

> 有一個 remote GitHub repo + 兩台電腦（desktop、laptop）共用同一個專案。  
> 想要建立開發分支（`dev`）→ 測試 → 最後 merge 回主分支（`main`）。

這是一個非常標準、專業的開發流程。  
以下我會用完整步驟說明，包含所有常見陷阱、每台電腦該做什麼、以及最後 merge 流程。

---

## 🧩 一、目前狀況確認

假設：

- Remote repo：`https://github.com/yourname/project.git`
    
- 目前所有電腦的 branch 都是 `main`
    
- Remote 代號是 `origin`
    

---

## 🚀 二、在 Desktop 建立新的 `dev` 分支

### Step 1️⃣：切到最新的 main

`git checkout main git pull origin main`

確保你 local 的 main 是最新版本。

---

### Step 2️⃣：從 main 建立新分支 dev

`git checkout -b dev`

這行等於：「從目前 main 的狀態開出一個新分支 dev 並切過去」。

---

### Step 3️⃣：將 dev 分支推送到 GitHub

`git push -u origin dev`

說明：

- `-u` 是設定「追蹤關係」(upstream tracking)  
    → 之後在 dev 分支上可直接用 `git push`、`git pull`。
    

成功後 GitHub 會出現一個新的 branch：`dev`。

---

## 💻 三、在另一台 Laptop 也切換到 dev 分支工作

### Step 1️⃣：先更新所有 remote branch 資訊

`git fetch origin`

### Step 2️⃣：切換到 dev 分支

`git checkout dev`

> ⚠️ 如果出現錯誤：「branch 'dev' set up to track remote branch 'dev' from 'origin'」的提示，  
> 表示已自動追蹤成功，不用再設定。

若沒有自動追蹤，可手動設定：

`git checkout -b dev origin/dev`

---

### Step 3️⃣：確認目前所在分支

`git branch`

會看到：

`* dev   main`

星號代表目前在 dev。

從這之後：

- 你在 **Desktop**、**Laptop** 上修改的內容都會在 dev branch。
    
- 使用 `git push` / `git pull` 都會同步 dev 分支，不會影響 main。
    

---

## 🧱 四、開發階段：多人在 dev branch 工作

### 一般流程

1. 兩台機器都在 `dev` 上開發；
    
2. 任何一台修改完後：
    
    `git add . git commit -m "你的修改說明" git push`
    
3. 另一台要接續工作前先：
    
    `git pull`
    
    以免版本衝突。
    

✅ **關鍵原則**：永遠 `pull` 最新的 dev 後再開始工作。

---

## 🔄 五、確認 code 沒問題後，merge dev → main

### Step 1️⃣：切回 main 分支

`git checkout main`

### Step 2️⃣：更新 main（確保與遠端同步）

`git pull origin main`

### Step 3️⃣：合併 dev 分支

`git merge dev`

> 若有衝突（conflict），Git 會提示衝突的檔案，手動解決後再：
> 
> `git add . git commit`

---

### Step 4️⃣：推送更新的 main 到 GitHub

`git push origin main`

---

### ✅ Step 5️⃣：（選擇性）刪除 dev 分支

如果這個 dev 階段結束了：

`git branch -d dev          # 刪除 local dev git push origin --delete dev   # 刪除 remote dev`

未來要再開新的開發分支，就再從 main 建一個新的 `dev`。

---

## 💡 實務建議與注意事項

|項目|建議做法|原因|
|---|---|---|
|開新分支|永遠從最新的 main 建立|避免過時基礎|
|每次工作前|先 `git pull`|避免衝突|
|commit message|清楚說明修改內容|方便追蹤|
|merge 時機|功能測試穩定、通過檢查|保持 main 穩定|
|merge 方法|可用 PR (Pull Request)|若在 GitHub 團隊開發，更安全、可審查|
|backup|若重要，merge 前可開 release tag|防止回不去|

---

## 📋 簡化流程摘要圖

`(main) ←───┐            │            ├──> (dev) ←─── Desktop 開發            │         ↑            │         └── Laptop 同步開發            │ merge 回 ──┘`