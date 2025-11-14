
1️⃣ 理解 `git fetch` vs `git pull` 差異；  
2️⃣ 如何在 `dev` branch 開發；  
3️⃣ 如何多人協作、同步與合併回 `main`。

我來用**清楚的層次結構 + 實際指令範例**一步步說明。

---

## 🧭 一、`git fetch` vs `git pull` 的差別

這兩個命令是 Git 同步遠端更新的兩種方式，  
但它們**行為不一樣**。

|指令|主要作用|是否會自動合併到本地分支|適用情境|
|---|---|---|---|
|🟡 `git fetch`|從遠端下載最新的 commit、branch、tag 資訊|❌ 不會自動合併|想先看遠端變化，不立即改動自己版本|
|🟢 `git pull`|= `git fetch` + `git merge`（自動合併）|✅ 自動把遠端的更新合併到當前分支|想直接更新自己 local 分支與遠端同步|

---

### 🔍 用圖理解差別

假設你目前在本地 `dev`：

`本地 dev: A — B — C 遠端 dev: A — B — C — D`

#### `git fetch`

只會把 D 下載下來，存放在「遠端追蹤分支」中：

`本地 dev: A — B — C origin/dev: A — B — C — D`

但你的本地 `dev` 不會動。

你可以再手動比較差異、決定是否合併：

`git diff dev origin/dev git merge origin/dev`

#### `git pull`

會直接幫你執行上面兩步：

`本地 dev: A — B — C — D`

=> 你的本地就更新成和遠端一致。

---

### ✅ 建議用法：

- **`git fetch`**：用於安全檢查、多人協作前同步（不改動本地）
    
- **`git pull`**：用於確定要合併遠端更新時