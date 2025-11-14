
## 你目前情境：有 `main` 和 `dev` 兩個分支

這是非常標準的開發模式 👇

`main → 穩定版本（release） dev  → 開發中版本（多人開發）`

---

![[Pasted image 20251114103236.png]]

### ✅ 你的日常開發流程（以自己在 local `dev` 為例）

1️⃣ 切到開發分支

```python
git checkout dev
```

2️⃣ 開始寫 code、測試、修改

3️⃣ 查看修改狀況
```python
git status
```


4️⃣ 加入改動並 commit
```python
git add .     # 添加文件到暂存区

git commit -m   # 将暂存区内容添加到仓库中
```


5️⃣ 把這些改動推送到遠端的 `dev`
```python
git push
```


> 如果是第一次 push 這個分支，可用：
> `-u` 會建立本地與遠端的追蹤關係，之後就能只打 `git push`。

```python
git push -u origin dev
```
---

### ✅ 定期同步團隊最新進度

因為多人都在 `dev` 上改：

```python
git pull
```
這樣會把遠端的更新合併進你的 local dev。  
⚠️ 若有 conflict（衝突），手動解決再 `git add`、`git commit`。

---

## 🧩 三、到 main branch 的合併流程

通常是：
```python
remote dev branch   #→ merge 到 remote `main` branch
```
### 有兩種常見做法：

#### 🟢 方式 1：在 GitHub 上用 Pull Request (最常見)

1. 每個人 push 自己的開發到 remote `dev`
    
2. 一位負責人（或 CI）在 GitHub 開 PR：
    
    > from: `dev` → to: `main`

1. 經 review / 測試後，Merge PR  
    → main 就有新版本。
    

#### 🔵 方式 2：在本地操作

1. 切回 main
    ```python
	git git checkout main
```
1. 更新 main
    
    ```python
	git git pull origin main
```
1. 合併 dev

    ```python
	git git merge dev
```
1. 推送更新
    ```python
	git push origin main
```

---

## 👥 四、多名開發者協作情況（2~3人）

是的，通常：

- 每個人都在同一個 remote `dev` branch 工作；
- 各自有 local `dev` 分支；
- 各自 push 到 remote `dev`；
- 最後再從 remote `dev` 合併到 remote `main`。

### 💡 協作時注意：

|任務|指令|說明|
|---|---|---|
|同步其他人更新|`git pull`|開發前一定先做，避免衝突|
|提交自己修改|`git add . && git commit -m "..." && git push`|建立小而清晰的 commit|
|查看誰改過什麼|`git log --oneline --graph --decorate`|可視化 commit 歷史|
|有衝突時解決|編輯檔案 → `git add .` → `git commit`|再繼續 push|
|查看分支狀態|`git branch -a`|顯示本地 + 遠端分支|

---

## 📋 五、整體工作流程圖（多人協作）

```
遠端 repo
│
├── main   ← 最穩定版本（release）
│
└── dev    ← 多人協作開發分支
       ↑       ↑
       │       │
   laptopA   desktopB
   (local dev) (local dev)

```

日常流程：  
1️⃣ 每個人先 `git pull origin dev`  
2️⃣ 改 code → `git add . && git commit -m "..."`  
3️⃣ 上傳 → `git push`  
4️⃣ 最後由其中一人或 CI → 把 `dev` merge 到 `main`。

---

## 🧠 六、最重要的 Git 指令清單

|功能|指令|備註|
|---|---|---|
|更新遠端分支資訊|`git fetch`|不改變本地|
|同步遠端更新|`git pull`|fetch + merge|
|提交修改|`git add . && git commit -m "msg"`||
|上傳修改|`git push`||
|切換分支|`git checkout branch_name`||
|建立新分支|`git checkout -b dev`||
|合併分支|`git merge dev`||
|刪除分支|`git branch -d dev`||
|查看分支狀態|`git branch -a`||
|檢查紀錄|`git log --oneline --graph --decorate`||
|比較差異|`git diff branch1 branch2`||
|解決衝突後繼續|`git add . && git commit`||

---

## ✅ 七、簡短流程摘要（適用團隊開發）

1️⃣ 每天工作前：
```python
git checkout dev

git pull origin dev
```


2️⃣ 寫完功能：
```python
`git add . 

git commit -m "新增功能A"

git push
```
3️⃣ 功能測試穩定後：

```python
git checkout main
git pull origin main
git merge dev
git push origin main
```
4️⃣ （可選）清除暫時分支：

`git branch -d dev`
```python
git branch -d dev
```