

|                           |     |
| ------------------------- | --- |
| [[#### Git 某一側未同步而要遺棄改動]] |     |
| [[#### Git 要使用到過去Commit]] |     |
|                           |     |
|                           |     |


#### Git 某一側未同步而要遺棄改動
```

git merge --abort
git fetch --all
git reset --hard origin/main

git reset --hard origin/dev

```



#### Git 要使用到過去Commit
```
切換到特定commit
[1]  git pull
[2]  git log origin/dev --oneline --decorate       (按q就會跳出了)
[3]  git switch --detach xxxxxnumber    or   git checkout xxxxxnumber 
[4]  切回舊版本   git switch dev     or    git checkout dev
[5]  git pull origin dev

[6]  git checkout -b temp_fix_from_old 56aac4f
[7]

把特定commit存在另一個folder
[1]  git pull
[2]  git log origin/dev --oneline --decorate
[3]  git worktree add “D:\folder”  xxxxxnumber 
git worktree remove - -force "D:\folder"

把特定commit直接複製到另一個folder
[1]  git clone D:\your_repo_path D:\compare_repo_copy
[2]  cd D:\compare_repo_copy
[3]  git checkout a1b2c3d

把舊的commit復原成最新的並push
[1]  git switch dev
[2]  git pull origin dev
[3]  git revert --no-commit 445335f..HEAD
[4]  git commit -m "Restore dev to state of commit 445335f"
[5]  git push origin dev

其他
[1] 查目前的remote repo and branch:
[2] 查目前最新的commit:
[3]
```




#### 一個branch從另一個branch分出去
```
譬如我這個refactor/watch-point-multi-image是在dev某個commit之後從dev分支出來的, 而我現在在refactor/watch-point-multi-image. 有沒有指令可以查到是從哪個branch 哪個commit之後分支的? 還有我現在要把refactor/watch-point-multi-image merge回到dev branch 要怎麼做呢

==== 查 branch 從哪裡分出去 ====
git merge-base dev refactor/watch-point-multi-image  
git log --oneline --graph --decorate --all  
git log --oneline dev..refactor/watch-point-multi-image

==== 合回 dev ====
git fetch origin  
git checkout dev         # 先切到你本地的 `dev`
git pull origin dev  
git merge --no-ff refactor/watch-point-multi-image  # 再把你本地的合併進你本地的 `dev`
git push origin dev

==== 合回 dev並刪除feature branch ====
git status  
git fetch origin  
git checkout dev  
git pull origin dev  
git merge --no-ff refactor/watch-point-multi-image  
git push origin dev  
git branch -d refactor/watch-point-multi-image  
git push origin --delete refactor/watch-point-multi-image```


```


我先直接給你最實用的 Git 查法與 merge 流程，重點是「怎麼推測這個 branch 從哪裡分出來」以及「安全地合回 dev」。
但要先講一個很重要的觀念：

**Git 不會正式記錄「這個 branch 當初是從哪個 branch 建出來的」**。  
所以你查到的通常是：

- 它**最可能**從哪個 branch 分出來
    
- 它和某 branch 的**共同祖先 commit** 是哪個
    

這在實務上已經夠用了。

---

# 一、怎麼查 `refactor/watch-point-multi-image` 大概是從哪裡分出來的

假設你懷疑它是從 `dev` 分出去的，最有用的是這個：

git merge-base dev refactor/watch-point-multi-image

這會回傳兩個 branch 的**共同祖先 commit**。

例如輸出：

abc1234

表示：

- `dev`
    
- `refactor/watch-point-multi-image`
    

這兩條線最後共同的點是 `abc1234`

通常這就代表：  
**你的 feature branch 很可能是在這個 commit 之後，從 `dev` 分出去的。**

---

## 想看那個 commit 是什麼內容

git show abc1234

或直接：

git log --oneline --decorate -n 20

---

## 最直觀：看整個分支圖

這個非常推薦：

git log --oneline --graph --decorate --all

你會看到像這樣：

*  d6ee4cb (HEAD -> refactor/watch-point-multi-image, origin/refactor/watch-point-multi-image) ...  
*  e123456 ...  
*  f234567 ...  
| *  aaaaaaa (dev) ...  
|/  
*  abc1234 ...  
*  9999999 ...

這裡通常可以看出：

- `abc1234` 是共同祖先
    
- `refactor/watch-point-multi-image` 是在那之後長出去的
    
- `dev` 則走另一條線
    

---

## 想看「feature branch 比 dev 多了哪些 commit」

git log --oneline dev..refactor/watch-point-multi-image

意思是：

> 列出在 `refactor/watch-point-multi-image` 裡有、但 `dev` 沒有的 commit

這對你 merge 前非常有用。

---

## 想看「dev 比 feature branch 多了哪些 commit」

git log --oneline refactor/watch-point-multi-image..dev

這可以檢查：

- 你分支出去後，`dev` 有沒有繼續前進
    
- merge 回去時會不會比較容易 conflict
    

---

## 更精準一點：找 feature branch 的第一個 commit

你可以用：

git rev-list --reverse dev..refactor/watch-point-multi-image | head -n 1

在 Windows PowerShell 可用：

git rev-list --reverse dev..refactor/watch-point-multi-image | Select-Object -First 1

這會找出：

> feature branch 相對於 dev 的第一個獨有 commit

這常常就是你真正開始做 `watch-point-multi-image` 改動的第一個 commit。

---

## 另一個常用查法：reflog

如果這個 branch 是你本機建立的，而且 reflog 還在，有時能直接看到：

git reflog show refactor/watch-point-multi-image

有機會看到像：

<sha> branch: Created from dev

但這不是每次都有，所以不能完全依賴。

---

# 二、你現在要 merge 回 `dev`，怎麼做最穩

我先給你**最標準安全流程**。

假設你目前人在：

refactor/watch-point-multi-image

---

## 做法 1：標準 merge 回 dev

### Step 1 先確認工作目錄乾淨

git status

如果有未提交修改，先 commit 或 stash。

---

### Step 2 抓最新 remote

git fetch origin

---

### Step 3 切到 dev

git checkout dev

---

### Step 4 先把 dev 更新到最新

如果你的 remote `dev` 才是團隊主線，建議先同步：

git pull origin dev

---

### Step 5 把 feature branch merge 進 dev

git merge refactor/watch-point-multi-image

如果沒有衝突，Git 會自動完成 merge。

如果有衝突，先解 conflict，再：

git add .  
git commit

---

### Step 6 推上 remote

git push origin dev

---

# 三、如果你想保留完整 branch 歷史，建議這樣 merge

如果你希望之後還清楚知道這個 feature branch 曾經存在過，可以用：

git merge --no-ff refactor/watch-point-multi-image

這樣即使 Git 可以 fast-forward，也會特別建立一個 merge commit。

好處是歷史更清楚：

- 以後一看就知道這批改動是 `refactor/watch-point-multi-image` 合進來的
    
- 很適合 feature branch workflow
    

對你這種做大型 refactor 的情境，我通常會比較推薦這個。

完整流程：

git fetch origin  
git checkout dev  
git pull origin dev  
git merge --no-ff refactor/watch-point-multi-image  
git push origin dev

---

# 四、merge 前我很建議你先做這兩個檢查

## 1. 看差異

git diff dev..refactor/watch-point-multi-image

這可以直接看：

> feature branch 相比 dev 改了哪些檔案和內容

---

## 2. 看 commit 列表

git log --oneline dev..refactor/watch-point-multi-image

這可以確認：

- 你是不是只帶入想帶的 commit
    
- 有沒有混進不相關改動
    

---

# 五、如果 `dev` 已經前進很多，merge 前可先把 feature branch 跟 dev 對齊

這不是必須，但常常有幫助。

你現在在 feature branch 上時，可以先：

git checkout refactor/watch-point-multi-image  
git fetch origin  
git merge origin/dev

這樣你先在 feature branch 上解掉來自 `dev` 的衝突，確認功能正常後，再 merge 回 `dev``。`

另一種做法是 rebase，但對你目前這種分支整合，我會建議先用 merge，比較安全、也比較不容易改寫歷史。

---

# 六、最常用的一套指令，直接給你

## 查 branch 從哪裡分出去

git merge-base dev refactor/watch-point-multi-image  
git log --oneline --graph --decorate --all  
git log --oneline dev..refactor/watch-point-multi-image

## 合回 dev

git fetch origin  
git checkout dev  
git pull origin dev  
git merge --no-ff refactor/watch-point-multi-image  
git push origin dev

---

# 七、你的情境下，我最推薦的實務流程

因為你這個 branch 看起來是較大的 refactor，我會建議：

git fetch origin  
git log --oneline --graph --decorate --all  
git diff dev..refactor/watch-point-multi-image  
git checkout dev  
git pull origin dev  
git merge --no-ff refactor/watch-point-multi-image  
git push origin dev

這樣最穩，也最清楚。

---

# 八、補一句最精準的回答

你問的是：

> 有沒有指令可以查到是從哪個 branch 哪個 commit 之後分支的？

精準答案是：

- **哪個 commit**：可以，用 `git merge-base`
    
- **哪個 branch**：Git 通常**不能保證正式查出**，只能從歷史圖和 reflog 高度推測





