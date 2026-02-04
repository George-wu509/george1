
|                        |     |
| ---------------------- | --- |
| [[#### Github公司帳號跟權限]] |     |
|                        |     |
|                        |     |
|                        |     |


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