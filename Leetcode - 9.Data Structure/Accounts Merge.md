Lintcode 1070
给定一个帐户列表，每个元素`accounts [i]`是一个字符串列表，其中第一个元素`accounts [i] [0]`是账户名称，其余元素是这个帐户的电子邮件。  
现在，我们想合并这些帐户。  
如果两个帐户有相同的电子邮件地址，则这两个帐户肯定属于同一个人。  
请注意，即使两个帐户具有相同的名称，它们也可能属于不同的人，因为两个不同的人可能会使用相同的名称。  
一个人可以拥有任意数量的账户，但他的所有帐户肯定具有相同的名称。  
合并帐户后，按以下格式返回帐户：每个帐户的第一个元素是名称，其余元素是**按字典序排序**后的电子邮件。  
帐户本身可以按任何顺序返回。


```python
"""
样例 1:
	输入:
	[
		["John", "johnsmith@mail.com", "john00@mail.com"],
		["John", "johnnybravo@mail.com"],
		["John", "johnsmith@mail.com", "john_newyork@mail.com"],
		["Mary", "mary@mail.com"]
	]
	
	输出: 
	[
		["John", 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com'],
		["John", "johnnybravo@mail.com"],
		["Mary", "mary@mail.com"]
	]

	解释: 
	第一个第三个John是同一个人的账户，因为这两个账户有相同的邮箱："johnsmith@mail.com".
	剩下的两个账户分别是不同的人。因为他们没有和别的账户有相同的邮箱。

	你可以以任意顺序返回结果。比如：
	
	[
		['Mary', 'mary@mail.com'],
		['John', 'johnnybravo@mail.com'],
		['John', 'john00@mail.com', 'john_newyork@mail.com', 'johnsmith@mail.com']
	]
	也是可以的。
```


```python
class Solution:
    def accountsMerge(self, accounts):
        self.initialize(len(accounts))
        email_to_ids = self.get_email_to_ids(accounts)
        
        # union
        for email, ids in email_to_ids.items():
            root_id = ids[0]
            for id in ids[1:]:
                self.union(id, root_id)
                
        id_to_email_set = self.get_id_to_email_set(accounts)
        
        merged_accounts = []
        for user_id, email_set in id_to_email_set.items():
            merged_accounts.append([
                accounts[user_id][0],
                *sorted(email_set),
            ])
        return merged_accounts
    
    def get_id_to_email_set(self, accounts):
        id_to_email_set = {}
        for user_id, account in enumerate(accounts):
            root_user_id = self.find(user_id)
            email_set = id_to_email_set.get(root_user_id, set())
            for email in account[1:]:
                email_set.add(email)
            id_to_email_set[root_user_id] = email_set
        return id_to_email_set
            
    def get_email_to_ids(self, accounts):
        email_to_ids = {}
        for i, account in enumerate(accounts):
            for email in account[1:]:
                email_to_ids[email] = email_to_ids.get(email, [])
                email_to_ids[email].append(i)
        return email_to_ids
        
    def initialize(self, n):
        self.father = {}
        for i in range(n):
            self.father[i] = i
            
    def union(self, id1, id2):
        self.father[self.find(id1)] = self.find(id2)

    def find(self, user_id):
        path = []
        while user_id != self.father[user_id]:
            path.append(user_id)
            user_id = self.father[user_id]
            
        for u in path:
            self.father[u] = user_id
            
        return user_id
```
pass


# **LintCode 1070: Accounts Merge 解法詳細解析**

## **問題描述**

在這道題目中，我們有多個**帳戶列表**，每個帳戶包含：

- **使用者名稱**
- **與該帳戶關聯的一組電子郵件**

同一個人可能會出現在多個帳戶中，而如果兩個帳戶**共用相同的電子郵件**，我們就認為它們屬於**同一個人**，需要合併。

---

## **解法分析**

這個問題的核心是 **找出所有帳戶的連通分量（Connected Components）**，並合併這些帳戶。我們可以將它視為**無向圖的連通分量問題**，並用**並查集（Union-Find）** 來解決：

1. **構建帳戶關係圖**
    
    - **節點**：每個帳戶（基於索引 `i`）。
    - **邊**：如果兩個帳戶共享相同的電子郵件，則這些帳戶應該屬於同一組。
    - **方式**：建立 **email → 帳戶 ID 列表的映射**，然後將所有共享該 email 的帳戶**合併（Union）**。
2. **使用並查集（Union-Find）來合併帳戶**
    
    - 透過 `find()` 確定兩個帳戶是否已經在同一集合。
    - 透過 `union()` 合併帳戶，使它們共享同一個 root ID。
3. **整理結果**
    
    - 建立 **ID → 所有電子郵件的對應關係**。
    - 將每個連通分量的電子郵件進行排序，並附上使用者名稱。

---

## **解法步驟**

### **Step 1: 建立 `email_to_ids`**

這個字典 `email_to_ids` 用來記錄每個電子郵件出現在哪些帳戶裡：

python

複製編輯

`email_to_ids = self.get_email_to_ids(accounts)`

**遍歷所有帳戶：**

nginx

複製編輯

`accounts = [     ["John", "johnsmith@mail.com", "john00@mail.com"],     ["John", "johnnybravo@mail.com"],     ["John", "johnsmith@mail.com", "john_newyork@mail.com"],     ["Mary", "mary@mail.com"] ]`

結果：

python

複製編輯

`email_to_ids = {     "johnsmith@mail.com": [0, 2],     "john00@mail.com": [0],     "johnnybravo@mail.com": [1],     "john_newyork@mail.com": [2],     "mary@mail.com": [3] }`

這表示：

- `johnsmith@mail.com` 出現在帳戶 `0` 和 `2`
- `john00@mail.com` 只出現在帳戶 `0`
- `johnnybravo@mail.com` 只出現在帳戶 `1`
- `john_newyork@mail.com` 只出現在帳戶 `2`
- `mary@mail.com` 只出現在帳戶 `3`

---

### **Step 2: 透過並查集合併帳戶**

我們遍歷 `email_to_ids`，對於每個 email，將所有擁有該 email 的帳戶 ID 進行 `union`。

python

複製編輯

`for email, ids in email_to_ids.items():     root_id = ids[0]     for id in ids[1:]:         self.union(id, root_id)`

這樣，我們會將**共享電子郵件的帳戶 ID 合併**，形成連通分量。

- `johnsmith@mail.com` 將 `0` 和 `2` 合併，現在 `0` 和 `2` 屬於同一個集合。
- 其他帳戶保持獨立。

最終的 `father` 結構：

yaml

複製編輯

`{ 0: 2, 1: 1, 2: 2, 3: 3 }`

這表示：

- `帳戶 0` 和 `帳戶 2` 已合併，根為 `2`
- `帳戶 1` 和 `帳戶 3` 各自獨立

---

### **Step 3: 建立 `id_to_email_set`**

python

複製編輯

`id_to_email_set = self.get_id_to_email_set(accounts)`

我們遍歷每個帳戶：

1. 找到該帳戶的 `root ID`（代表帳戶）。
2. 將該帳戶的所有電子郵件加入 `id_to_email_set[root ID]`。

計算過程：

perl

複製編輯

`帳戶 0 → root ID = 2 → 加入 {"johnsmith@mail.com", "john00@mail.com"} 帳戶 1 → root ID = 1 → 加入 {"johnnybravo@mail.com"} 帳戶 2 → root ID = 2 → 加入 {"johnsmith@mail.com", "john_newyork@mail.com"} 帳戶 3 → root ID = 3 → 加入 {"mary@mail.com"}`

合併後：

python

複製編輯

`id_to_email_set = {     2: {"johnsmith@mail.com", "john00@mail.com", "john_newyork@mail.com"},     1: {"johnnybravo@mail.com"},     3: {"mary@mail.com"} }`

這表示：

- `帳戶 0` 和 `帳戶 2` 合併成 `帳戶 2`，它擁有 `"johnsmith@mail.com", "john00@mail.com", "john_newyork@mail.com"`
- `帳戶 1` 和 `帳戶 3` 各自獨立。

---

### **Step 4: 格式化輸出**

python

複製編輯

`merged_accounts = [] for user_id, email_set in id_to_email_set.items():     merged_accounts.append([         accounts[user_id][0],  # 使用者名稱         *sorted(email_set),    # 排序電子郵件     ])`

輸出：

python

複製編輯

`[     ["John", "john00@mail.com", "john_newyork@mail.com", "johnsmith@mail.com"],     ["John", "johnnybravo@mail.com"],     ["Mary", "mary@mail.com"] ]`

---

## **時間複雜度分析**

|步驟|複雜度|
|---|---|
|建立 `email_to_ids`|`O(NM)`|
|進行 `Union-Find` 合併|`O(NM α(N)) ≈ O(NM)`|
|建立 `id_to_email_set`|`O(NM)`|
|格式化輸出（排序）|`O(NM log M)`|
|**總計**|`O(NM log M)`|

- `N`：帳戶數量
- `M`：平均每個帳戶的電子郵件數
- `α(N)`：阿克曼函數的反函數，近似 `O(1)`

這個算法可以高效處理大量帳戶與電子郵件的合併問題。

---

## **其他解法**

### **1. 使用 DFS 來遍歷連通分量**

- 建立 **`email → email`** 圖，每個帳戶內的電子郵件形成一個無向圖。
- 遍歷所有未訪問的電子郵件，通過 DFS 來標記所有相連的電子郵件，並合併到同一個帳戶。
- **時間複雜度：`O(NM log M)`**，與並查集類似。

### **2. 使用 BFS 來遍歷連通分量**

- 與 DFS 方法相似，但使用隊列來進行廣度優先搜尋。
- **時間複雜度：`O(NM log M)`**，與 DFS 類似。

---

## **總結**

- **最佳解法**：並查集（Union-Find）+ 哈希表，`O(NM log M)`。
- **其他解法**：DFS 或 BFS 也可以解，但通常效率相似。
- **並查集優勢**：更容易擴展，且可用於更多類型的連通分量問題。

這道題的核心思想是**圖的連通性**，並查集在這類問題上能夠提供高效的解法