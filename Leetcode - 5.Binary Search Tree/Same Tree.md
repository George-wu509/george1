
相同的樹 Same Tree

LeetCode 100 / LintCode 469 給定兩棵二元樹的根節點 `p` 和 `q`，編寫一個函數來檢驗這兩棵樹是否相同。 如果兩棵樹在結構上相同，並且節點具有相同的值，則認為它們是相同的。

**样例 1：** 输入：
```Python
"""
p = [1, 2, 3], q = [1, 2, 3]
    1         1
   / \       / \
  2   3     2   3
"""
```

输出：
```Python
True
```

解释： 兩棵樹的結構完全一致，且對應位置的節點值也完全相同。

**样例 2：** 输入：
```Python
"""
p = [1, 2], q = [1, None, 2]
    1         1
   /           \
  2             2
"""
```

输出：
```Python
False
```

解释： 雖然節點值都有 1 和 2，但 `p` 的 2 在左子樹，`q` 的 2 在右子樹，結構不同。

**样例 3：** 输入：
```Python
"""
p = [1, 2, 1], q = [1, 1, 2]
    1         1
   / \       / \
  2   1     1   2
"""
```

输出：
```Python
False
```

解释： 兩棵樹的結構相同，但對應位置上的節點值不同。

```Python
from typing import Optional

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        # 兩個節點都為空，代表這條路徑比對完成且相同
        if not p and not q:
            return True
        # 其中一個為空，或兩個節點的值不相等，直接回傳 False
        if not p or not q or p.val != q.val:
            return False
        # 遞迴比對左子樹與右子樹，必須兩邊都相同才回傳 True
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```


```Python
from collections import deque
from typing import Optional

class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        queue = deque([(p, q)])
        
        while queue:
            node1, node2 = queue.popleft()
            
            # 兩者皆空，跳過繼續比對其他節點
            if not node1 and not node2:
                continue
            # 其中一者為空，或節點值不同
            if not node1 or not node2 or node1.val != node2.val:
                return False
                
            # 將對應的左右子節點成對放入佇列
            queue.append((node1.left, node2.left))
            queue.append((node1.right, node2.right))
            
        return True
```

解釋 (DFS 遞迴解法): step1: 同時從兩棵樹的根節點 `p` 與 `q` 開始同步遞迴遍歷 step2: 終止條件判斷：若 `p` 與 `q` 皆為 `None`，表示到底了且結構一致，回傳 `True` step3: 若只有其中一個為 `None`（結構不同），或 `p.val != q.val`（數值不同），直接回傳 `False` step4: 假設 `p = [1, 2, 3]`, `q = [1, 2, 3]`，根節點 `1 == 1` 通過檢查 step5: 遞迴檢查左子樹 `isSameTree(2, 2)` -> `2 == 2`，其左右子節點皆為 `None` -> 回傳 `True` step6: 遞迴檢查右子樹 `isSameTree(3, 3)` -> `3 == 3`，其左右子節點皆為 `None` -> 回傳 `True` step7: 左右子樹結果做 `and` 運算 (`True and True`)，最終回傳 `True`

解釋 (核心觀念): 判斷兩棵樹相同，等價於拆解為三個子問題同時成立：

1. 當前兩個根節點 `p` 和 `q` 的值相等
    
2. `p` 的左子樹與 `q` 的左子樹相同
    
3. `p` 的右子樹與 `q` 的右子樹相同
    

### **LeetCode 100 - Same Tree（相同的樹）**

這道題目是 Blind 75 與 NeetCode 150 中最基礎的 **二元樹遞迴（DFS）** 入門題，同時也是進階題（如「對稱二元樹」、「另一棵樹的子樹」）的核心子程序。問題要求同步遍歷兩棵樹，驗證**拓撲結構**與**節點數值**是否完全一致。

## **解法解析**

這道題可以轉化為 **同步分治（Divide and Conquer）** 來求解，將整棵樹的比較拆解為「當前節點 + 左子樹 + 右子樹」的比較。

### **思路**

1. **觀察邊界與終止條件（Base Cases）**
    
    - 比較兩個節點 `p` 和 `q` 時，共有四種狀態組合：
        
        1. `p` 為空，且 `q` 為空 → **合法**，代表該分支同步走到了盡頭，回傳 `True`。
            
        2. `p` 為空，但 `q` 不為空 → **不合法**，結構不對稱，回傳 `False`。
            
        3. `p` 不為空，但 `q` 為空 → **不合法**，結構不對稱，回傳 `False`。
            
        4. `p` 與 `q` 皆不為空 → 進一步比較 `p.val == q.val`，若不相等回傳 `False`。
            
2. **邏輯精簡技巧**
    
    - 只要先寫 `if not p and not q: return True`，就能排除「兩者皆空」的情況。
        
    - 接著寫 `if not p or not q:` 時，就必定代表「其中一個為空、另一個不為空」，可直接與 `p.val != q.val` 合併成一行判斷，代碼非常乾淨。
        
3. **代碼執行流程**
    
    - 檢查當前 `p` 與 `q` 是否觸發終止條件（全空回傳 `True`，單空或值不同回傳 `False`）。
        
    - 當前層級驗證通過後，向下遞迴呼叫 `isSameTree(p.left, q.left)` 與 `isSameTree(p.right, q.right)`。
        
    - 利用 Python 的短路求值（Short-circuit evaluation），只要左子樹回傳 `False`，右子樹就不會再繼續遍歷，直接向上回傳 `False`。
        

## **具體舉例**

假設輸入為範例 2 的不對稱樹 `p = [1, 2]`, `q = [1, None, 2]`，DFS 遞迴呼叫流程如下：

|遞迴層級|當前 `p` 節點|當前 `q` 節點|條件判斷結果|動作 / 回傳值|
|---|---|---|---|---|
|第 1 層（根）|`Node(1)`|`Node(1)`|皆非空且 `1 == 1`|繼續比對左右子樹|
|第 2 層（左）|`Node(2)`|`None`|`not q` 成立（單邊為空）|直接回傳 `False`|
|第 2 層（右）|`None`|`Node(2)`|因左子樹已為 `False`，短路不執行|—|
|第 1 層（彙總）|—|—|`False and (略)`|最終回傳 `False`|

最後返回 `False`，在發現結構差異的第一時間便提前結束遍歷。

## **時間與空間複雜度分析**

- **時間複雜度：O(min(N, M))**
    
    - 其中 `N` 和 `M` 分別為兩棵樹的節點數。因為只要遇到任何一個節點不匹配就會立刻終止，最多只會遍歷節點數較少的那棵樹，最壞情況（兩樹完全相同）下為 **O(N)**。
        
- **空間複雜度：O(min(H1, H2))**
    
    - 其中 `H1` 和 `H2` 分別為兩棵樹的高度，空間開銷來自遞迴呼叫棧（Recursion Stack）或 BFS 佇列。
        
    - 在平衡二元樹的情況下為 **O(log N)**；在最壞情況（退化成單邊鏈表的斜樹）下為 **O(N)**。
        

## **其他解法**

### **1. BFS 同步層序遍歷（Queue 成對比對）**

- **思路**：
    
    - 使用佇列 `deque`，每次將成對的節點 `(p, q)` 放入佇列中。
        
    - 每次彈出一對節點 `(node1, node2)`，若皆為 `None` 則 `continue`；若單邊為 `None` 或值不同則直接回傳 `False`；否則將 `(node1.left, node2.left)` 與 `(node1.right, node2.right)` 依序加入佇列。
        
- **時間複雜度**：O(min(N, M))
    
- **空間複雜度**：O(min(W1, W2))，取決於樹的最大寬度，滿二元樹時為 O(N)。
    

### **2. 序列化比對法（前序遍歷 + 空節點標記）**

- **思路**：
    
    - 將兩棵樹分別進行包含空節點標記（例如 `#`）的前序遍歷，轉成字串或陣列後直接比較 `serialize(p) == serialize(q)`。（注意：必須保留空節點標記，否則 `[1, 2]` 與 `[1, None, 2]` 的前序結果都會是 `[1, 2]`）。
        
- **時間複雜度**：O(N + M)（即使開頭就不同也必須完整遍歷兩棵樹）
    
- **空間複雜度**：O(N + M)
    

## **總結**

- **面試首選解法**：**DFS 遞迴解法**，僅需 6 行核心代碼，邏輯極度清晰，時間與空間效率皆為最優，且無需額外引入資料結構。
    
- **避免堆疊溢位解法**：若題目測試資料的樹極深（例如超過 1000 層的斜樹可能觸發 Python `RecursionError`），應改用 **BFS 佇列迭代解法** 或手動維護 Stack 的迭代 DFS。
    

## **二元樹結構比較 / 遞迴驗證類問題整理**

|題號|題目名稱|問題類型|解法|
|---|---|---|---|
|**LeetCode 100**|Same Tree（相同的樹）|**判斷兩棵樹結構與節點值是否完全相同**|**同步 DFS**，同時比較 `(p.left, q.left)` 與 `(p.right, q.right)`|
|**LeetCode 101**|Symmetric Tree（對稱二元樹）|**判斷一棵樹是否為左右鏡像對稱**|**鏡像同步 DFS**，比較 `(p.left, q.right)` 與 `(p.right, q.left)`|
|**LeetCode 572**|Subtree of Another Tree（另一棵樹的子樹）|**判斷 `subRoot` 是否為 `root` 的子樹**|**雙重 DFS**：外層遍歷 `root` 每個節點，內層直接呼叫 `isSameTree(node, subRoot)`|
|**LeetCode 226**|Invert Binary Tree（翻轉二元樹）|**將二元樹所有左右子樹鏡像反轉**|**DFS 後序 / 前序遍歷**，交換 `root.left, root.right`|
|**LeetCode 951**|Flip Equivalent Binary Trees（翻轉等價二元樹）|**允許翻轉任意節點的左右子樹，判斷兩樹是否等價**|**DFS 分支討論**：`(不翻轉左左 & 右右) or (翻轉左右 & 右左)`|
|**LeetCode 110**|Balanced Binary Tree（平衡二元樹）|**判斷每個節點的左右子樹高度差是否不超過 1**|**自底向上 DFS**，回傳樹高，若不平衡則提早回傳 `-1`|
|**LeetCode 617**|Merge Two Binary Trees（合併二元樹）|**將兩棵樹重疊位置的節點值相加合併**|**同步 DFS**，若一方為空直接回傳另一方，皆非空則建立 `TreeNode(p.val + q.val)`|

### **延伸變形題：對稱二元樹（Symmetric Tree - LeetCode 101）**

給定一個二元樹的根節點 `root`，檢查它是否軸對稱（即這棵樹的左子樹與右子樹互為鏡像）。

Python

```
from typing import Optional

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        if not root:
            return True
            
        def isMirror(left: Optional[TreeNode], right: Optional[TreeNode]) -> bool:
            if not left and not right:
                return True
            if not left or not right or left.val != right.val:
                return False
            # 與 Same Tree 的唯一差別：外側比外側 (left.left vs right.right)，內側比內側 (left.right vs right.left)
            return isMirror(left.left, right.right) and isMirror(left.right, right.left)
            
        return isMirror(root.left, root.right)
```