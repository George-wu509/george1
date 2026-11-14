
另一棵樹的子樹 Subtree of Another Tree

LeetCode 572 / LintCode 1165 給你兩棵二元樹 `root` 和 `subRoot`。檢驗 `root` 中是否包含和 `subRoot` 具有相同結構和節點值的子樹。如果存在，返回 `True`；否則，返回 `False`。 二元樹 `tree` 的一棵子樹包括 `tree` 的某個節點和這個節點的所有後代節點。`tree` 也可以看做它自身的一棵子樹。

**样例 1：** 输入：
```Python
"""
root = [3, 4, 5, 1, 2], subRoot = [4, 1, 2]
     3
    / \
   4   5        4
  / \          / \
 1   2        1   2
"""
```

输出：
```Python
True
```

解释： `root` 中以節點 `4` 為根的子樹，其結構與所有後代節點值都與 `subRoot` 完全相同。

**样例 2：** 输入：
```Python
"""
root = [3, 4, 5, 1, 2, None, None, None, None, 0], subRoot = [4, 1, 2]
     3
    / \
   4   5        4
  / \          / \
 1   2        1   2
    /
   0
"""
```

输出：
```Python
False
```

解释： 雖然 `root` 中包含 `[4, 1, 2]` 的部分結構，但節點 `2` 底下還有一個子節點 `0`，而子樹必須包含「該節點的所有後代節點」，因此不匹配。

**样例 3：** 输入：
```Python
"""
root = [1, 1], subRoot = [1]
   1
  /        1
 1
"""
```

输出：
```Python
True
```

解释： 樹中允許出現重複的節點值。根節點 `1` 雖然值相同但結構不符，繼續往下找左子節點 `1` 時即可完全匹配。

```Python
from typing import Optional

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # 邊界條件：空樹是任何樹的子樹
        if not subRoot:
            return True
        # subRoot 非空，但 root 已經遍歷到底為空，不可能匹配
        if not root:
            return False

        # 1. 檢查以當前 root 為起點是否與 subRoot 完全相同
        if self.isSameTree(root, subRoot):
            return True

        # 2. 若當前不相同，則繼續往左子樹或右子樹尋找是否有匹配的起點
        return self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)

    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        if not p and not q:
            return True
        if not p or not q or p.val != q.val:
            return False
        return self.isSameTree(p.left, q.left) and self.isSameTree(p.right, q.right)
```


```Python
from typing import Optional

class Solution:
    def isSubtree(self, root: Optional[TreeNode], subRoot: Optional[TreeNode]) -> bool:
        # 序列化輔助函數：前序遍歷 + 邊界分隔符(^) + 空節點標記(#)
        def serialize(node: Optional[TreeNode]) -> str:
            if not node:
                return "#,"
            # 加上 '^' 避免出現 "12" 包含 "2" 的誤判情況
            return f"^{node.val}," + serialize(node.left) + serialize(node.right)

        root_str = serialize(root)
        sub_str = serialize(subRoot)

        # 將樹的比對問題轉化為字串子序列匹配問題
        return sub_str in root_str
```

解釋 (雙重 DFS 解法): step1: 這個問題可以拆解成兩個遞迴函數：外層 `isSubtree` 負責「遍歷 `root` 的每一個節點作為起點」，內層 `isSameTree` 負責「驗證兩棵樹是否完全相同」 step2: 處理邊界條件：若 `subRoot` 為空回傳 `True`；若 `root` 為空回傳 `False` step3: 假設 `root = [3, 4, 5, 1, 2]`, `subRoot = [4, 1, 2]`，先從根節點 `3` 開始呼叫 `isSameTree(3, 4)` -> `3 != 4` 回傳 `False` step4: 因為根節點不匹配，執行 `isSubtree(root.left, subRoot) or isSubtree(root.right, subRoot)`，先往左子樹 `Node(4)` 移動 step5: 到達 `Node(4)`，呼叫 `isSameTree(4, 4)` -> 根節點 `4 == 4`，左子樹 `1 == 1`，右子樹 `2 == 2`，且葉節點下方皆為 `None` -> 回傳 `True` step6: 外層透過 `or` 短路運算接收到 `True`，無需再檢查右子樹 `Node(5)`，直接回傳 `True`

解釋 (核心觀念):**`Subtree of Another Tree` = `遍歷樹的每個節點` + `LeetCode 100 (Same Tree)`**。 需要注意兩大易錯點：

1. **子樹必須包含所有後代**：不能只匹配到一半就停下，必須一路比對到葉節點的 `None`（由 `isSameTree` 保證）。
    
2. **節點值可能重複**：當 `root.val == subRoot.val` 但結構不符時（如範例 3），**不能直接回傳 `False`**，必須繼續遞迴檢查 `root.left` 與 `root.right`。
    

### **LeetCode 572 - Subtree of Another Tree（另一棵樹的子樹）**

這道題目是 Blind 75 與 NeetCode 150 中「樹結構比對」的代表性題目，直接建立在 **LeetCode 100（Same Tree）** 的基礎之上。它同時考驗了雙重遞迴（Nested DFS）**的結構拆解能力，以及將**樹結構轉化為字串匹配（Serialization + KMP）的降維思考技巧。

## **解法解析**

這道題可以轉化為 **在 `root` 中尋找某個節點 `u`，使得以 `u` 為根的子樹與 `subRoot` 為「相同的樹」**。

### **思路**

1. **拆分職責（雙層遞迴設計）**
    
    - **外層函數 `isSubtree(root, subRoot)`**：扮演「搜尋者」的角色，遍歷 `root` 中的每一個節點，詢問：「如果把當前節點當作起點，能不能跟 `subRoot` 吻合？」
        
    - **內層函數 `isSameTree(p, q)`**：扮演「驗證者」的角色，嚴格比對兩棵樹從根到葉子的每一個節點值與空指標位置是否完全一致。
        
2. **避開常見陷阱（重複值與不完整子樹）**
    
    - **陷阱一（提前回傳 False）**：許多初學者會寫成 `if root.val == subRoot.val: return self.isSameTree(root, subRoot)`。這在遇到重複值（如 `root = [1, 1]`, `subRoot = [1]`）時會出錯——當外層的 `1` 驗證失敗時，應該繼續往內層的 `1` 尋找，而不是直接宣告失敗。
        
    - **陷阱二（忽略子節點之後的結構）**：如範例 2，若 `subRoot` 已經走到 `None` 而 `root` 還有子節點 `0`，`isSameTree` 中的 `if not p or not q: return False` 會精準攔截這種「只匹配一半」的情況。
        
3. **代碼執行流程**
    
    - 在 `isSubtree` 中，先判斷 `if not subRoot: return True` 與 `if not root: return False`。
        
    - 接著呼叫 `if self.isSameTree(root, subRoot): return True`。
        
    - 若當前節點不吻合，則回傳 `self.isSubtree(root.left, subRoot) or self.isSubtree(root.right, subRoot)`，只要左右任一側找到即算成功。
        

## **具體舉例**

假設輸入為範例 2：`root = [3, 4, 5, 1, 2, None, None, None, None, 0]`，`subRoot = [4, 1, 2]`，雙重 DFS 的執行追蹤如下：

|外層 `isSubtree` 節點|觸發內層 `isSameTree` 比對|比對失敗原因 / 結果|外層下一步動作|
|---|---|---|---|
|`Node(3)`|`isSameTree(3, 4)`|`3 != 4` → `False`|繼續搜尋左子樹 `Node(4)`|
|`Node(4)`|`isSameTree(4, 4)` → 左 `1==1`、右 `2==2` → 檢查 `2.left` (`Node(0)` vs `None`)|單邊為空 → `False`|左子樹全數失敗，改搜右子樹 `Node(5)`|
|`Node(1)`（4 的左子）|`isSameTree(1, 4)`|`1 != 4` → `False`|繼續搜其左右子節點 (`None`) → `False`|
|`Node(2)`（4 的右子）|`isSameTree(2, 4)`|`2 != 4` → `False`|繼續搜 `Node(0)` → `False`|
|`Node(5)`|`isSameTree(5, 4)`|`5 != 4` → `False`|左右皆為 `None` → 全部結束|

最終所有分支皆回傳 `False`，總結果為 `False`。

## **時間與空間複雜度分析**

- **時間複雜度：O(N × M)**
    
    - 其中 `N` 為 `root` 的節點數，`M` 為 `subRoot` 的節點數。在最壞情況下（例如 `root` 和 `subRoot` 所有節點的值都相同，但葉子不匹配），對於 `root` 的每一個節點（共 `N` 個），都可能需要完整執行一次 `isSameTree`（耗時 `O(M)`），因此最壞時間複雜度為 **O(N × M)**。
        
- **空間複雜度：O(H_root)**
    
    - 空間開銷取決於遞迴呼叫棧的最大深度，即 `root` 的高度 `H_root`（因為 `isSameTree` 的遞迴深度不會超過 `root` 剩餘的高度）。
        
    - 在平衡二元樹情況下為 **O(log N)**；在最壞情況（退化成單邊斜樹）下為 **O(N)**。
        

## **其他解法**

### **1. 前序序列化 + 字串匹配（KMP / Rolling Hash）**

- **思路**：
    
    - 將 `root` 與 `subRoot` 分別透過「包含空節點標記 `#`」的前序遍歷轉成唯一對應的字串。
        
    - **關鍵細節**：每個節點值前面必須加上前綴分隔符（例如 `^` 或 `,`），否則 `root = [12]` 序列化成 `"12,#,#"` 會誤判包含 `subRoot = [2]` 的 `"2,#,#"`；加上前綴後 `"^12,#,#"` 就不會包含 `"^2,#,#"`。
        
    - 接著利用 **KMP 演算法**（或 Python 內建的 Boyer-Moore/Two-Way `in` 運算子）在線性時間內判斷 `sub_str` 是否為 `root_str` 的子字串。
        
- **時間複雜度**：O(N + M)（搭配 KMP 演算法可達到理論最優線性時間）
    
- **空間複雜度**：O(N + M)（需要額外儲存兩棵樹的序列化字串）
    

### **2. 樹雜湊（Tree Hashing / Merkle Tree Hash）**

- **思路**：
    
    - 透過後序遍歷（Bottom-Up），為每棵子樹計算一個唯一的雜湊值：`hash(node) = hash(str(node.val) + "#" + hash(node.left) + "#" + hash(node.right))`（可搭配 SHA-256 或雙重多項式雜湊避免碰撞）。
        
    - 將 `root` 中所有子樹的雜湊值存入一個 `HashSet`，最後只需 O(1)O(1)O(1) 檢查 `hash(subRoot)` 是否存在於集合中即可。
        
- **時間複雜度**：O(N + M)
    
- **空間複雜度**：O(N + M)
    

## **總結**

- **面試首選解法**：**雙重 DFS 遞迴解法（`isSubtree` + `isSameTree`）**，代碼簡潔清晰、無需額外 O(N)O(N)O(N) 記憶體空間，且在實際測試資料中因早期剪枝（`val` 不相等立刻返回），平均表現非常快。
    
- **進階追問（Follow-up）應對**：若面試官追問「如何將時間複雜度從 O(N×M)O(N \times M)O(N×M) 優化到 O(N+M)O(N + M)O(N+M)？」，即可提出 **前序序列化 + KMP 字串匹配** 或 **Merkle 樹雜湊法**，並務必點出「空節點標記 `#`」與「節點值前綴分隔符 `^`」兩個防止結構誤判的關鍵細節。
    

## **二元樹子樹匹配 / 樹比對類問題整理**

|題號|題目名稱|問題類型|解法|
|---|---|---|---|
|**LeetCode 572**|Subtree of Another Tree（另一棵樹的子樹）|**判斷 `subRoot` 是否為 `root` 的完整子樹**|**雙重 DFS**（外層遍歷 + 內層 `isSameTree`）或 **前序序列化 + KMP**|
|**LeetCode 100**|Same Tree（相同的樹）|**判斷兩棵樹結構與數值是否完全一致**|**同步 DFS**，同時比對左右子節點與空指標|
|**LeetCode 1367**|Linked List in Binary Tree（二元樹中的列表）|**判斷鏈結串列是否為二元樹中一條向下的路徑**|**雙重 DFS**（注意：鏈結串列只需匹配一條向下路徑，不需匹配所有後代）|
|**LeetCode 652**|Find Duplicate Subtrees（尋找重複的子樹）|**找出二元樹中所有結構與值相同的重複子樹**|**後序 DFS 序列化 + HashMap 計數**（首次計數達到 `2` 時收集結果）|
|**LeetCode 1948**|Delete Duplicate Folders in System（刪除系統中的重複資料夾）|**多叉樹（Trie）中識別並刪除結構相同的子樹**|**字典樹建構 + 後序子樹序列化雜湊**|
|**LeetCode 951**|Flip Equivalent Binary Trees（翻轉等價二元樹）|**允許翻轉任意左右子樹，判斷兩樹是否等價**|**DFS 分支遞迴**：檢查 `(不翻轉) or (翻轉)`|
|**LeetCode 872**|Leaf-Similar Trees（葉子相似的樹）|**判斷兩棵樹從左到右的葉節點序列是否相同**|**DFS 收集葉節點陣列** 後直接比對|

### **延伸變形題：二元樹中的列表（Linked List in Binary Tree - LeetCode 1367）**

給你一棵以 `root` 為根的二元樹和一個 `head` 為第一個節點的鏈結串列。如果在二元樹中，存在一條一直向下的路徑，且每個點的數值恰好一一對應以 `head` 為首的鏈結串列中每個節點的值，那麼請你返回 `True`，否則返回 `False`。

Python

```
from typing import Optional

class Solution:
    def isSubPath(self, head: Optional[ListNode], root: Optional[TreeNode]) -> bool:
        if not head:
            return True
        if not root:
            return False

        # 外層結構與 Subtree of Another Tree 完全相同：嘗試從當前 root 開始，或往左右子樹尋找
        return (
            self.dfs(head, root)
            or self.isSubPath(head, root.left)
            or self.isSubPath(head, root.right)
        )

    def dfs(self, head: Optional[ListNode], node: Optional[TreeNode]) -> bool:
        # 鏈結串列已經全部匹配完畢，無需管二元樹下方是否還有後代（與 LeetCode 572 的最大差異）
        if not head:
            return True
        # 鏈結串列還沒走完，但二元樹已經到底，或節點值不相等
        if not node or head.val != node.val:
            return False

        # 只要往左子樹「或」往右子樹能接續匹配下去即可
        return self.dfs(head.next, node.left) or self.dfs(head.next, node.right)
```