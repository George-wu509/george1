
二元搜尋樹的最近共同祖先 Lowest Common Ancestor of a Binary Search Tree

LeetCode 235 (LintCode 1311) 給定一個**二元搜尋樹（Binary Search Tree, BST）**，找到該樹中兩個指定節點 `p` 和 `q` 的**最近共同祖先（Lowest Common Ancestor, LCA）**。

百度百科中最近共同祖先的定義為：「對於有根樹 T 的兩個節點 p、q，最近共同祖先表示為一個節點 x，滿足 x 是 p、q 的祖先且 x 的深度盡可能大（**一個節點也可以是它自己的祖先**）。」

**样例 1：** 输入：
```Python
root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 8
```

输出：
```Python
6
```

解释： 節點 2 和節點 8 的最近共同祖先是 6（2 在 6 的左子樹，8 在 6 的右子樹，6 是兩者的分岔點）。

**样例 2：** 输入：
```Python
root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4
```

输出：
```Python
2
```

解释： 節點 2 和節點 4 的最近共同祖先是 2，因為根據定義，一個節點也可以是它自己的祖先（4 在 2 的右子樹中）。

**样例 3：** 输入：
```Python
root = [2,1], p = 2, q = 1
```

输出：
```Python
2
```

解释： 節點 1 在節點 2 的左子樹，且節點 2 是自身的祖先，故回傳 2。


```Python
# 方法一：迭代法 (Iterative) - 空間最佳解 O(1)
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        curr = root
        while curr:
            if p.val > curr.val and q.val > curr.val:
                curr = curr.right
            elif p.val < curr.val and q.val < curr.val:
                curr = curr.left
            else:
                return curr
```


```Python
# 方法二：遞迴法 (Recursive) - 空間 O(H)
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```

解釋: step1: 利用 BST 的核心性質：**左子樹所有節點 < 根節點 < 右子樹所有節點**。 step2: 從根節點 `curr = root` 開始往下找，比較 `p.val`、`q.val` 與 `curr.val` 的大小關係。 step3: 若 `p.val` 和 `q.val` **都大於** `curr.val`，代表 `p` 和 `q` 都在右子樹，所以往右走 (`curr = curr.right`)。 step4: 若 `p.val` 和 `q.val` **都小於** `curr.val`，代表 `p` 和 `q` 都在左子樹，所以往左走 (`curr = curr.left`)。 step5: 若不符合上述兩種情況（即 `p` 和 `q` 分別在左右兩側，或者其中一個剛好等於 `curr`），代表當前節點 `curr` 就是分岔點，直接回傳 `curr`。

`root = [6,2,8,0,4,7,9,null,null,3,5], p = 2, q = 4` -> `result = 2`

1. `curr = 6`：2 和 4 都小於 6 -> 往左子樹移動 (`curr = 2`)
    
2. `curr = 2`：`p.val == 2` 等於 `curr.val`，無法再同時大於或小於 -> 找到分岔點，回傳 `2`
    

解釋: 不需要像一般二元樹那樣遍歷整棵樹找路徑。

只要 `p` 和 `q` 在同一側，就繼續往那一側深入；**第一個讓 `p` 和 `q` 分開（或剛好撞上其中一個）的節點，就是最近共同祖先 (LCA)**。

### **LeetCode 235 - Lowest Common Ancestor of a Binary Search Tree（二元搜尋樹的最近共同祖先）**

這道題目是 **二元搜尋樹（BST）** 性質應用的經典代表題（Blind 75 & NeetCode 150 必考題）。問題要求在已知 BST 結構的前提下，找出兩個目標節點 `p` 與 `q` 的最近共同祖先。

## **解法解析**

這道題可以善用 **二元搜尋樹的有序性** 來進行「剪枝搜尋」，完全不需要走訪另一半邊的子樹。

### **思路**

1. **觀察規律（BST 性質）**
    
    - 對於任意節點 `curr`：
        
        - 左子樹的所有值 `< curr.val`
            
        - 右子樹的所有值 `> curr.val`
            
    - 題目保證所有節點值都是**唯一**的，且 `p` 和 `q` 一定存在於樹中。
        
2. **判斷分岔條件（三種情況）**
    
    - **情況 A（同在右側）**：`p.val > curr.val` 且 `q.val > curr.val` → 兩者都在右子樹，`curr` 還不夠深，往 `curr.right` 移動。
        
    - **情況 B（同在左側）**：`p.val < curr.val` 且 `q.val < curr.val` → 兩者都在左子樹，`curr` 還不夠深，往 `curr.left` 移動。
        
    - **情況 C（出現分岔或命中）**：`p` 和 `q` 一左一右，或 `curr` 本身就是 `p` 或 `q` → 若再往下走，一定會丟失其中一個節點，因此 `curr` 就是**最近共同祖先**。
        
3. **代碼執行流程**
    
    - 使用指標 `curr = root` 從根節點出發。
        
    - 進入 `while curr:` 迴圈，檢查 `p` 和 `q` 是否同側。
        
    - 若同側則更新 `curr` 往左或往右；一旦不同側（觸發 `else`），立刻 `return curr`。
        

## **具體舉例**

假設 `root = [6, 2, 8, 0, 4, 7, 9, null, null, 3, 5]`，我們要找 `p = 3` 和 `q = 5` 的最近共同祖先，變數變化如下：

|**步驟**|**curr.val**|**p=3, q=5 與 curr 的比較**|**決策與動作**|
|---|---|---|---|
|初始化|`6`|`3 < 6` 且 `5 < 6`（同在左側）|往左子樹走 (`curr = curr.left`)|
|第 1 步|`2`|`3 > 2` 且 `5 > 2`（同在右側）|往右子樹走 (`curr = curr.right`)|
|第 2 步|`4`|`3 < 4` 但 `5 > 4`（一左一右，分岔！）|觸發 `else`，回傳節點 `4`|

最後返回節點 `4`，表示 `3` 和 `5` 的最近共同祖先為 `4`。

## **時間與空間複雜度分析**

- **時間複雜度：O(H)**（其中 `H` 為樹的高度）
    
    - 每一輪迴圈都會往下一層移動，最多走訪從根節點到葉節點的一條路徑：
        
        - **平均情況（平衡 BST）**：樹高 $H = \log N$，時間複雜度為 **O(log N)**。
            
        - **最壞情況（斜曲樹 Skewed Tree）**：樹退化成鏈結串列，樹高 $H = N$，時間複雜度為 **O(N)**。
            
- **空間複雜度：O(1)**（迭代法）
    
    - 只使用了一個指標 `curr` 進行追蹤，沒有使用額外的遞迴堆疊或陣列，空間複雜度為 **O(1)**。
        
    - _(註：若使用遞迴法，因為需要系統呼叫堆疊，空間複雜度為 **O(H)**。)_
        

## **其他解法**

### **1. 遞迴法（Recursive BST Search）**

- **思路**：
    
    - 邏輯與迭代法完全相同，只是透過函式呼叫自己 (`lowestCommonAncestor(root.left/right, p, q)`) 往下層推進。
        
- **時間複雜度**：O(H)
    
- **空間複雜度**：O(H)（遞迴呼叫堆疊深度）
    

### **2. 紀錄路徑法（Path Comparison）**

- **思路**：
    
    - 分別從 `root` 搜尋 `p` 和 `q`，並把經過的節點存成兩個陣列 `path_p` 和 `path_q`，最後比對兩個陣列中**最後一個相同的節點**。
        
- **時間複雜度**：O(H)
    
- **空間複雜度**：O(H)（需額外儲存兩條路徑）
    

### **3. 一般二元樹的後序遍歷（DFS Post-order）**

- **思路**：
    
    - 不利用 BST 大小性質，直接當作普通二元樹（同 LeetCode 236），左右子樹分別遞迴尋找 `p` 和 `q`。雖然也能通過，但會遍歷整棵樹，沒有發揮 BST 的優勢。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(H)
    

## **總結**

- **最佳解法（面試首選）**：**迭代法（Iterative）**，利用 BST 大小比較一路向下找分岔點，時間 `O(H)`，空間僅需 `O(1)`。
    
- **核心口訣**：**「同大往右、同小往左、分岔即答案」**。
    
- **常見陷阱**：不要忘記題目允許「節點本身可以是自己的祖先」，因此當 `curr.val == p.val` 或 `curr.val == q.val` 時，`else` 分支會直接將 `curr` 回傳，這在邏輯上是完全正確的。
    

## **樹與 LCA 類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 235**|Lowest Common Ancestor of a BST|**二元搜尋樹（BST）**，求兩節點最近共同祖先|**利用 BST 大小性質迭代**，同側移動、分岔回傳，時間 `O(H)` 空間 `O(1)`|
|**LeetCode 236**|Lowest Common Ancestor of a Binary Tree|**一般二元樹（無大小順序）**，求兩節點最近共同祖先|**後序遍歷（Post-order DFS）**，由下往上回報是否找到 `p` 或 `q`，時間 `O(N)`|
|**LeetCode 1644**|Lowest Common Ancestor of a Binary Tree II|一般二元樹，但 **`p` 或 `q` 可能不存在於樹中**|**後序遍歷 + 計數器**，必須走完整棵樹確認兩個節點都存在才回傳 LCA|
|**LeetCode 1650**|Lowest Common Ancestor of a Binary Tree III|每個節點帶有 **`parent` 指標**（無根節點 `root`）|**雙指標（同 Linked List 相交問題 LeetCode 160）**，走完換對方起點，空間 `O(1)`|
|**LeetCode 1676**|Lowest Common Ancestor of a Binary Tree IV|一般二元樹，求 **多個節點陣列 `nodes`** 的最近共同祖先|**HashSet + 後序遍歷**，只要當前節點在 `nodes` 集合中就往上回報|
|**LeetCode 1123**|Lowest Common Ancestor of Deepest Leaves|一般二元樹，求 **所有最深葉節點** 的最近共同祖先|**後序遍歷回傳 `(節點, 深度)`**，左右深度相等時當前節點即為 LCA|
|**LeetCode 98**|Validate Binary Search Tree|驗證一棵樹是否為合法的二元搜尋樹（BST）|**DFS 傳遞上下界 `(min_val, max_val)`** 或中序遍歷檢查是否嚴格遞增|
|**LeetCode 230**|Kth Smallest Element in a BST|找出 BST 中第 `k` 小的元素|**中序遍歷（In-order Traversal）**，BST 中序遍歷即為由小到大排序|

### **進階延伸題：一般二元樹的最近共同祖先（LeetCode 236）**

如果題目把「二元搜尋樹（BST）」改成「一般二元樹（Binary Tree）」，節點沒有左小右大的規律，該怎麼求最近共同祖先？

Python

```
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        # 1. 終止條件：走到底部(None)，或是剛好找到 p 或 q，直接往上回報
        if not root or root == p or root == q:
            return root
        
        # 2. 分治遞迴：分別去左右子樹尋找 p 和 q
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        
        # 3. 後序處理（根據左右子樹的回報結果做決定）：
        # 如果左右子樹都各自找到一個目標，代表 p 和 q 分居兩側，當前 root 就是 LCA！
        if left and right:
            return root
        
        # 如果只有一邊找到（另一邊為 None），就把找到的那一邊繼續往上傳遞
        return left if left else right
```