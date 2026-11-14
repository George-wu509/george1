
統計二元樹中的好節點 Count Good Nodes in Binary Tree

LeetCode 1448 給你一棵根為 `root` 的二元樹，請你返回二元樹中**好節點**的數目。 「好節點」X 定義為：從根節點到該節點 X 所經過的路徑中，沒有任何節點的值**大於** X 的值（即 `X.val >= 路徑上的最大值`）。

**样例 1：** 输入：
```Python
"""
root = [3, 1, 4, 3, None, 1, 5]
      3          <--- 好節點 (根節點，路徑最大為 3)
     / \
    1   4        <--- 4 是好節點 (路徑 3->4)；1 不是 (3 > 1)
   /   / \
  3   1   5      <--- 3 是好節點 (路徑 3->1->3)；5 是好節點 (路徑 3->4->5)；1 不是 (路徑 3->4->1)
"""
```

输出：
```Python
4
```

解释：

- 根節點 `(3)` 永遠是好節點。
    
- 節點 `4` -> `(3, 4)` 是路徑中的最大值。
    
- 節點 `5` -> `(3, 4, 5)` 是路徑中的最大值。
    
- 節點 `3` -> `(3, 1, 3)` 是路徑中的最大值（注意：**相等**也算好節點）。 總共 4 個好節點。
    

**样例 2：** 输入：
```Python
"""
root = [3, 3, None, 4, 2]
      3          <--- 好節點
     /
    3            <--- 好節點 (路徑 3->3，3 >= 3)
   / \
  4   2          <--- 4 是好節點 (路徑 3->3->4)；2 不是 (路徑 3->3->2，3 > 2)
"""
```

输出：
```Python
3
```

解释： 節點 `2` 的祖先路徑上有 `3`，因為 `3 > 2`，所以 `2` 不是好節點。

**样例 3：** 输入：
```Python
"""
root = [1]
"""
```

输出：
```Python
1
```

解释： 根節點本身就是好節點。

```Python
from typing import Optional

class Solution:
    def goodNodes(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], max_so_far: int) -> int:
            if not node:
                return 0
            
            # 判斷當前節點是否為好節點（大於或等於路徑上的最大值）
            good = 1 if node.val >= max_so_far else 0
            
            # 更新傳遞給子節點的路徑最大值
            new_max = max(max_so_far, node.val)
            
            # 累加左子樹與右子樹的好節點數量
            good += dfs(node.left, new_max)
            good += dfs(node.right, new_max)
            
            return good

        # 初始路徑最大值設為 root.val（或 float('-inf')，避免節點值為負數時出錯）
        return dfs(root, root.val)
```


```Python
from collections import deque
from typing import Optional

class Solution:
    def goodNodes(self, root: Optional[TreeNode]) -> int:
        if not root:
            return 0
            
        good_count = 0
        # 佇列中同時儲存 (當前節點, 從根到該節點父輩的路徑最大值)
        queue = deque([(root, root.val)])
        
        while queue:
            node, max_so_far = queue.popleft()
            
            if node.val >= max_so_far:
                good_count += 1
                
            new_max = max(max_so_far, node.val)
            
            if node.left:
                queue.append((node.left, new_max))
            if node.right:
                queue.append((node.right, new_max))
                
        return good_count
```

解釋 (DFS 遞迴解法): step1: 採用「自頂向下（Top-Down）」的前序遍歷，在遞迴參數中額外攜帶一個狀態 `max_so_far`，代表「從根節點走到當前節點之前，路徑上的最大值」 step2: 若 `node` 為空，代表走到底了，回傳 `0` step3: 假設輸入為 `root = [3, 1, 4, 3, None, 1, 5]`，從根節點 `dfs(3, 3)` 開始：`3 >= 3` 成立 (`good = 1`)，更新 `new_max = 3` step4: 往左子樹遞迴 `dfs(1, 3)`：`1 >= 3` 不成立 (`good = 0`)，`new_max` 維持 `3`；接著呼叫左子節點 `dfs(3, 3)`：`3 >= 3` 成立 (`good = 1`)，左半邊累計回傳 `1` step5: 往右子樹遞迴 `dfs(4, 3)`：`4 >= 3` 成立 (`good = 1`)，更新 `new_max = 4` step6: 繼續遞迴 `4` 的左右子節點：`dfs(1, 4)` 回傳 `0`（因 `1 < 4`），`dfs(5, 4)` 回傳 `1`（因 `5 >= 4`），右半邊累計回傳 `1 + 0 + 1 = 2` step7: 根節點彙總 `1 + 1 + 2`，最終答案 = `4`

解釋 (核心觀念):**自頂向下傳遞狀態（Top-Down DFS）**： 要判斷某個節點是不是好節點，並不需要每次都回頭重走整條路徑找最大值；只要在往下遞迴的過程中，**順便把目前為止看過的最大值 `max_so_far` 當作參數往下傳**即可。每條從根到葉的分支各自維護自己的 `max_so_far`，左右子樹之間互不干擾。

### **LeetCode 1448 - Count Good Nodes in Binary Tree（統計二元樹中的好節點）**

這道題目是 NeetCode 150 中最經典的 **自頂向下（Top-Down）深度優先搜尋** 題型。相比於「求樹高」、「判斷平衡樹」等需要子樹先回傳資訊的「自底向上（Bottom-Up）」題目，本題的核心在於**父節點將路徑歷史資訊（累積最大值）向子節點傳遞**。

## **解法解析**

這道題可以轉化為 **在樹的遍歷過程中，為每條從根出發的路徑維護一個動態前綴最大值** 來求解。

### **思路**

1. **狀態定義與傳遞方向**
    
    - 一個節點 `X` 是否為好節點，完全取決於「從 `root` 到 `X` 的唯一路徑上的所有祖先節點」。
        
    - 因此我們必須採用 **前序遍歷（根 → 左 → 右）**，在訪問子節點之前，先用當前節點更新路徑最大值 `new_max = max(max_so_far, node.val)`，再將 `new_max` 傳給左、右子樹。
        
    - 因為 `max_so_far` 是透過函數參數（基本型別 `int` 傳值）傳遞的，當左子樹遞迴結束、回溯去走右子樹時，`max_so_far` 會自動恢復成該分岔點的值，無需手動回溯（Backtracking）復原狀態。
        
2. **避開常見陷阱（負數節點值與相等條件）**
    
    - **陷阱一（初始值設為 `0`）**：題目限制節點值範圍為 [−104,104][-10^4, 10^4][−104,104]（包含負數）。如果將初始 `max_so_far` 設為 `0`，當整棵樹都是負數（例如 `root = [-1, -5, -2]`）時，根節點 `-1 >= 0` 會被誤判為 `False`！務必初始化為 `root.val` 或 `float('-inf')`。
        
    - **陷阱二（嚴格大於 vs 大於等於）**：題目定義路徑上「沒有任何節點的值**大於** X」，代表遇到與祖先相等的值（如 `3 -> 3`）時，**依然算作好節點**，判斷條件必須寫 `node.val >= max_so_far`。
        
3. **代碼執行流程**
    
    - 定義輔助函數 `dfs(node, max_so_far)`，回傳以 `node` 為根的子樹中好節點的總數。
        
    - 遇空節點回傳 `0`。
        
    - 計算當前節點貢獻值 `good = 1 if node.val >= max_so_far else 0`。
        
    - 計算新的路徑最大值 `new_max = max(max_so_far, node.val)`，並遞迴加上 `dfs(node.left, new_max)` 與 `dfs(node.right, new_max)`。
        

## **具體舉例**

假設輸入為範例 1：`root = [3, 1, 4, 3, None, 1, 5]`，DFS 遞迴的狀態變化如下：

|遍歷順序|當前節點 `node.val`|傳入的 `max_so_far`|條件 `node.val >= max_so_far`|是否為好節點|傳給子節點的 `new_max`|
|---|---|---|---|---|---|
|1（根）|`3`|`3`|`3 >= 3`（True）|**是 (+1)**|`3`|
|2（左）|`1`|`3`|`1 >= 3`（False）|否 (+0)|`3`|
|3（左-左）|`3`|`3`|`3 >= 3`（True）|**是 (+1)**|`3`|
|4（右）|`4`|`3`|`4 >= 3`（True）|**是 (+1)**|`4`|
|5（右-左）|`1`|`4`|`1 >= 4`（False）|否 (+0)|`4`|
|6（右-右）|`5`|`4`|`5 >= 4`（True）|**是 (+1)**|`5`|

將所有標記為「是」的節點相加：`1 + 0 + 1 + 1 + 0 + 1 = 4`，最終返回 `4`。

## **時間與空間複雜度分析**

- **時間複雜度：O(N)**
    
    - 其中 `N` 為二元樹的節點總數。每個節點只會被訪問一次，每次比較與更新最大值皆為 O(1)O(1)O(1) 操作，因此總時間複雜度為 **O(N)**。
        
- **空間複雜度：O(H)**
    
    - 其中 `H` 為二元樹的高度，空間開銷來自遞迴呼叫棧（Recursion Stack）。
        
    - 在平衡二元樹的情況下為 **O(log N)**；在最壞情況（樹退化成單邊鏈表的斜樹）下為 **O(N)**。
        

## **其他解法**

### **1. BFS 層序遍歷（Queue 綁定路徑狀態）**

- **思路**：
    
    - 使用佇列 `deque` 進行廣度優先搜尋，但佇列中不能只存節點 `node`，必須將「該節點對應的路徑最大值」打包成元組 `(node, max_so_far)` 一起存入佇列。
        
    - 每次彈出 `(node, max_so_far)` 判斷是否累加計數器，並將 `(node.left, new_max)` 與 `(node.right, new_max)` 推入佇列。適合用來避免極深斜樹造成的遞迴堆疊溢位（Stack Overflow）。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(W)，其中 `W` 為樹的最大寬度，滿二元樹時為 O(N)。
    

### **2. 全局變數 / 外部計數器 DFS**

- **思路**：
    
    - 不透過遞迴函數的回傳值做加法，而是在外層維護一個 `self.count = 0`（或 `nonlocal count`），只要 `node.val >= max_so_far` 就執行 `count += 1`，`dfs` 函數本身不回傳數值（`None`）。邏輯完全等價，寫法更貼近直覺模擬。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(H)，最壞 O(N)。
    

## **總結**

- **面試首選解法**：**自頂向下 DFS（帶回傳值或外部計數器）**，代碼簡潔、無需額外佇列空間，且「將路徑狀態作為遞迴參數往下傳」是解決二元樹路徑問題的標準範式。
    
- **核心設計口訣**：當題目條件依賴「從根到當前節點的路徑歷史（如前綴和、最大值、最小值、路徑字串）」時，直接把該狀態放進 DFS 的參數列裡往下傳。
    

## **二元樹路徑狀態傳遞（Top-Down DFS）類問題整理**

|題號|題目名稱|問題類型|解法|
|---|---|---|---|
|**LeetCode 1448**|Count Good Nodes in Binary Tree（統計二元樹中的好節點）|**統計路徑上大於等於所有祖先的節點數**|**Top-Down DFS**，參數傳遞路徑前綴最大值 `max_so_far`|
|**LeetCode 1026**|Maximum Difference Between Node and Ancestor（節點與其祖先之間的最大差值）|**求任意祖先與後代節點值的最大絕對差**|**Top-Down DFS**，參數同時傳遞路徑的 `cur_min` 與 `cur_max`|
|**LeetCode 112**|Path Sum（路徑總和）|**判斷是否存在根到葉的路徑總和等於 `targetSum`**|**Top-Down DFS**，參數傳遞剩餘目標值 `targetSum - node.val`|
|**LeetCode 113**|Path Sum II（路徑總和 II）|**找出所有根到葉總和等於 `targetSum` 的路徑**|**Top-Down DFS + 回溯（Backtracking）**，維護當前路徑陣列 `path`|
|**LeetCode 437**|Path Sum III（路徑總和 III）|**統計任意向下路徑（不限起終點）總和等於目標的數量**|**Top-Down DFS + 前綴和雜湊表（Prefix Sum HashMap）** O(N)|
|**LeetCode 129**|Sum Root to Leaf Numbers（求根節點到葉節點數字之和）|**將每條根到葉路徑組成的數字相加**|**Top-Down DFS**，參數傳遞累積數字 `cur_num * 10 + node.val`|
|**LeetCode 98**|Validate Binary Search Tree（驗證二元搜尋樹）|**判斷是否符合左小右大的 BST 性質**|**Top-Down DFS**，參數傳遞允許的數值上下界 `(lower, upper)`|

### **延伸變形題：節點與其祖先之間的最大差值（Maximum Difference Between Node and Ancestor - LeetCode 1026）**

給定二元樹的根節點 `root`，找出存在於不同節點 `A` 和 `B` 之間的最大值 `V`，其中 `V = |A.val - B.val|`，且 `A` 是 `B` 的祖先。

Python

```
from typing import Optional

class Solution:
    def maxAncestorDiff(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode], cur_min: int, cur_max: int) -> int:
            # 走到葉節點下方時，這條路徑上的「最大值 - 最小值」就是該路徑能產生的最大差值
            if not node:
                return cur_max - cur_min
            
            # 同時更新路徑上的最小值與最大值
            cur_min = min(cur_min, node.val)
            cur_max = max(cur_max, node.val)
            
            # 分別往左右子樹探索，取較大的差值
            left_diff = dfs(node.left, cur_min, cur_max)
            right_diff = dfs(node.right, cur_min, cur_max)
            
            return max(left_diff, right_diff)

        return dfs(root, root.val, root.val)
```