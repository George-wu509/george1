
二元樹的右視圖 Binary Tree Right Side View

LeetCode 199 / LintCode 760 給定一棵二元樹的根節點 `root`，想像自己站在它的右側，按照從頂部到底部的順序，返回從右側所能看到的節點值。

**样例 1：** 输入：
```Python
"""
root = [1, 2, 3, None, 5, None, 4]
    1            <---
  /   \
 2     3         <---
  \     \
   5     4       <---
"""
```

输出：
```Python
[1, 3, 4]
```

解释： 每一層最右邊的節點分別是 1、3、4。

**样例 2：** 输入：
```Python
"""
root = [1, 2, 3, 4, None, None, None, 5]
      1          <---
    /   \
   2     3       <---
  /
 4               <---
/
5                <---
"""
```

输出：
```Python
[1, 3, 4, 5]
```

解释： 雖然第 3、4 層沒有右子樹的節點，但從右側看過去，左子樹最右邊的節點 4 和 5 不會被遮擋，因此也能被看到。

**样例 3：** 输入：
```Python
"""
root = []
"""
```

输出：
```Python
[]
```

解释： 空樹看不到任何節點。

```Python
from collections import deque
from typing import List, Optional

class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                # 當前層的最後一個節點，即為最右側節點
                if i == level_size - 1:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
        return result
```


```Python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        result = []
        
        def dfs(node: Optional[TreeNode], depth: int) -> None:
            if not node:
                return
            # 每一層第一次訪問到的節點，一定是最右邊的節點
            if depth == len(result):
                result.append(node.val)
            # 先走右子樹，再走左子樹
            dfs(node.right, depth + 1)
            dfs(node.left, depth + 1)
            
        dfs(root, 0)
        return result
```

解釋 (BFS 解法): step1: 利用 Queue 進行層序遍歷 (Level Order Traversal)，每次處理一整層的節點 step2: 若 `root` 為空，直接回傳 `[]` step3: 假設樹為 `[1, 2, 3, None, 5, None, 4]`，初始 `queue = [1]`, `result = []` step4: 第 0 層 (`level_size = 1`): `i = 0` 彈出 `1`，因為 `i == level_size - 1` (0 == 0)，`result = [1]`，加入子節點 -> `queue = [2, 3]` step5: 第 1 層 (`level_size = 2`): `i = 0` 彈出 `2` (加入 `5`)；`i = 1` 彈出 `3` (加入 `4`)，因為 `1 == 1`，`result = [1, 3]` -> `queue = [5, 4]` step6: 第 2 層 (`level_size = 2`): `i = 0` 彈出 `5`；`i = 1` 彈出 `4`，因為 `1 == 1`，`result = [1, 3, 4]` -> `queue = []` step7: `queue` 為空結束迴圈，所以答案 = `[1, 3, 4]`

解釋 (核心觀念): 右視圖並不是「一直往 `root.right` 走到底」，而是**每一層的最右側節點**。 BFS 每次取該層最後一個元素 (`i == level_size - 1`)；DFS 則採用 `根 -> 右 -> 左` 順序，每到達一個新深度 (`depth == len(result)`) 就記錄該節點。

### **LeetCode 199 - Binary Tree Right Side View（二元樹的右視圖）**

這道題目是 Blind 75 與 NeetCode 150 的經典樹類題目，核心考點是 **廣度優先搜尋（BFS 層序遍歷）** 或 **深度優先搜尋（DFS 反向前序遍歷）**。問題要求找出二元樹「每一層最右邊的節點」，容易掉入「只沿著 `right` 指針往下找」的陷阱。

## **解法解析**

這道題可以轉化為 **提取二元樹每一層的最後一個節點** 來求解。

### **思路**

1. **觀察規律（避開常見陷阱）**
    
    - 許多人第一直覺會寫 `while root: res.append(root.val); root = root.right`，這是**錯誤**的。
        
    - 當右子樹較短、左子樹較深時（如範例 2），左子樹深處的節點會從右側露出來。
        
    - 因此，右視圖的本質是：**二元樹中每一層（Level）最右側的那個節點**。
        
2. **核心邏輯（BFS 層序遍歷）**
    
    - 使用佇列 `queue` 儲存每一層的節點，從第 `0` 層（根節點）開始逐層向下掃描。
        
    - 在進入每一層的迴圈前，先用 `level_size = len(queue)` 鎖定當前層的節點總數。
        
    - 依序從左到右彈出該層節點，當索引 `i == level_size - 1` 時，代表這就是該層最右邊的節點，將其加入 `result`。
        
3. **代碼執行流程**
    
    - 若 `root` 為 `None`，直接回傳空陣列 `[]`。
        
    - 初始化 `queue = deque([root])` 與 `result = []`。
        
    - **外層 `while queue` 控制層數**，**內層 `for i in range(level_size)` 遍歷該層所有節點**。
        
    - 遍歷過程中將左右子節點依序放入 `queue`（供下一層使用），並在該層最後一步記錄節點值。
        
    - 最後返回 `result`。
        

## **具體舉例**

假設輸入為範例 2 的不對稱樹 `root = [1, 2, 3, 4, None, None, None, 5]`，BFS 的變數變化如下：

|**層數 (Depth)**|**level_size**|**彈出節點順序**|**觸發記錄 (i == level_size - 1)**|**result 狀態**|**下一層 queue**|
|---|---|---|---|---|---|
|第 0 層|`1`|`1`|`i = 0` → 記錄 `1`|`[1]`|`[2, 3]`|
|第 1 層|`2`|`2` → `3`|`i = 1` → 記錄 `3`|`[1, 3]`|`[4]`|
|第 2 層|`1`|`4`|`i = 0` → 記錄 `4`|`[1, 3, 4]`|`[5]`|
|第 3 層|`1`|`5`|`i = 0` → 記錄 `5`|`[1, 3, 4, 5]`|`[]`|

最後返回 `[1, 3, 4, 5]`，完整捕捉到了左側深處露出的節點 `4` 與 `5`。

## **時間與空間複雜度分析**

- **時間複雜度：O(N)**
    
    - 其中 `N` 為二元樹的節點總數。無論是 BFS 還是 DFS，每個節點都只會被訪問一次，進出佇列或遞迴呼叫皆為 `O(1)`，因此總時間複雜度為 **O(N)**。
        
- **空間複雜度：O(N)**（或寫作 **O(W)** / **O(H)**）
    
    - **BFS 解法**：空間取決於二元樹的最大寬度 `W`。在最壞情況（滿二元樹）下，最後一層有 `N/2` 個節點，因此佇列空間複雜度為 **O(N)**。
        
    - **DFS 解法**：空間取決於遞迴呼叫棧的深度（樹高 `H`）。平衡二元樹為 **O(log N)**，最壞情況（退化成鏈表的斜樹）為 **O(N)**。
        

## **其他解法**

### **1. DFS 反向前序遍歷（Root -> Right -> Left）**

- **思路**：
    
    - 改變傳統前序遍歷（根左右）的順序，改為 **根 → 右 → 左**。
        
    - 傳遞當前深度 `depth`（根節點為 `0`）。因為每一層都是先往右走，所以**當 `depth == len(result)` 時，代表這是第一次到達該深度，該節點必然是這一層最右邊的節點**。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(H)，平衡樹下為 O(log N)，比 BFS 更省空間。
    

### **2. BFS 單層列表覆蓋法**

- **思路**：
    
    - 不使用 `deque` 計數，而是每一層用一個臨時陣列 `level` 收集該層所有節點，每層結束後直接取 `level[-1]` 加入 `result`，並更新 `queue = next_level`。
        
- **時間複雜度**：O(N)
    
- **空間複雜度**：O(N)
    

## **總結**

- **面試最推薦解法**：**BFS 層序遍歷（使用 `deque` + `level_size`）**，邏輯最直觀、不易出錯，且能直接套用到所有二元樹層序遍歷變形題。
    
- **空間最優解法（平衡樹情況）**：**DFS 反向前序遍歷（Root → Right → Left）**，代碼極簡，且在樹較寬但較淺時空間開銷遠小於 BFS。
    

## **二元樹層序遍歷 / 視圖類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 199**|Binary Tree Right Side View（二元樹的右視圖）|**求每一層最右側的節點**|**BFS 取每層最後一個節點**，或 **DFS (`根->右->左`) 記錄首次到達深度的節點**|
|**LeetCode 102**|Binary Tree Level Order Traversal（二元樹的層序遍歷）|**逐層由左至右輸出所有節點**|**BFS 基礎模板**，用 `level_size` 分層收集節點值|
|**LeetCode 107**|Binary Tree Level Order Traversal II（層序遍歷 II）|**自底向上逐層輸出節點**|**BFS 層序遍歷**，最後將結果反轉 `result[::-1]`|
|**LeetCode 103**|Binary Tree Zigzag Level Order Traversal（鋸齒形層序遍歷）|**奇數層由左至右、偶數層由右至左**|**BFS 層序遍歷**，搭配布林旗標或奇偶層反轉該層陣列|
|**LeetCode 637**|Average of Levels in Binary Tree（二元樹的層平均值）|**計算每一層節點值的平均值**|**BFS 層序遍歷**，累加每層總和除以 `level_size`|
|**LeetCode 515**|Find Largest Value in Each Tree Row（在每個樹行中找最大值）|**找出每一層的最大值**|**BFS 層序遍歷**（每層維護 `max_val`）或 **DFS 傳遞深度更新最大值**|
|**LeetCode 116 / 117**|Populating Next Right Pointers in Each Node（填充每個節點的下一個右側節點指針）|**將同層節點用 `next` 指針串聯**|**BFS 層序遍歷**（同層 `curr.next = queue[0]`），或利用已建立的 `next` 指針達成 `O(1)` 空間|
|**LeetCode 987**|Vertical Order Traversal of a Binary Tree（二元樹的垂序遍歷）|**按垂直座標 `(row, col)` 從左到右、從上到下分組**|**BFS / DFS 記錄 `(col, row, val)`** + 哈希表分組與排序|
|**LeetCode 545**|Boundary of Binary Tree（二元樹的邊界）|**求逆時針一圈的邊界節點（左邊界 + 葉子 + 右邊界）**|**分段 DFS**：分別求左邊界（由上到下）、所有葉節點（由左到右）、右邊界（由下到上）|

### **延伸變形題：二元樹的左視圖（Binary Tree Left Side View）**

給定一棵二元樹的根節點 `root`，想像自己站在它的左側，按照從頂部到底部的順序，返回從左側所能看到的節點值。

Python

```
from collections import deque
from typing import List, Optional

class Solution:
    def leftSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        result = []
        queue = deque([root])
        
        while queue:
            level_size = len(queue)
            for i in range(level_size):
                node = queue.popleft()
                # 與右視圖唯一差別：取每一層的第一個節點 (i == 0)
                if i == 0:
                    result.append(node.val)
                if node.left:
                    queue.append(node.left)
                if node.right:
                    queue.append(node.right)
                    
        return result
```