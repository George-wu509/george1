
### **LintCode 1360：Symmetric Tree**

#### **題目描述**

給定一棵二叉樹，判斷它是否是對稱的（即左右子樹是鏡像的）。

例如：
```python
輸入：
    1
   / \
  2   2
 / \ / \
3  4 4  3

輸出：True

```
text
```python
輸入：
    1
   / \
  2   2
   \    \
   3     3

輸出：False

```
---

### **DFS 解法**

#### **算法思路**

對稱的二叉樹滿足以下條件：

1. 根節點的左右子樹是鏡像的。
2. 對於兩棵子樹：
    - 它們的根節點值相等。
    - 左子樹的左子樹和右子樹的右子樹是鏡像的。
    - 左子樹的右子樹和右子樹的左子樹是鏡像的。

我們使用深度優先搜索（DFS）遞歸檢查每對節點是否滿足鏡像條件。

---
Example:
**样例1**
```
输入: {1,2,2,3,4,4,3}
输出: true
解释:
    1
   / \
  2   2
 / \ / \
3  4 4  3
{1,2,2,3,4,4,3}这棵二叉树是对称的
```
**样例2**
```
输入: {1,2,2,#,3,#,3}
输出: false
解释:
    1
   / \
  2   2
   \   \
   3    3
很显然这棵二叉树并不对称
```


#### **代碼詳解**

```python
class Solution:
    def is_symmetric(self, root):
        if not root:
            return True  # 空樹是對稱的
        return self.dfs(root.left, root.right)  # 檢查左右子樹是否對稱

    def dfs(self, left, right):
        if not left and not right:  # 同時為空
            return True
        if not left or not right:  # 只有一個為空
            return False
        if left.val != right.val:  # 根節點值不同
            return False
        
        # 遞歸檢查左右子樹的對稱性
        return self.dfs(left.left, right.right) and self.dfs(left.right, right.left)

```
pass

#### **執行過程舉例**

輸入：
```python
    1
   / \
  2   2
 / \ / \
3  4 4  3

```

1. **初始調用**：
    
    - `root.left = 2, root.right = 2`。
    - 調用 `dfs(2, 2)`。
2. **第一層遞歸**：
    
    - `left = 2, right = 2`。
    - 值相等，進一步檢查：
        - 調用 `dfs(3, 3)` 檢查左子樹的左子樹和右子樹的右子樹。
        - 調用 `dfs(4, 4)` 檢查左子樹的右子樹和右子樹的左子樹。
3. **第二層遞歸**：
    
    - `dfs(3, 3)`：
        - 值相等，左右子樹均為空，返回 `True`。
    - `dfs(4, 4)`：
        - 值相等，左右子樹均為空，返回 `True`。
4. **回溯**：
    
    - 第一層返回 `True`，說明整棵樹是對稱的。

輸出：`True`

---

#### **複雜度分析**

1. **時間複雜度**：
    
    - 每個節點都會被訪問一次。
    - 時間複雜度為 **O(n)O(n)O(n)**，其中 nnn 是節點數量。
2. **空間複雜度**：
    
    - 遞歸調用棧的最大深度為樹的高度 hhh。
    - 空間複雜度為 **O(h)O(h)O(h)**，最壞情況下為 O(n)O(n)O(n)（當樹退化為鏈表）。

---

### **其他解法簡述**

#### 1. **BFS（迭代法）**

- 使用佇列，同時存放對稱的兩個節點。
- 每次從佇列取出兩個節點，檢查是否鏡像：
    - 若都為空，繼續。
    - 若只有一個為空或值不相等，返回 `False`。
    - 否則，將它們的子節點按對稱順序加入佇列。

```python
from collections import deque

class Solution:
    def isSymmetric(self, root):
        if not root:
            return True
        queue = deque([(root.left, root.right)])  # 初始化佇列
        
        while queue:
            left, right = queue.popleft()
            if not left and not right:
                continue
            if not left or not right:
                return False
            if left.val != right.val:
                return False
            # 按對稱順序加入子節點
            queue.append((left.left, right.right))
            queue.append((left.right, right.left))
        
        return True

```

**時間複雜度**：

- 與 DFS 相同，為 O(n)O(n)O(n)。

**空間複雜度**：

- 需要存儲節點對，最壞情況下為 O(n)O(n)O(n)。

---

#### 2. **簡化遞歸**

- 將根節點的左右子樹轉為字符串，檢查它們是否相等（不推薦，效率低）。

#### 3. **鏡像構造檢查**

- 构造一棵镜像树，检查是否与原树相等。