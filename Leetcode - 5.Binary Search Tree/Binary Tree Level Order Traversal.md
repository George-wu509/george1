
LintCode第69题“二叉树的层次遍历”要求返回按层次顺序遍历二叉树的结果。下面我们通过详细解释和具体例子（`tree = {1,2,3,4,5,#,7,8}`）一步步讲解解决方法。

---

### 问题描述

给定一个二叉树，返回按层次顺序（从上到下、从左到右）遍历的结果，输出为一个二维数组。

**示例：**

**输入：**

`tree = {1,2,3,4,5,#,7,8}`

树的结构如下：

```markdown
         1
       /   \
      2     3
     / \      \
    4   5      7
   /
  8

```
**输出：**

`[   [1],   [2, 3],   [4, 5, 7],   [8] ]`

---

### 解法：广度优先搜索（BFS）

我们用**广度优先搜索（BFS）**来逐层遍历树。BFS通过队列来实现，它能够保证按照层次顺序遍历每个节点。

#### 步骤说明

1. **初始化**：
    
    - 创建一个空列表`result`用于保存最终结果。
    - 创建一个队列`queue`，初始只包含树的根节点。
2. **按层遍历**：
    
    - 每次处理队列中的所有节点（当前层的节点），并将它们的值存入一个临时数组`current_level`。
    - 如果某节点有左或右子节点，将这些子节点加入队列。
3. **保存当前层结果**：
    
    - 将`current_level`添加到`result`中。
4. **继续下一层**：
    
    - 直到队列为空，表示所有节点已遍历完。
5. **返回结果**：
    
    - 输出最终的`result`。

---

### Python 实现代码

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

def levelOrder(root):
    if not root:
        return []

    result = []
    queue = deque([root])  # 初始化队列

    while queue:
        level_size = len(queue)  # 当前层的节点数量
        current_level = []      # 存储当前层节点的值

        for _ in range(level_size):
            node = queue.popleft()  # 弹出队首节点
            current_level.append(node.val)  # 记录节点值

            # 将子节点加入队列
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(current_level)  # 将当前层结果加入最终结果

    return result

```
---

### 具体例子：`tree = {1,2,3,4,5,#,7,8}`

#### 输入树结构：

```markdown
         1
       /   \
      2     3
     / \      \
    4   5      7
   /
  8

```

#### 构造树：
```python
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.right = TreeNode(7)
root.left.left.left = TreeNode(8)

```


#### 步骤详解：

1. **初始化**：
    
    - `result = []`
    - `queue = deque([1])`
2. **第一层遍历**：
    
    - `level_size = 1`
    - `current_level = []`
    - 处理节点1：
        - 弹出节点1，加入`current_level = [1]`。
        - 节点1的左孩子2和右孩子3加入队列。
    - 当前层结束：`result = [[1]]`。
3. **第二层遍历**：
    
    - `level_size = 2`
    - `current_level = []`
    - 处理节点2：
        - 弹出节点2，加入`current_level = [2]`。
        - 节点2的左孩子4和右孩子5加入队列。
    - 处理节点3：
        - 弹出节点3，加入`current_level = [2, 3]`。
        - 节点3的右孩子7加入队列。
    - 当前层结束：`result = [[1], [2, 3]]`。
4. **第三层遍历**：
    
    - `level_size = 3`
    - `current_level = []`
    - 处理节点4：
        - 弹出节点4，加入`current_level = [4]`。
        - 节点4的左孩子8加入队列。
    - 处理节点5：
        - 弹出节点5，加入`current_level = [4, 5]`。
        - 节点5无孩子。
    - 处理节点7：
        - 弹出节点7，加入`current_level = [4, 5, 7]`。
        - 节点7无孩子。
    - 当前层结束：`result = [[1], [2, 3], [4, 5, 7]]`。
5. **第四层遍历**：
    
    - `level_size = 1`
    - `current_level = []`
    - 处理节点8：
        - 弹出节点8，加入`current_level = [8]`。
        - 节点8无孩子。
    - 当前层结束：`result = [[1], [2, 3], [4, 5, 7], [8]]`。
6. **结束遍历**：
    
    - 队列为空，返回`result`。

---

#### 最终输出：


`[   [1],   [2, 3],   [4, 5, 7],   [8] ]`

---

### 总结

- **时间复杂度**：O(n)，每个节点只访问一次。
- **空间复杂度**：O(n)，队列中可能存储所有节点。

通过BFS，我们可以高效地按层次遍历二叉树并返回结果。希望这个详细步骤对你有所帮助！如果有疑问，请随时提出！ 😊