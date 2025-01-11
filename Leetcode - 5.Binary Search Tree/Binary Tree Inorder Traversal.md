
LintCode第67题“二叉树的中序遍历”要求按**中序遍历**的顺序返回二叉树的所有节点值。

---

### 什么是中序遍历？

中序遍历是一种遍历二叉树的方法，按以下顺序访问节点：

1. 遍历左子树；
2. 访问根节点；
3. 遍历右子树。

---

### 输入示例

**输入：**

`tree = {1,2,3,4,5,#,7,8}`

对应的二叉树结构如下：

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

`[8, 4, 2, 5, 1, 3, 7]`

---

### 解法1：递归法

#### 思路

递归法是解决二叉树遍历问题的经典方法。我们按照中序遍历规则依次递归访问二叉树的节点。

#### 步骤

1. 如果当前节点为空，直接返回（递归的结束条件）。
2. 递归遍历左子树；
3. 访问当前节点并将其值加入结果数组；
4. 递归遍历右子树。

---
Example:
**样例 1：**
输入：
```
二叉树 = {1,2,3}
```
输出：
```
[2,1,3]
```
解释：
```
      1
    /   \
  2       3
```
它将被序列化为{1,2,3}中序遍历

**样例 2：**
输入：
```
二叉树 = {1,#,2,3}
```
输出：
```
[1,3,2]
```
解释：
```
     1
       \
        2
       /
      3
```
它将被序列化为{1,#,2,3}中序遍历

#### Python实现代码
```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class Solution:
    def inorder_traversal(self, root: TreeNode) -> List[int]:
        result = []

        def dfs(node):
            if not node:
                return
            dfs(node.left)           # 遍历左子树
            result.append(node.val)  # 访问当前节点
            dfs(node.right)          # 遍历右子树

        dfs(root)
        return result
```
pass

---

### 解法2：迭代法（栈）

#### 思路

迭代法通过显式使用栈模拟递归过程，逐步完成中序遍历。

#### 步骤

1. 初始化一个空栈`stack`和结果数组`result`。
2. 从根节点开始，沿着左子树不断向下，将每个节点压入栈。
3. 当无法继续向左时，弹出栈顶节点，访问该节点并将其值加入结果数组。
4. 转向该节点的右子树并重复上述步骤。
5. 栈为空且当前节点为空时，遍历结束。

---

#### Python实现代码
```python
def inorderTraversal(root):
    result = []
    stack = []
    current = root

    while stack or current:
        # 向左子树深入
        while current:
            stack.append(current)
            current = current.left

        # 弹出栈顶节点并访问
        current = stack.pop()
        result.append(current.val)

        # 转向右子树
        current = current.right

    return result

```

---

### 示例：`tree = {1,2,3,4,5,#,7,8}`

#### 输入二叉树结构：

```markdown
         1
       /   \
      2     3
     / \      \
    4   5      7
   /
  8

```

#### 构造树代码：

```python
root = TreeNode(1)
root.left = TreeNode(2)
root.right = TreeNode(3)
root.left.left = TreeNode(4)
root.left.right = TreeNode(5)
root.right.right = TreeNode(7)
root.left.left.left = TreeNode(8)

```

---

### 解法1：递归法详细过程

1. 从根节点`1`开始：
    
    - 遍历左子树：
        - 节点`2`的左子树：
            - 节点`4`的左子树：
                - 访问节点`8`，结果：`[8]`。
            - 返回节点`4`，结果：`[8, 4]`。
        - 返回节点`2`，结果：`[8, 4, 2]`。
        - 节点`2`的右子树：
            - 访问节点`5`，结果：`[8, 4, 2, 5]`。
    - 返回节点`1`，结果：`[8, 4, 2, 5, 1]`。
2. 遍历右子树：
    
    - 节点`3`的右子树：
        - 访问节点`7`，结果：`[8, 4, 2, 5, 1, 3, 7]`。

#### 输出：

`[8, 4, 2, 5, 1, 3, 7]`

---

### 解法2：迭代法详细过程

1. 初始化：
    
    - `stack = []`，`result = []`，`current = 1`。
2. 第一步：沿左子树深入：
    
    - 压入节点`1`：`stack = [1]`，`current = 2`。
    - 压入节点`2`：`stack = [1, 2]`，`current = 4`。
    - 压入节点`4`：`stack = [1, 2, 4]`，`current = 8`。
    - 压入节点`8`：`stack = [1, 2, 4, 8]`，`current = None`。
3. 第二步：弹出并访问：
    
    - 弹出节点`8`，结果：`result = [8]`，`current = None`。
    - 弹出节点`4`，结果：`result = [8, 4]`，`current = None`。
    - 弹出节点`2`，结果：`result = [8, 4, 2]`，`current = 5`。
4. 第三步：转向右子树：
    
    - 压入节点`5`：`stack = [1, 5]`，`current = None`。
    - 弹出节点`5`，结果：`result = [8, 4, 2, 5]`，`current = None`。
    - 弹出节点`1`，结果：`result = [8, 4, 2, 5, 1]`，`current = 3`。
5. 第四步：处理右子树：
    
    - 压入节点`3`：`stack = [3]`，`current = 7`。
    - 压入节点`7`：`stack = [3, 7]`，`current = None`。
    - 弹出节点`7`，结果：`result = [8, 4, 2, 5, 1, 3, 7]`。
6. 栈为空，结束遍历。
    

#### 输出：

`[8, 4, 2, 5, 1, 3, 7]`

---

### 总结

- **递归法**：
    
    - 简单直观，但对深度大的树可能造成栈溢出。
    - 时间复杂度：O(n)，空间复杂度：O(h)（递归栈深度为树的高度）。
- **迭代法**：
    
    - 更适合深度大的树，通过显式栈模拟递归过程。
    - 时间复杂度：O(n)，空间复杂度：O(h)。

两种方法都能正确完成中序遍历，希望这些详细步骤对你有帮助！如果有问题，请随时提问！ 😊

