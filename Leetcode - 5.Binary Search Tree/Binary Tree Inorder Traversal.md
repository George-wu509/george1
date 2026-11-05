
|                               |     |
| ----------------------------- | --- |
| [[#### 甚麼是in-order, 解法1：递归法]] |     |
| [[#### 解法1詳細解釋]]              |     |
| [[#### 解法2：迭代法（栈）]]           |     |
| [[#### 解法1,2 都是DFS]]          |     |
|                               |     |


#### 甚麼是in-order, 解法1：递归法
```
甚麼是in-order, 解法1：递归法
```
LintCode第67题“二叉树的中序遍历”要求按**中序遍历**的顺序返回二叉树的所有节点值。
##### 什麼是中序遍历？
中序遍历是一种遍历二叉树的方法，按以下顺序访问节点：
1. 遍历左子树；
2. 访问根节点；
3. 遍历右子树。

###### 输入示例
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

##### 解法1：递归法
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

###### Python实现代码
```python
class TreeNode:
    def __init__(self, val=0):
        self.val = val
        self.left = None
        self.right = None

class Solution:
    def inorder_traversal(self, root: TreeNode) -> List[int]:
        result = []
        self.dfs(root, result)
        return result
               
    def dfs(self, node, result):
        if not node:
            return
        self.dfs(node.left, result)
        result.append(node.val)
        self.dfs(node.right, result)
```
pass
解釋: 



#### 解法1詳細解釋
```
這是leetcode Binary Tree Inorder Traversal的解法. 我知道in-order traversal是遍历二叉树的方法，按以下顺序访问节点：遍历左子树->访问根节点->遍历右子树. 不過我不太能理解如何用dfs做到那個遍歷的順序, 怎麼把這根
```
要理解 DFS 遞迴如何實現中序遍歷（Inorder Traversal），核心概念在於「呼叫堆疊（Call Stack）的暫停與恢復」。

當程式執行到 `self.dfs(node.left, result)` 時，當前節點的工作**尚未完成**，而是被「暫停」並壓入堆疊，直到左子樹全部走完回傳後，才會繼續執行下一行 `result.append(node.val)`。
![[Pasted image 20261105200007.png]]

##### 二元樹結構範例：`root = [1, 2, 3, 4, 5]`
按 LeetCode 的層序（Level-order）表示法，樹的結構如下：
```
        1
       / \
      2   3
     / \
    4   5
```
###### DFS 程式碼的 3 個步驟
在每個節點上，`dfs` 函式都執行相同的指令：
1. `self.dfs(node.left, result)` ：**先往左走到底**
2. `result.append(node.val)` ：**處理自己（根）**
3. `self.dfs(node.right, result)` ：**再往右探索**

###### 完整遞迴執行過程追蹤
我們著重觀察 `result` 的變化與程式碼行的執行順序：
###### 1. 從根節點 `1` 開始
- 呼叫 `dfs(1)`：執行 `dfs(1.left)`，暫停 `1` 的記錄，進入 `dfs(2)`。
###### 2. 來到節點 `2`
- 呼叫 `dfs(2)`：執行 `dfs(2.left)`，暫停 `2` 的記錄，進入 `dfs(4)`。
###### 3. 來到最左葉子節點 `4`
- 呼叫 `dfs(4)`：
    1. `dfs(4.left)` 進入 `None` 立即回傳。
    2. 執行 `result.append(4)` $\rightarrow$ **`result = [4]`**
    3. `dfs(4.right)` 進入 `None` 立即回傳。
- `dfs(4)` 執行完畢，**回到 `dfs(2)` 暫停的位置**。
###### 4. 回到節點 `2`（左子樹已完結）
- `dfs(2)` 的第 1 行已完成，繼續執行第 2 行：
    - 執行 `result.append(2)` $\rightarrow$ **`result = [4, 2]`**
- 執行第 3 行 `dfs(2.right)` $\rightarrow$ 進入 `dfs(5)`。
###### 5. 來到節點 `5
- 呼叫 `dfs(5)`：
    1. `dfs(5.left)` 進入 `None` 回傳
    2. 執行 `result.append(5)` $\rightarrow$ **`result = [4, 2, 5]`**
    3. `dfs(5.right)` 進入 `None` 回傳。
- `dfs(5)` 執行完畢，回到 `dfs(2)`。
###### 6. 回到節點 `2`
- 節點 `2` 的左右都已走完，`dfs(2)` 結束，**回到根節點 `dfs(1)` 暫停的位置**。
###### 7. 回到根節點 `1`（左子樹 `[4, 2, 5]` 全部完結）
- `dfs(1)` 的第 1 行已完成，繼續執行第 2 行：
    - 執行 `result.append(1)` $\rightarrow$ **`result = [4, 2, 5, 1]`**
- 執行第 3 行 `dfs(1.right)` $\rightarrow$ 進入 `dfs(3)`。
###### 8. 來到節點 `3`
- 呼叫 `dfs(3)`：
    1. `dfs(3.left)` 進入 `None` 回傳。
    2. 執行 `result.append(3)` $\rightarrow$ **`result = [4, 2, 5, 1, 3]`**
    3. `dfs(3.right)` 進入 `None` 回傳。
- `dfs(3)` 執行完畢，回到 `dfs(1)`。
###### 9. 結束
- `dfs(1)` 全部執行結束，最終回傳 `[4, 2, 5, 1, 3]`。

##### 總結
這三行程式碼之所以能達成「左 $\rightarrow$ 根 $\rightarrow$ 右」：
- 把 `self.dfs(node.left)` 放最上面，保證了**無條件一路向左鑽到最深處**才開始填寫數據。
- `result.append(node.val)` 放在中間，確保**左邊子樹處理完了，才記錄自己**。
- `self.dfs(node.right)` 放最後，確保**記錄完自己後，才進去右邊子樹重複相同的邏輯**。



---
#### 解法2：迭代法（栈）
```
解法2：迭代法（栈）
```
##### 解法2：迭代法（栈）
###### 思路
迭代法通过显式使用栈模拟递归过程，逐步完成中序遍历。
###### 步骤
1. 初始化一个空栈`stack`和结果数组`result`。
2. 从根节点开始，沿着左子树不断向下，将每个节点压入栈。
3. 当无法继续向左时，弹出栈顶节点，访问该节点并将其值加入结果数组。
4. 转向该节点的右子树并重复上述步骤。
5. 栈为空且当前节点为空时，遍历结束。
###### Python实现代码
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

###### 示例：`tree = {1,2,3,4,5,#,7,8}`

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




#### 解法1,2 都是DFS
```
binary tree的pre-order, in-order, post-order traversal是不是都有兩種方式實現, 一種就是前面提到的DFS 遞迴, 另一種是用stack的迭代法, 這是不是就是BFS? 那這兩種方法有沒有哪個比較推薦或哪個比較好? 也請詳細介紹用stack的迭代法: def inorderTraversal(root):
```

##### 觀念澄清：Stack 疊代法是 BFS 嗎？

**不是，Stack 疊代法依然是 DFS（Depth-First Search，深度優先搜尋）。**

- **DFS（深度優先）**：核心原理是 **LIFO（Last-In, First-Out，後進先出）**。
    
    - **遞迴法**：使用系統隱式的 Call Stack。
        
    - **疊代法**：使用我們自己宣告的 `stack = []`。
        
    - 兩者存取的順序與邏輯完全相同，因此 Pre-order、In-order、Post-order 的疊代法都屬於 **DFS**。
        
- **BFS（廣度優先）**：即「層序遍歷（Level-order Traversal）」，核心資料結構是 **Queue（佇列，FIFO 先進先出）**，是一層一層（按樹的深度）往外擴張走訪。
    

### 遞迴 vs 疊代：推薦哪一個？

|**特性**|**DFS 遞迴法**|**DFS Stack 疊代法**|
|---|---|---|
|**程式碼可讀性**|高，簡潔且邏輯直觀|較低，需手動維護狀態與指標|
|**記憶體風險**|當樹極深時（如 $N > 10,000$ 的單鏈樹），可能引發**系統 Stack Overflow**|記憶體在 Heap 區，**不會引發系統 Call Stack 溢位**|
|**時間與空間複雜度**|$O(N)$ 時間, $O(H)$ 空間 ($H$ 為樹高)|$O(N)$ 時間, $O(H)$ 空間 ($H$ 為樹高)|
|**面試實務推薦**|**首選**（80% 的情況下優先寫遞迴，簡短不易出錯）|面試官要求「不用遞迴」或強調極端記憶體安全時使用|

### Stack 疊代法詳細拆解

同樣以 `root = [1, 2, 3, 4, 5]` 為例，二元樹結構如下：

Plaintext

```
        1
       / \
      2   3
     / \
    4   5
```

#### 程式碼核心思維

疊代法用一個 `while current` 迴圈來模擬遞迴「一路向左鑽到底」的動作；當無法再向左時，從 `stack` 彈出頂端節點（相當於遞迴回溯），記錄該節點後，再將指標轉向其右子樹。

#### 執行步驟推演

|**步驟**|**當前狀態與操作**|**stack 變化**|**result 變化**|**current 走向**|
|---|---|---|---|---|
|**初始化**|`current = 1`|`[]`|`[]`|準備進入主迴圈|
|**1. 向左一路鑽**|`1` 入棧 $\rightarrow$ `2` 入棧 $\rightarrow$ `4` 入棧|`[1, 2, 4]`|`[]`|`current` 變為 `None`（`4.left`）|
|**2. 處理葉子 `4`**|內層迴圈結束，彈出 `4`，記錄 `4`|`[1, 2]`|`[4]`|`current = 4.right` (`None`)|
|**3. 處理節點 `2`**|`current` 為 `None`，彈出 `2`，記錄 `2`|`[1]`|`[4, 2]`|`current = 2.right` (`5`)|
|**4. 處理節點 `5`**|`5` 入棧 $\rightarrow$ 彈出 `5`，記錄 `5`|`[1]`|`[4, 2, 5]`|`current = 5.right` (`None`)|
|**5. 處理根節點 `1`**|`current` 為 `None`，彈出 `1`，記錄 `1`|`[]`|`[4, 2, 5, 1]`|`current = 1.right` (`3`)|
|**6. 處理節點 `3`**|`3` 入棧 $\rightarrow$ 彈出 `3`，記錄 `3`|`[]`|`[4, 2, 5, 1, 3]`|`current = 3.right` (`None`)|
|**7. 結束**|`stack` 為空且 `current` 為 `None`|`[]`|`[4, 2, 5, 1, 3]`|退出迴圈，傳回 `result`|

#### 程式碼兩段式邏輯解釋

1. **`while current:` 內層迴圈**：相當於遞迴呼叫 `dfs(node.left)`。只要當前節點存在，就把它壓入 `stack`（暫存起來，待會再處理），然後繼續往左走。
    
2. **`current = stack.pop()` 與 `current.right`**：相當於遞迴走完左邊後回到當前節點 `result.append(node.val)`，接著呼叫 `dfs(node.right)` 進入右子樹重複相同邏輯。





| 题目编号               | 题目名称（英文）                                            | 题目简述（中文）              | 样例                                                                                    | 解法                          |
| ------------------ | --------------------------------------------------- | --------------------- | ------------------------------------------------------------------------------------- | --------------------------- |
| 66<br><br>(e)      | [[Binary Tree Preorder Traversal]]二叉树的前序遍历          | 返回二叉树的前序遍历结果。         | 输入:  <br>root = <br>[1,null,2,3]  <br><br>输出:  <br>[1,2,3]                            | 使用递归或栈完成前序遍历操作。             |
| 67<br>*<br>(e)     | [[Binary Tree Inorder Traversal]]- 二叉树的中序遍历         | 返回二叉树的中序遍历结果。         | 输入:  <br>root = <br>[1,null,2,3]  <br><br>输出:  <br>[1,3,2]                            | 使用递归或栈完成中序遍历操作。             |
| 68<br><br>(e)<br>  | [[Binary Tree Postorder Traversal]]- 二叉树的后序遍历       | 返回二叉树的后序遍历结果。         | 输入:  <br>root = <br>[1,null,2,3]  <br><br>输出:  <br>[3,2,1]                            | 使用递归或栈完成后序遍历操作。             |
| 70<br>*<br>(m)<br> | [[Binary Tree Level Order Traversal II]]二叉树的层次遍历 II | 返回二叉树的层次遍历结果（从底层到顶层）。 | 输入:  <br>root = <br>[3,9,20,null,<br>null,15,7]  <br><br>输出:  [<br>[15,7],[9,20],[3]] | 使用队列按层遍历节点，结果反转。            |
| 480<br>**<br>(e)   | 二叉树的<br>所有路径 <br>[[Binary Tree Paths]] <br>         | 找出二叉树中从根到叶子的所有路径。     | 输入：<br>{1,2}<br><br>输出：<br>["1->2"]                                                   | 使用 DFS 遍历所有路径，记录从根到叶子的每条路径。 |
