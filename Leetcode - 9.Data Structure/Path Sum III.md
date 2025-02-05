Lintcode 1240
给定一个二叉树，它的每个结点都存放着一个整数值。找出路径和等于给定数值的路径总数。
路径不需要从根节点开始，也不需要在叶子节点结束，但是路径方向必须是向下的（只能从父节点到子节点）。
二叉树不超过1000个节点，且节点数值范围是 [-1000000,1000000] 的整数。

**样例 1:**
```python
输入：root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8
输出：3
解释：
      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

返回 3。 和为8的路径为:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```
**样例 2:**
```python
输入：root = [11,6,-3], sum = 17
输出：1
解释：
      11
     /  \
    6   -3
返回 1。 和为17的路径为:

1.  11 -> 6
```


```python
def path_sum(self, root: TreeNode, sum: int) -> int:
	prefix = collections.defaultdict(int)
	prefix[0] = 1

	def dfs(root, curr):
		if not root:
			return 0
		
		ret = 0
		curr += root.val
		ret += prefix[curr - sum]
		prefix[curr] += 1
		ret += dfs(root.left, curr)
		ret += dfs(root.right, curr)
		prefix[curr] -= 1

		return ret

	return dfs(root, 0)
```
pass


# **LintCode 1240: Path Sum III（路径总和 III）**

---

## **题目描述**

给定一个二叉树 `root` 和一个整数 `sum`，求**路径总数**，使得每条路径上的节点值之和等于 `sum`。  
路径可以从任何节点开始，也可以在任何节点结束，但必须是**从父节点到子节点的连续路径**。

---

## **示例**
```python
输入:
       10
      /  \
     5   -3
    / \    \
   3   2   11
  / \   \
 3  -2   1
sum = 8

输出:
3

```
sum = 8  输出: 3`

**解释**： 满足路径和等于 `8` 的路径有：

1. `[5, 3]`
2. `[5, 2, 1]`
3. `[-3, 11]`

---

## **解法：前缀和 + DFS（双指针）**

### **核心思路**

4. **前缀和**（Prefix Sum）
    
    - 记录**从根节点到当前节点的路径和**。
    - 如果在当前路径上有 `prefix[curr] - sum` 出现，则存在一条路径使得 `sum` 恰好等于目标值。
5. **哈希表存储前缀和**
    
    - 维护 `prefix` 这个 **哈希表（字典）** 记录已访问的路径和的次数。
    - **`prefix[curr] - sum` 在 `prefix` 中的次数表示满足条件的路径数**。
6. **DFS 遍历树**
    
    - **前序遍历（Preorder DFS）**，计算当前路径和 `curr` 并更新 `prefix`。
    - **递归左子树** 和 **右子树**。
    - **回溯**（Backtracking）：在回到上层时，撤销 `prefix[curr]` 的修改，确保不影响其他路径。

---

## **代码解析**
```python
import collections

def path_sum(self, root: TreeNode, sum: int) -> int:
    prefix = collections.defaultdict(int)
    prefix[0] = 1  # 记录初始路径和0出现1次

    def dfs(root, curr):
        if not root:
            return 0
        
        ret = 0
        curr += root.val  # 更新当前路径和

        # 计算路径总数
        ret += prefix[curr - sum]  # 找到之前路径和为 curr - sum 的次数

        # 记录当前路径和
        prefix[curr] += 1
        
        # 递归遍历左右子树
        ret += dfs(root.left, curr)
        ret += dfs(root.right, curr)

        # 回溯，撤销当前路径和的影响
        prefix[curr] -= 1

        return ret

    return dfs(root, 0)

```
---

## **逐步执行分析**

**输入**

`root = [10,5,-3,3,2,None,11,3,-2,None,1] sum = 8`

**树结构**
```python
       10
      /  \
     5   -3
    / \    \
   3   2   11
  / \   \
 3  -2   1

```
### **执行 DFS 遍历**

#### **初始化**

|变量|值|
|---|---|
|`prefix = {0: 1}`|记录路径和0出现1次|
|`curr = 0`|当前路径和|

---

#### **DFS 递归步骤**

##### **遍历 `10`**

- `curr = 10`
- `prefix = {0:1, 10:1}`
- **继续递归左子树 `5`**。

##### **遍历 `5`**

- `curr = 10 + 5 = 15`
- `prefix = {0:1, 10:1, 15:1}`
- **继续递归左子树 `3`**。

##### **遍历 `3`**

- `curr = 15 + 3 = 18`
- `prefix = {0:1, 10:1, 15:1, 18:1}`
- **找到 `prefix[curr - sum] = prefix[18 - 8] = prefix[10] = 1`**，路径 `[5, 3]` **满足条件**。

##### **遍历 `3`（左子树）**

- `curr = 18 + 3 = 21`
- `prefix = {0:1, 10:1, 15:1, 18:1, 21:1}`
- **没有符合的路径**，回溯。

##### **遍历 `-2`**

- `curr = 18 - 2 = 16`
- `prefix = {0:1, 10:1, 15:1, 18:1, 16:1}`
- **没有符合的路径**，回溯。

---

##### **遍历 `2`（右子树）**

- `curr = 15 + 2 = 17`
- `prefix = {0:1, 10:1, 15:1, 17:1}`
- **没有符合的路径**。

##### **遍历 `1`**

- `curr = 17 + 1 = 18`
- **找到 `prefix[18 - 8] = prefix[10] = 1`**，路径 `[5, 2, 1]` **满足条件**。

---

##### **遍历 `-3`（右子树）**

- `curr = 10 - 3 = 7`
- **没有符合的路径**。

##### **遍历 `11`**

- `curr = 7 + 11 = 18`
- **找到 `prefix[18 - 8] = prefix[10] = 1`**，路径 `[-3, 11]` **满足条件**。

最终，**找到 3 条路径和等于 `8`**：

7. `[5, 3]`
8. `[5, 2, 1]`
9. `[-3, 11]`

返回：

`3`

---

## **时间与空间复杂度分析**

10. **时间复杂度**
    
    - **`O(n)`**：每个节点访问一次，进行 `O(1)` 操作。
11. **空间复杂度**
    
    - **`O(n)`**：最坏情况下，哈希表 `prefix` 需要存储 `O(n)` 个路径和。

---

## **其他解法**

### **1. 朴素 DFS（O(n^2)）**

- **思路**
    - 对每个节点，**遍历所有子路径** 计算和。
    - **缺点**：每个节点都递归遍历 `O(n)`，总复杂度 **`O(n^2)`**。
```python
def path_sum(root, sum):
    if not root:
        return 0
    
    return count_paths(root, sum) + path_sum(root.left, sum) + path_sum(root.right, sum)

def count_paths(node, target):
    if not node:
        return 0

    return (1 if node.val == target else 0) + count_paths(node.left, target - node.val) + count_paths(node.right, target - node.val)

```

**时间复杂度 `O(n^2)`，空间复杂度 `O(1)`。**

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + DFS（最佳解法）**|`O(n)`|`O(n)`|**适用于大规模数据**|
|朴素 DFS|`O(n^2)`|`O(1)`|**数据量较小时可用**|

🚀 **前缀和 `O(n)` 解法是最优，适用于大规模数据！**