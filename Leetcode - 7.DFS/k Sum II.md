Lintcode 90
给定 `n` 个不同的正整数，整数 `k`（1<=k<=n1<=k<=n）以及一个目标数字。在这 `n` 个数里面找出 `k` 个不同的数，使得这 `k` 个数的和等于目标数字，你需要找出所有满足要求的方案（方案顺序不作要求）。


Example:
**样例 1：**
输入：
```python
"""
数组 = [1,2,3,4]
k = 2
target = 5
```
输出：
```python
"""
[[1,4],[2,3]]
```
解释：
1+4=5,2+3=5

**样例 2：**
输入：
```python
"""
数组 = [1,3,4,6]
k = 3
target = 8
```
输出：
```python
"""
[[1,3,4]]
```
解释：
1+3+4=8

#### **代碼解析**

```python
class Solution:
    def kSumII(self, A, k, target):
        A = sorted(A)  # 排序數組，方便剪枝
        subsets = []   # 保存結果
        self.dfs(A, 0, k, target, [], subsets)
        return subsets

    def dfs(self, A, index, k, target, subset, subsets):
        # 遞歸終止條件
        if k == 0 and target == 0:
            subsets.append(list(subset))  # 找到符合條件的子集
            return
        
        if k == 0 or target <= 0:  # 剪枝條件
            return

        # 遍歷剩餘的數字
        for i in range(index, len(A)):
            subset.append(A[i])  # 選擇當前數字
            self.dfs(A, i + 1, k - 1, target - A[i], subset, subsets)  # 遞歸處理下一層
            subset.pop()  # 回溯，移除當前數字

```
pass

## **解法：回溯（DFS + 剪枝）**

### **核心思路**

1. **使用回溯（DFS）枚举所有可能的子集**
    
    - 递归地选择或不选择当前元素，直到找到 `k` 个元素并且其和等于 `target`。
2. **剪枝优化**
    
    - **提前排序** 数组 `A`，使得递归过程可以跳过不可能的情况。
    - **如果 `k == 0` 但 `target != 0`，提前剪枝**，避免不必要的递归。
    - **如果 `target` 变为负数，提前返回**，避免继续搜索无效解。

---

## **代码解析**
```python
class Solution:
    def kSumII(self, A, k, target):
        A = sorted(A)  # 先排序，方便剪枝
        subsets = []   # 结果数组
        self.dfs(A, 0, k, target, [], subsets)
        return subsets

    def dfs(self, A, index, k, target, subset, subsets):
        # 递归终止条件
        if k == 0 and target == 0:
            subsets.append(list(subset))  # 找到符合条件的子集
            return
        
        if k == 0 or target <= 0:  # 剪枝
            return

        # 遍历剩余的数字
        for i in range(index, len(A)):
            subset.append(A[i])  # 选择当前数字
            self.dfs(A, i + 1, k - 1, target - A[i], subset, subsets)  # 递归搜索
            subset.pop()  # 回溯

```

---

## **执行过程**

**输入**

`A = [1,2,3,4] k = 2 target = 5`

---

### **Step 1: 排序**

`A = sorted(A)  # A 变为 [1, 2, 3, 4]`

虽然本题不要求排序，但排序后可以 **优化剪枝过程**，确保递归时不出现重复选择的情况。

---

### **Step 2: 递归搜索**

`dfs(A, index=0, k=2, target=5, subset=[], subsets=[])`

- `index = 0`，选择 `1`

    `subset = [1], target = 4, k = 1`
    
    继续递归：

    `dfs(A, index=1, k=1, target=4, subset=[1], subsets=[])`
    

---

### **Step 3: 继续递归**

- 选择 `2`

    `subset = [1,2], target = 2, k = 0`
    
    **不符合条件**，回溯。
    
- 选择 `3`

    `subset = [1,3], target = 1, k = 0`
    
    **不符合条件**，回溯。
    
- 选择 `4`

    `subset = [1,4], target = 0, k = 0`
    
    **找到解**，加入 `subsets`。
    

---

### **Step 4: 继续搜索**

回到 `index = 0`，不选 `1`，直接选 `2`：

`subset = [2], target = 3, k = 1`

继续搜索：

`dfs(A, index=2, k=1, target=3, subset=[2], subsets=[[1,4]])`

- 选择 `3`

    `subset = [2,3], target = 0, k = 0`
    
    **找到解**，加入 `subsets`。

---

### **最终结果**

`subsets = [[1, 4], [2, 3]]`

---

## **时间与空间复杂度分析**

### **时间复杂度**

1. **最坏情况下**，`A` 有 `n` 个元素，我们需要从 `n` 个数中选 `k` 个，即：
    
    $C(n, k) = \frac{n!}{k!(n-k)!}$
    
    **复杂度 ≈ `O(2^n)`，但实际远小于 `2^n`**，因为剪枝减少了搜索空间。
    
2. **排序** 复杂度为 `O(n log n)`，但相对于 `O(2^n)` 级别，影响较小。
    

### **空间复杂度**

1. **递归深度**：最多 `k` 层，即 `O(k)`
2. **存储 `subsets` 结果**：最坏情况 `O(C(n, k))`

---

## **其他解法**

### **1. 回溯（不排序）**

- **思路**
    - 直接在 `A` 的原始顺序下进行回溯，不排序。
    - **适用于无需剪枝优化的情况**。

### **2. 迭代法（非递归）**

- **思路**
    - 维护 `k` 个选择项，每次遍历 `A` 中的数字进行组合。
    - 适用于 `k` 较小时，避免递归栈溢出。

---

## **方法比较**

|方法|思路|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|---|
|**回溯（当前解法）**|**DFS + 剪枝**|`O(C(n,k))`|`O(k)`|**适用于大数据集**|
|**回溯（不排序）**|**DFS 遍历所有组合**|`O(2^n)`|`O(k)`|**适用于小规模数据**|
|**迭代法**|**遍历所有 `k` 个组合**|`O(C(n,k))`|`O(k)`|**适用于 `k` 较小时**|

---

## **总结**

- **最优解** ✅ **回溯 + 剪枝 `O(C(n,k))`**
- **数据较小时可用 `O(2^n)` 的直接回溯**
- **若 `k` 较小，可使用迭代法**

🚀 **"回溯 + 剪枝" 是最优解，适用于大规模数据！**