390
给定一个整数矩阵 `A`, 它有如下特性:

- 相邻的整数不同
- 矩阵有 `n` 行 `m` 列，n和m不会小于3。
- 对于所有的 `i < n`, 都有 `A[i][0] < A[i][1] && A[i][m - 2] > A[i][m - 1]`
- 对于所有的 `j < m`, 都有 `A[0][j] < A[1][j] && A[n - 2][j] > A[n - 1][j]`

我们定义一个位置 `[i,j]` 是峰值, 当且仅当它满足
找到该矩阵的一个峰值元素, 返回它的坐标.

**样例 1:**
```python
输入: 
    [
      [1, 2, 3, 6,  5],
      [16,41,23,22, 6],
      [15,17,24,21, 7],
      [14,18,19,20,10],
      [13,14,11,10, 9]
    ]
输出: [1,1]
解释: [2,2] 也是可以的. [1,1] 的元素是 41, 大于它四周的每一个元素 (2, 16, 23, 17).
```
**样例 2:**
```python
输入: 
    [
      [1, 5, 3],
      [4,10, 9],
      [2, 8, 7]
    ]
输出: [1,1]
解释: 只有这一个峰值
```


```python
class Solution:
   
    def find_peak_i_i(self, A):
        left, right = 0, len(A[0]) - 1
        up, down = 0, len(A) - 1
        
        while left + 1 < right or up + 1 < down:
            if up - down >= right - left:
                row = (up + down) // 2
                max_col = self.findMaxCol(row, left, right, A)
                if A[row][max_col] < A[row - 1][max_col]:
                    down = row - 1
                elif A[row][max_col] < A[row + 1][max_col]: 
                    up = row + 1
                else:
                    return [row, max_col]
            else:
                col = (left + right) // 2
                max_row = self.findMaxRow(up, down, col, A)
                if A[max_row][col] < A[max_row][col + 1]:
                    left = col + 1
                elif A[max_row][col] < A[max_row][col - 1]:
                    right = col - 1
                else:
                    return [max_row, col]
        
        return [up, left]
    
    def findMaxRow(self, up, down, col, A):
        max_val = -sys.maxsize
        row = -1
        for i in range(up, down + 1):
            if A[i][col] > max_val:
                max_val = A[i][col]
                row = i
        return row
    
    def findMaxCol(self, row, left, right, A):
        max_val = -sys.maxsize
        col = -1
        for i in range(left, right + 1):
            if A[row][i] > max_val:
                max_val = A[row][i]
                col = i
        return col
```
pass



### **LintCode 390：寻找峰值 II**

---

#### **题目描述**

给定一个二维数组，找到一个局部峰值。峰值的定义是：

- 数组中的某个元素比其上下左右四个相邻元素都大。
- 边界上的元素只有和存在的邻居比较。

输出任意一个峰值的位置 `(row, col)` 即可。

---

## **解法：二维二分搜索法（双指针）**

### **核心思路**

1. 利用 **二维数组的特性**：
    
    - 峰值一定存在（数学证明）。
    - 每次检查一列（或一行）的最大值，再在相邻行（或列）中继续搜索，直到找到一个峰值。
2. 使用双指针进行二分搜索：
    
    - 每次选择中间的一列（或行）。
    - 找到该列（或行）的最大值位置，作为候选峰值。
    - 比较该候选位置与相邻位置的值，决定向左、右（或上、下）哪一侧继续搜索。
3. **收敛性**：
    
    - 每次选择中间列（或行），问题规模减半，时间复杂度为 `O(n log m)` 或 `O(m log n)`（`n` 是行数，`m` 是列数）。

---

### **详细步骤**

#### **初始化**

1. 使用双指针从两端向中间收敛：
    
    - 如果按列搜索：`left = 0, right = len(matrix[0]) - 1`。
    - 如果按行搜索：`top = 0, bottom = len(matrix) - 1`。
2. 每次选择中间列（或行）：
    
    - 找到该列（或行）的最大值及其位置。
3. 比较：
    
    - 如果最大值比相邻列（或行）大，说明找到了一个峰值。
    - 否则，向值更大的那一侧继续搜索。

#### **停止条件**

- 当搜索范围收敛到一个单一列（或行）时，直接返回峰值的位置。

---

### **代码实现**
```python
def findPeakII(matrix):
    def get_col_max(mid_col):
        max_row = 0
        max_value = matrix[0][mid_col]
        for i in range(len(matrix)):
            if matrix[i][mid_col] > max_value:
                max_value = matrix[i][mid_col]
                max_row = i
        return max_row, max_value

    left, right = 0, len(matrix[0]) - 1
    while left <= right:
        mid_col = (left + right) // 2
        max_row, max_value = get_col_max(mid_col)

        # 比较与相邻列的值
        if mid_col > 0 and matrix[max_row][mid_col] < matrix[max_row][mid_col - 1]:
            right = mid_col - 1
        elif mid_col < len(matrix[0]) - 1 and matrix[max_row][mid_col] < matrix[max_row][mid_col + 1]:
            left = mid_col + 1
        else:
            # 找到峰值
            return max_row, mid_col

```

---

### **示例分析**

#### **输入矩阵**

plaintext

複製編輯

`[   [1,  2,  3,  6,  5],   [16, 41, 23, 22, 6],   [15, 17, 24, 21, 7],   [14, 18, 19, 20, 10],   [13, 14, 11, 10, 9] ]`

#### **运行过程**

1. **第一步**：`left = 0, right = 4`，选择中间列 `mid_col = 2`。
    
    - 在列 2 中找到最大值：`matrix[2][2] = 24`，位置 `(2, 2)`。
    - 比较与列 1 和列 3：
        - `matrix[2][2] > matrix[2][1]`（24 > 17）。
        - `matrix[2][2] < matrix[2][3]`（24 < 21）。
    - 向右搜索，`left = 3`。
2. **第二步**：`left = 3, right = 4`，选择中间列 `mid_col = 3`。
    
    - 在列 3 中找到最大值：`matrix[3][3] = 20`，位置 `(3, 3)`。
    - 比较与列 2 和列 4：
        - `matrix[3][3] > matrix[3][2]`（20 > 19）。
        - `matrix[3][3] > matrix[3][4]`（20 > 10）。
    - 找到峰值，返回 `(3, 3)`。

#### **输出结果**

返回峰值位置 `(3, 3)`。

---

### **复杂度分析**

1. **时间复杂度**：
    
    - 每次选择一列，需 `O(n)` 时间找到该列的最大值。
    - 搜索范围每次减半，需进行 `log m` 次搜索。
    - 总时间复杂度为 **`O(n log m)`**。
2. **空间复杂度**：
    
    - 使用常数空间，空间复杂度为 **`O(1)`**。

---

### **其他解法**

1. **暴力法**：
    
    - 遍历整个矩阵，检查每个元素是否是局部峰值。
    - 时间复杂度：`O(n * m)`。
    - 缺点：效率低，不适合大规模矩阵。
2. **行二分搜索**：
    
    - 类似列二分搜索，按行进行二分搜索，每次找到行中的最大值并向上或向下搜索。
    - 时间复杂度：`O(m log n)`。

---

### **双指针的选择比较**

1. **按列进行双指针搜索**：
    
    - `left, right = 0, len(matrix[0]) - 1`。
    - 更适合矩阵列数较少、行数较多的情况。
2. **按行进行双指针搜索**：
    
    - `top, bottom = 0, len(matrix) - 1`。
    - 更适合矩阵行数较少、列数较多的情况。

---

### **总结**

- 使用 **二维二分搜索法** 是解决峰值问题的高效方法，时间复杂度为 `O(n log m)` 或 `O(m log n)`。
- 根据矩阵形状选择按列或按行搜索。
- 暴力法虽然简单，但时间效率较低。