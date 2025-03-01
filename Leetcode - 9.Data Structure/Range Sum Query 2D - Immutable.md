Lintcode 665
给一 二维矩阵,计算由左上角 `(row1, col1)` 和右下角 `(row2, col2)` 划定的矩形内元素和.

**样例1**
```python
"""
输入：
[[3,0,1,4,2],[5,6,3,2,1],[1,2,0,1,5],[4,1,0,1,7],[1,0,3,0,5]]
sumRegion(2, 1, 4, 3)
sumRegion(1, 1, 2, 2)
sumRegion(1, 2, 2, 4)
输出：
8
11
12
解释：
给出矩阵
[
  [3, 0, 1, 4, 2],
  [5, 6, 3, 2, 1],
  [1, 2, 0, 1, 5],
  [4, 1, 0, 1, 7],
  [1, 0, 3, 0, 5]
]
sumRegion(2, 1, 4, 3) = 2 + 0 + 1 + 1 + 0 + 1 + 0 + 3 + 0 = 8
sumRegion(1, 1, 2, 2) = 6 + 3 + 2 + 0 = 11
sumRegion(1, 2, 2, 4) = 3 + 2 + 1 + 0 + 1 + 5 = 12
```
**样例2**
```python
"""
输入：
[[3,0],[5,6]]
sumRegion(0, 0, 0, 1)
sumRegion(0, 0, 1, 1)
输出：
3
14
解释：
给出矩阵
[
  [3, 0],
  [5, 6]
]
sumRegion(0, 0, 0, 1) = 3 + 0 = 3
sumRegion(0, 0, 1, 1) = 3 + 0 + 5 + 6 = 14
```


```python
class NumMatrix:
    """
    @param: matrix: a 2D matrix
    """
    def __init__(self, matrix):
        self.prefix = [[0] * (len(matrix[0]) + 1) for _ in range(len(matrix) + 1)]
        for i in range(1, len(matrix) + 1):
            for j in range(1, len(matrix[0]) + 1):
                self.prefix[i][j] = self.prefix[i - 1][j] + self.prefix[i][j - 1] - self.prefix[i - 1][j - 1] + matrix[i - 1][j - 1]

    """
    @param: row1: An integer
    @param: col1: An integer
    @param: row2: An integer
    @param: col2: An integer
    @return: An integer
    """
    def sumRegion(self, row1, col1, row2, col2):
        return self.prefix[row2 + 1][col2 + 1] - self.prefix[row1][col2 + 1] - self.prefix[row2 + 1][col1] + self.prefix[row1][col1]
```
pass

# **LintCode 665 - Range Sum Query 2D - Immutable（二维前缀和查询）**

## **题目解析**

给定一个 `m x n` 的矩阵 `matrix`，要求实现：

1. **预处理矩阵，使得 `sumRegion(row1, col1, row2, col2)` 能在 `O(1)` 时间内查询某个子矩阵的总和**。
2. **矩阵内容不可变**，即数据初始化后不能修改。

这道题的关键在于 **使用前缀和（Prefix Sum）来优化查询**，从而避免 `O(mn)` 的暴力求和计算。

---

## **解法解析**

### **思路**

1. **构建二维前缀和数组 `prefix[i][j]`**：
    
    - `prefix[i][j]` 表示**从矩阵 `(0,0)` 到 `(i-1,j-1)` 之间所有元素的总和**。
    - 通过 **动态规划递推公式** 计算前缀和： 
    
	`prefix[i][j]=prefix[i-1][j]+prefix[i][j-1]-prefix[i-1][j-1]+matrix[i-1][j-1]`

    - 这个公式的直觉理解是：
        - `prefix[i-1][j]` 是当前行以上的总和
        - `prefix[i][j-1]` 是当前列左侧的总和
        - `prefix[i-1][j-1]` 被重复计算了一次，所以要减掉
        - `matrix[i-1][j-1]` 是当前格子自身的值。
2. **计算任意区域 `[row1, col1]` 到 `[row2, col2]` 的总和**：
    
    - 通过前缀和数组快速计算矩形区域的总和： 

`sumRegion(row1,col1,row2,col2)=prefix[row2+1][col2+1]-prefix[row1][col2+1]-prefix[row2+1][col1]+prefix[row1][col1]`

    - 解释：
        - `prefix[row2+1][col2+1]`：从 `(0,0)` 到 `(row2, col2)` 的总和。
        - `prefix[row1][col2+1]`：减去 `(0,0)` 到 `(row1-1, col2)` 的部分（上方）。
        - `prefix[row2+1][col1]`：减去 `(0,0)` 到 `(row2, col1-1)` 的部分（左方）。
        - `prefix[row1][col1]`：由于前两项都减去了 `(0,0)` 到 `(row1-1, col1-1)` 的部分，因此需要加回来。

---

## **具体举例**

假设 `matrix = [[3, 0, 1, 4, 2], [5, 6, 3, 2, 1], [1, 2, 0, 1, 5], [4, 1, 0, 1, 7], [1, 0, 3, 0, 5]]`，矩阵如下：
```python
"""
  3   0   1   4   2
  5   6   3   2   1
  1   2   0   1   5
  4   1   0   1   7
  1   0   3   0   5
```

### **Step 1: 计算 `prefix[][]`**

`prefix` 的构造如下：
```python
"""
  0   0   0   0   0   0
  0   3   3   4   8  10
  0   8  14  18  24  27
  0   9  17  21  28  36
  0  13  22  26  34  49
  0  14  23  30  38  58
```

例如：

- `prefix[3][3] = 21`，表示 **从 `(0,0)` 到 `(2,2)` 的总和**：
```python
"""
3   0   1
5   6   3
1   2   0
```
    
    计算方式：

    `3 + 0 + 1 + 5 + 6 + 3 + 1 + 2 + 0 = 21`
    

---

### **Step 2: 计算 `sumRegion(row1, col1, row2, col2)`**

假设 `sumRegion(2, 1, 4, 3)`，即求 `(2,1)` 到 `(4,3)` 的总和：

複製編輯

  `2   0   1   1   0   1   0   3   0`

计算：
```python
"""
prefix[5][4] - prefix[2][4] - prefix[5][1] + prefix[2][1]
= 38 - 8 - 14 + 3
= 19

```
最终答案 `19`。

---

## **时间与空间复杂度分析**

### **时间复杂度**

- **初始化 `prefix[][]`**：
    - 需要遍历 `O(m * n)` 次，**O(mn)**。
- **查询 `sumRegion()`**：
    - 仅涉及 4 次访问，**O(1)**。

综合来看：

- **预处理：O(mn)**
- **查询：O(1)**

### **空间复杂度**

- 额外存储 `prefix`，大小为 `O((m+1) * (n+1)) = O(mn)`。
- **空间复杂度：O(mn)**。

---

## **其他解法**

### **1. 暴力解法（O(mn) 查询）**

- **思路**：
    - 每次 `sumRegion(row1, col1, row2, col2)` 时，直接遍历子矩阵 **逐个累加**。
- **时间复杂度**：
    - **O(mn) 查询，O(1) 预处理**（无预处理）。
- **适用于**：
    - 仅需查询 **极少** 次。

### **2. 一维前缀和（空间 O(n) 优化）**

- **思路**：
    - 仅存每行的前缀和，查询时 **按行计算** 总和。
- **时间复杂度**：
    - 预处理：O(mn)
    - 查询：O(m)
- **空间复杂度**：
    - O(n)
- **适用于**：
    - 行数 `m` 很大，列数 `n` 较小。

---

## **LintCode 相关题目**

这类题目通常涉及 **前缀和、动态规划、矩阵查询优化**，以下是相似题目：

|**题号**|**题目名称**|**难度**|**核心技术**|
|---|---|---|---|
|**LintCode 665**|Range Sum Query 2D - Immutable|⭐⭐⭐|**二维前缀和**|
|**LintCode 207**|Interval Sum|⭐⭐⭐|**一维前缀和**|
|**LintCode 248**|Count of Smaller Number|⭐⭐⭐|**树状数组（Fenwick Tree）**|
|**LintCode 1310**|Shortest Subarray with Sum at Least K|⭐⭐⭐⭐|**前缀和 + 双端队列**|
|**LintCode 603**|Largest Divisible Subset|⭐⭐⭐|**DP + 组合数**|
|**LintCode 1828**|Queries on Number of Points Inside a Circle|⭐⭐⭐|**二维前缀和查询**|

---

## **总结**

1. **最优解法：二维前缀和**
    
    - 预处理 `O(mn)`，查询 `O(1)`，空间 `O(mn)`。
2. **其他解法**
    
    - **暴力解法**：查询 `O(mn)`，适用于极少查询。
    - **行前缀和优化**：查询 `O(m)`，适用于列较少。
3. **关键知识点**
    
    - **前缀和优化矩阵查询**
    - **动态规划式 `prefix[i][j]` 计算**
    - **快速 `O(1)` 计算区域和**
-


## **二维前缀和（2D Prefix Sum）的计算与应用**

### **1. 二维前缀和的定义**

对于一个二维矩阵 `matrix`，二维前缀和 `prefix_sum[i][j]` 代表从矩阵的左上角 `(0,0)` 到 `(i,j)` 这个子矩阵内所有元素的累加和：

prefix_sum[i][j]=∑x=0i∑y=0jmatrix[x][y]prefix\_sum[i][j] = \sum_{x=0}^{i} \sum_{y=0}^{j} matrix[x][y]prefix_sum[i][j]=x=0∑i​y=0∑j​matrix[x][y]

这意味着 `prefix_sum[i][j]` 代表矩阵 `matrix` **从左上角 (0,0) 到 (i,j) 形成的矩形区域的总和**。

---

### **2. 二维前缀和的计算公式**

要计算二维前缀和，我们可以使用以下递推公式：

prefix_sum[i][j]=matrix[i][j]+prefix_sum[i−1][j]+prefix_sum[i][j−1]−prefix_sum[i−1][j−1]prefix\_sum[i][j] = matrix[i][j] + prefix\_sum[i-1][j] + prefix\_sum[i][j-1] - prefix\_sum[i-1][j-1]prefix_sum[i][j]=matrix[i][j]+prefix_sum[i−1][j]+prefix_sum[i][j−1]−prefix_sum[i−1][j−1]

其中：

- `matrix[i][j]`：当前元素的值
- `prefix_sum[i-1][j]`：上一行的前缀和
- `prefix_sum[i][j-1]`：左边列的前缀和
- `prefix_sum[i-1][j-1]`：由于 `prefix_sum[i-1][j]` 和 `prefix_sum[i][j-1]` **都包含了 `prefix_sum[i-1][j-1]`**，所以需要减去一次以避免重复计算

---

### **3. 示例计算**

假设我们有以下 `4x4` 矩阵：

makefile

複製編輯

`matrix = 1  2  3  4 5  6  7  8 9 10 11 12 13 14 15 16`

计算 `prefix_sum`：

|`i,j`|`matrix[i][j]`|`prefix_sum[i-1][j]`|`prefix_sum[i][j-1]`|`prefix_sum[i-1][j-1]`|`prefix_sum[i][j]`|
|---|---|---|---|---|---|
|(0,0)|1|0|0|0|1|
|(0,1)|2|0|1|0|3|
|(0,2)|3|0|3|0|6|
|(0,3)|4|0|6|0|10|
|(1,0)|5|1|0|0|6|
|(1,1)|6|3|6|1|14|
|(1,2)|7|6|14|3|24|
|(1,3)|8|10|24|6|36|
|(2,0)|9|6|0|0|15|
|(2,1)|10|14|15|6|33|
|(2,2)|11|24|33|14|54|
|(2,3)|12|36|54|24|78|
|(3,0)|13|15|0|0|28|
|(3,1)|14|33|28|15|60|
|(3,2)|15|54|60|33|96|
|(3,3)|16|78|96|54|136|

最终 `prefix_sum` 结果：

複製編輯

`1   3   6   10 6   14  24  36 15  33  54  78 28  60  96  136`

---

### **4. 使用二维前缀和查询子矩阵的和**

如果我们想查询 **任意子矩阵 `(r1, c1)` 到 `(r2, c2)` 之间的元素和**，可以用：

sum(r1,c1,r2,c2)=prefix_sum[r2][c2]−prefix_sum[r1−1][c2]−prefix_sum[r2][c1−1]+prefix_sum[r1−1][c1−1]sum(r1, c1, r2, c2) = prefix\_sum[r2][c2] - prefix\_sum[r1-1][c2] - prefix\_sum[r2][c1-1] + prefix\_sum[r1-1][c1-1]sum(r1,c1,r2,c2)=prefix_sum[r2][c2]−prefix_sum[r1−1][c2]−prefix_sum[r2][c1−1]+prefix_sum[r1−1][c1−1]

#### **例子**

查询 `(1,1)` 到 `(2,2)` 这个子矩阵的和：

複製編輯

`6  7 10 11`

代入公式：

sum(1,1,2,2)=prefix_sum[2][2]−prefix_sum[0][2]−prefix_sum[2][0]+prefix_sum[0][0]sum(1,1,2,2) = prefix\_sum[2][2] - prefix\_sum[0][2] - prefix\_sum[2][0] + prefix\_sum[0][0]sum(1,1,2,2)=prefix_sum[2][2]−prefix_sum[0][2]−prefix_sum[2][0]+prefix_sum[0][0] =54−6−15+1=34= 54 - 6 - 15 + 1 = 34=54−6−15+1=34

---

### **5. 代码实现**

python

複製編輯

`def compute_prefix_sum(matrix):     rows, cols = len(matrix), len(matrix[0])     prefix_sum = [[0] * cols for _ in range(rows)]      for i in range(rows):         for j in range(cols):             prefix_sum[i][j] = matrix[i][j]             if i > 0:                 prefix_sum[i][j] += prefix_sum[i - 1][j]             if j > 0:                 prefix_sum[i][j] += prefix_sum[i][j - 1]             if i > 0 and j > 0:                 prefix_sum[i][j] -= prefix_sum[i - 1][j - 1]      return prefix_sum  def query_sum(prefix_sum, r1, c1, r2, c2):     result = prefix_sum[r2][c2]     if r1 > 0:         result -= prefix_sum[r1 - 1][c2]     if c1 > 0:         result -= prefix_sum[r2][c1 - 1]     if r1 > 0 and c1 > 0:         result += prefix_sum[r1 - 1][c1 - 1]     return result  # 測試 matrix = [     [1, 2, 3, 4],     [5, 6, 7, 8],     [9, 10, 11, 12],     [13, 14, 15, 16] ] prefix_sum = compute_prefix_sum(matrix)  # 查詢 (1,1) 到 (2,2) 的子矩陣和 print(query_sum(prefix_sum, 1, 1, 2, 2))  # 34`

---

### **6. 二维前缀和的应用**

1. **快速计算子矩阵和**：
    - 适用于 `K` 次查询矩阵和的问题，如 Lintcode 1315 **“Range Sum Query 2D”**。
2. **最大子矩阵和（Kadane’s Algorithm 变体）**：
    - Lintcode 944 **“最大子矩阵”**。
3. **前缀和优化 DP 问题**：
    - 如 **“最大正方形”** Lintcode 436。
4. **动态窗口处理问题**：
    - 如滑动窗口最大值。

---

### **7. 总结**

- **二维前缀和** 扩展了一维前缀和概念，可高效计算子矩阵和。
- **时间复杂度**：
    - **预处理**：`O(n*m)`
    - **查询子矩阵和**：`O(1)`
- **适用于**：多次区间查询、动态规划优化、最大子矩阵和问题等。

掌握 **二维前缀和** 可以大幅度提高矩阵查询相关问题的效率，是许多动态规划和数学优化问题的关键技巧！

  

O

搜尋