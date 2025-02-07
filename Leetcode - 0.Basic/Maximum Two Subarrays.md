Lintcode 42
给定一个整数数组，找出两个 _不重叠_ 子数组使得它们的和最大。  每个子数组的数字在数组中的位置应该是连续的。  返回最大的和。

**样例 1：**
输入：
```python
#nums = [1, 3, -1, 2, -1, 2]
```
输出：
```python
#7
```
解释：
最大的子数组为 [1, 3] 和 [2, -1, 2] 或者 [1, 3, -1, 2] 和 [2].  

**样例 2：**
输入：
```python
#nums = [5,4]
```
输出：
```python
#9
```
解释：
最大的子数组为 [5] 和 [4].


```python
import sys

class Solution:
    """
    @param: nums: A list of integers
    @return: An integer denotes the sum of max two non-overlapping subarrays
    """
    def max_two_sub_arrays(self, nums):
        n = len(nums)
        
        # 计算以i位置为结尾的前后缀最大连续和
        left_max = nums[:]
        right_max = nums[:]
        
        for i in range(1, n):
            left_max[i] = max(nums[i], left_max[i - 1] + nums[i])

        for i in range(n - 2, -1, -1):
            right_max[i] = max(nums[i], right_max[i + 1] + nums[i])
        
        # 计算前后缀部分最大连续和
        prefix_max = left_max[:]
        postfix_max = right_max[:]
    
        for i in range(1, n):
            prefix_max[i] = max(prefix_max[i], prefix_max[i - 1])
            
        for i in range(n - 2, -1, -1):
            postfix_max[i] = max(postfix_max[i], postfix_max[i + 1])
        
        result = -sys.maxsize
        for i in range(n - 1):
            result = max(result, prefix_max[i] + postfix_max[i + 1])
        
        return result
```
pass

### 解题思路
https://www.lintcode.com/problem/42/solution/32938
这题是最大子段和的升级版。我们只要在求最大子段和的基础上，计算出前后缀的最大子段和，就可以枚举分界点来计算结果。

### 代码思路

对于前缀的最大子段和，我们可以先求以`i`位置为结尾的最大子段和的值`leftMax[i]`。

‘leftMax[i][i−1]+nums[i][i])‘‘leftMax[i][i−1]+nums[i][i])‘

`max`中的两个参数分别代表延续前一个为结尾的最大字段和，或者当前的`nums[i]`成为一段新的子段的两种情况。

计算前缀最大子段和`prefixMax`，计算前缀最大值即可。

‘prefixMax[i][i],prefix[i−1])‘‘prefixMax[i][i],prefix[i−1])‘

后缀的值也同理进行计算。

最后枚举分界点，取最大值，最终的结果为：

‘max0n−2prefix[i][i+1]‘‘max0n−2​prefix[i][i+1]‘。

### 复杂度分析

设数组长度为`N`。

#### 时间复杂度

- 只需常数次地遍历数组，时间复杂度为`O(N)`。

#### 空间复杂度

- 需要常数个额外数组来记录当前位置结尾前后缀最大连


### **LintCode 42 - Maximum Two Subarrays**

這道題目要求 **找出兩個不相交的子數組，使其總和最大**，並返回該最大總和。

這是一道變種的 **最大子數組和（Maximum Subarray Sum）** 問題。相較於 **LintCode 41（Maximum Subarray）**，這題的額外限制是 **子數組不可重疊**，因此必須考慮將數組拆成兩部分，使得兩部分的最大和相加後最大化。

---

## **解法解析**

這裡使用 **前綴最大和 + 後綴最大和** 的策略來解決問題。

### **思路**

1. **計算左側最大子數組和 `left_max`**
    
    - 記錄 **從左到右遍歷時**，每個 `i` 位置 **結尾** 的最大子數組和。
    - 這與 **Kadane’s Algorithm**（最大子數組和演算法）類似。
2. **計算右側最大子數組和 `right_max`**
    
    - 記錄 **從右到左遍歷時**，每個 `i` 位置 **開頭** 的最大子數組和。
3. **計算 `prefix_max` 和 `postfix_max`**
    
    - `prefix_max[i]`：儲存 **從 `0` 到 `i` 位置的最大子數組和**，確保 `left_max` 中的最大值不會因為後續數值減少而消失。
    - `postfix_max[i]`：儲存 **從 `i` 到 `n-1` 位置的最大子數組和**，確保 `right_max` 中的最大值不會因為前序數值減少而消失。
4. **遍歷 `i = 0` 到 `n-2`，計算最大值**
    
    - `result = max(result, prefix_max[i] + postfix_max[i + 1])`
    - 這樣確保前後兩部分 **互不重疊**，並且其和最大。

---

## **具體舉例**

假設 `nums = [1, 3, -1, 2, -1, 2]`，我們的變數變化如下：

### **Step 1: 計算 `left_max`**

計算 **以 `i` 結尾的最大子數組和**：

|`i`|`nums[i]`|`left_max[i]`|
|---|---|---|
|0|1|1|
|1|3|4 (`1+3`)|
|2|-1|3 (`4-1`)|
|3|2|5 (`3+2`)|
|4|-1|4 (`5-1`)|
|5|2|6 (`4+2`)|
left_max = [1,4,3,5,4,6]

### **Step 2: 計算 `right_max`**

計算 **以 `i` 開頭的最大子數組和**：

| `i` | `nums[i]` | `right_max[i]` |
| --- | --------- | -------------- |
| 5   | 2         | 2              |
| 4   | -1        | 1 (`2-1`)      |
| 3   | 2         | 3 (`1+2`)      |
| 2   | -1        | 2 (`3-1`)      |
| 1   | 3         | 5 (`2+3`)      |
| 0   | 1         | 6 (`5+1`)      |
right_max = [6,5,2,3,1,2]

### **Step 3: 計算 `prefix_max`**

left_max = [1,4,3,5,4,6]
prefix_max = [1,4,4,5,5,6]

計算 **從 `0` 到 `i` 的最大子數組和**：

|`i`|`prefix_max[i]`|
|---|---|
|0|1|
|1|4|
|2|4|
|3|5|
|4|5|
|5|6|

### **Step 4: 計算 `postfix_max`**

right_max = [6,5,2,3,1,2]
postfix_max = [6,5,3,3,2,2]

計算 **從 `i` 到 `n-1` 的最大子數組和**：

|`i`|`postfix_max[i]`|
|---|---|
|5|2|
|4|2|
|3|4|
|2|4|
|1|6|
|0|7|

### **Step 5: 遍歷 `i = 0` 到 `n-2` 計算最大值**

|`i`|`prefix_max[i]`|`postfix_max[i+1]`|`prefix_max[i] + postfix_max[i+1]`|
|---|---|---|---|
|0|1|6|**7**|
|1|4|4|**8**|
|2|4|4|**8**|
|3|5|2|**7**|
|4|5|2|**7**|

最大結果為 **8**，對應的兩個不相交子數組為：

- **左側： `[1, 3]`，和為 `4`**
- **右側： `[2, -1, 2]`，和為 `4`**

---

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - `left_max` 計算一次 `O(n)`
    - `right_max` 計算一次 `O(n)`
    - `prefix_max` 計算一次 `O(n)`
    - `postfix_max` 計算一次 `O(n)`
    - `result` 遍歷一次 `O(n)`
    - **總時間複雜度為 `O(n)`**
- **空間複雜度：O(n)**
    
    - 需要存儲 `left_max`, `right_max`, `prefix_max`, `postfix_max`，所以額外使用 `O(n)` 的空間。

---

## **其他解法**

### **1. 動態規劃（DP + 分段）**

- **思路**：利用動態規劃分段計算兩個子數組和，並找出最佳組合。
- **時間複雜度**：O(n²)
- **空間複雜度**：O(n)

### **2. 分治法（Divide and Conquer）**

- **思路**：使用類似 **最大子數組和的分治解法**，每次將數組劃分為三部分：
    1. **完全在左側的子數組**
    2. **完全在右側的子數組**
    3. **跨越中點的兩個子數組**
- **時間複雜度**：O(n log n)
- **空間複雜度**：O(log n)

### **3. 暴力法（Brute Force）**

- **思路**：枚舉所有可能的兩個不重疊子數組，計算其總和並取最大值。
- **時間複雜度**：O(n³)
- **空間複雜度**：O(1)

---

## **總結**

- **最優解法**：前綴和 + 後綴和（本題使用的方法），時間複雜度 `O(n)`，空間複雜度 `O(n)`。
- **適用於大數據**：這種解法比 `O(n²)` 和 `O(n³)` 的方法更高效，能夠在大數據集上快速執行。
- **關鍵思路**：透過 **Kadane's Algorithm** 計算 `left_max` 和 `right_max`，並利用 `prefix_max` 和 `postfix_max` 確保選擇的是最大值。