Lintcode 402
给定一个整数数组，请找出一个连续子数组，使得该子数组的和最大。输出答案时，请分别返回第一个数字和最后一个数字的下标。（如果存在多个答案，请返回字典序最小的）
**样例 1:**
```python
输入: [-3, 1, 3, -3, 4]
输出: [1, 4]
```
**样例 2:**
```python
输入: [0, 1, 0, 1]
输出: [0, 3]
解释: 字典序最小.
```


```python
    def continuous_subarray_sum(self, a):
        n = len(a)
        if n == 0: return []
        
        import sys
        minsum, presum = 0, 0
        maxsum = -sys.maxsize
        minindex = -1
        l, r = -1 , -1
        for i in range(n):
            presum = presum + a[i]
            if presum - minsum > maxsum:
                maxsum = presum - minsum
                l, r = minindex + 1, i
            if presum < minsum:
                minsum = presum
                minindex = i
        
        return [l, r]
```
pass

## **LintCode 402: Continuous Subarray Sum（连续子数组和）**

---

### **题目描述**
给定一个整数数组 `a`，找到 **和最大** 的 **连续子数组**，返回其起始和结束索引。
#### **示例**

`输入: a = [-3, 1, 3, -3, 4]  输出: [1, 4]`

**解释**
``最大子数组是 [1, 3, -3, 4]，和为 `5`，索引范围 `[1, 4]`。``

---

## **解法：前缀和 + 双指针**

### **核心思路**

1. **利用前缀和 `presum[i]`**
    
    - `presum[i]` 表示从 `a[0]` 到 `a[i]` 的累积和。
    - 子数组 `a[l:r]` 的和等于：`presum[r] - presum[l-1]`
    - **目标**：找到 `l` 和 `r` 使得 `presum[r] - presum[l-1]` 最大。
2. **维护 `minsum` 记录最小前缀和**
    
    - `minsum`：**遍历过程中最小的 `presum`**。
    - `minindex`：`minsum` 对应的索引。
3. **双指针 `l, r`**
    
    - `l` 记录最大子数组的起始索引。
    - `r` 记录最大子数组的结束索引。

---

### **代码解析**
```python
def continuous_subarray_sum(self, a: List[int]) -> List[int]:
    n = len(a)
    if n == 0: return []

    import sys
    minsum, presum = 0, 0  # minsum: 记录最小的前缀和
    maxsum = -sys.maxsize  # 记录最大子数组和
    minindex = -1  # minsum 对应的索引
    l, r = -1, -1  # 记录最大子数组的索引范围

    for i in range(n):
        presum += a[i]  # 计算当前前缀和

        # 检查是否当前区间 a[minindex+1:i] 和最大
        if presum - minsum > maxsum:
            maxsum = presum - minsum
            l, r = minindex + 1, i

        # 更新最小前缀和
        if presum < minsum:
            minsum = presum
            minindex = i

    return [l, r]

```

---

## **逐步执行分析**

**输入**

`a = [-3, 1, 3, -3, 4]`

### **初始化**

|变量|初始值|
|---|---|
|`minsum`|`0`|
|`presum`|`0`|
|`maxsum`|`-∞`|
|`minindex`|`-1`|
|`l`|`-1`|
|`r`|`-1`|

---

### **遍历 `a`**

|`i`|`a[i]`|`presum` 变化|`minsum` 变化|`maxsum` 变化|`l`|`r`|`minindex`|
|---|---|---|---|---|---|---|---|
|0|-3|`-3`|`-3`|`-3`|`0`|`0`|`0`|
|1|1|`-2`|`-3`|`1`|`1`|`1`|`0`|
|2|3|`1`|`-3`|`4`|`1`|`2`|`0`|
|3|-3|`-2`|`-3`|`4`|`1`|`2`|`0`|
|4|4|`2`|`-3`|`5`|`1`|`4`|`0`|

最终返回：

`[1, 4]`

---

## **时间与空间复杂度分析**

4. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次 `O(1)` 计算 `presum` 和 `maxsum`。
5. **空间复杂度**
    
    - **`O(1)`**：只使用了常数个变量。

---

## **其他解法**

### **1. Kadane’s Algorithm（O(n)）**

- **思路**
    - 维护当前最大子数组 `sum`。
    - 如果 `sum < 0`，重新开始计算。
    - 适用于 **返回最大和，但无法直接返回索引**。
    
```python
def continuous_subarray_sum(a):
    ans = -float('inf')
    sum = 0
    start, end = 0, -1
    result = [-1, -1]

    for i in range(len(a)):
        if sum < 0:
            sum = a[i]
            start = i
        else:
            sum += a[i]

        if sum > ans:
            ans = sum
            result = [start, i]

    return result

```

- **时间复杂度 `O(n)`，空间复杂度 `O(1)`。**

---

### **2. 暴力法 `O(n^2)`**

- **思路**
    - 枚举所有子数组 `a[i:j]`，计算 `sum(a[i:j])`。
    - **缺点**：时间复杂度过高。

python
```python
def continuous_subarray_sum(nums):
    n = len(nums)
    ans = -float('inf')
    result = [-1, -1]

    for i in range(n):
        sum = 0
        for j in range(i, n):
            sum += nums[j]
            if sum > ans:
                ans = sum
                result = [i, j]

    return result

```

- **时间复杂度 `O(n^2)`，空间复杂度 `O(1)`。**

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 双指针（最佳解法）**|`O(n)`|`O(1)`|**大数据最优**|
|Kadane's Algorithm|`O(n)`|`O(1)`|**返回最大和，不返回索引**|
|暴力法|`O(n^2)`|`O(1)`|**适用于小数据**|

🚀 **前缀和 `O(n)` 解法是最优，适用于大规模数据！**