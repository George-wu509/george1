Lintcode 139
给定一个整数数组，找到一个和最接近于零的子数组。返回满足要求的子数组的起始位置和结束位置。你只需要考虑**其中的一种可能性**即可，具体可以参考样例。

**样例1**
```python
输入: 
[-3,1,1,-3,5] 
输出: 
[0,2]
解释: 返回 [0,2], [1,3], [1,1], [2,2], [0,4] 中的任意一个均可。
```
**样例2**
```python
输入: 
[2147483647]
输出: 
[0,0]
```

```python
    def subarray_sum_closest(self, nums):
        prefix_sum = [(0, -1)]
        for i, num in enumerate(nums):
            prefix_sum.append((prefix_sum[-1][0] + num, i))
            
        prefix_sum.sort()
        
        closest, answer = sys.maxsize, []
        for i in range(1, len(prefix_sum)):
            if closest > prefix_sum[i][0] - prefix_sum[i - 1][0]:
                closest = prefix_sum[i][0] - prefix_sum[i - 1][0]
                left = min(prefix_sum[i - 1][1], prefix_sum[i][1]) + 1
                right = max(prefix_sum[i - 1][1], prefix_sum[i][1])
                answer = [left, right]
        
        return answer
```
pass

nums =  [-3, 1, 1, -3, 5]
解釋:
step1. 從id=0遍歷到尾然後將前缀和跟id都存起來
prefix_sum = [(0,-1),(-3,0),(-2,1),(-1,2),(-4,3),(1,4)]
step2. 排序prefix_sum
prefix_sum = [(-4,3),(-3,0),(-2,1),(-1,2),(0,-1),(1,4)]
step3. 從id=0遍歷到尾計算排序過的prefix_sum, 比相鄰的prefix_sum. 譬如:
 prefix_sum[3]- prefix_sum[0] 等同於從id1,2,3的和  1+1+-3=-1 


# **LintCode 139: Subarray Sum Closest（最接近零的子数组和）**

---

## **题目描述**

给定一个整数数组 `nums`，找到**和最接近 0** 的子数组，返回其**起始索引和结束索引**。

#### **示例**

`输入： nums = [-3, 1, 1, -3, 5]  输出： (1, 3)`

**解释**

`子数组 [1,1,-3] 的和为 -1，是最接近 0 的子数组。`

---

## **解法：前缀和 + 排序（双指针）**

### **核心思路**

1. **前缀和**
    
    - 计算 `prefix_sum[i]`，表示 `nums[0]` 到 `nums[i]` 的累积和： 
    - **如果某两个 `prefix_sum` 非常接近**，说明**它们之间的子数组和接近 `0`**： 
2. **排序 + 双指针**
    
    - 将所有 `(prefix_sum, index)` 排序。
    - **相邻的 `prefix_sum[i]` 之差越小，表示 `nums[l:r]` 的和越接近 0**。
    - 遍历排序后的 `prefix_sum`，找到**最小的差值**，返回索引。

---

### **代码解析**
```python
import sys

def subarray_sum_closest(self, nums):
    prefix_sum = [(0, -1)]  # 记录 (前缀和, 索引)，初始化前缀和0出现于-1位置

    # 计算前缀和
    for i, num in enumerate(nums):
        prefix_sum.append((prefix_sum[-1][0] + num, i))

    # 按前缀和排序
    prefix_sum.sort()

    closest, answer = sys.maxsize, []

    # 找最小的前缀和差值
    for i in range(1, len(prefix_sum)):
        diff = prefix_sum[i][0] - prefix_sum[i - 1][0]
        if diff < closest:
            closest = diff
            left = min(prefix_sum[i - 1][1], prefix_sum[i][1]) + 1
            right = max(prefix_sum[i - 1][1], prefix_sum[i][1])
            answer = [left, right]

    return answer

```

---

## **逐步执行分析**

**输入**

`nums = [-3, 1, 1, -3, 5]`

---

### **步骤 1：计算前缀和**

|`i`|`num`|`prefix_sum[i]` 计算|`prefix_sum` 结果|
|---|---|---|---|
|-1|0|`0`|`[(0, -1)]`|
|0|-3|`0 + (-3) = -3`|`[(-3, 0)]`|
|1|1|`-3 + 1 = -2`|`[(-2, 1)]`|
|2|1|`-2 + 1 = -1`|`[(-1, 2)]`|
|3|-3|`-1 + (-3) = -4`|`[(-4, 3)]`|
|4|5|`-4 + 5 = 1`|`[(1, 4)]`|

最终：

`prefix_sum = [(0, -1), (-3, 0), (-2, 1), (-1, 2), (-4, 3), (1, 4)]`

---

### **步骤 2：排序 `prefix_sum`**

排序后：

`prefix_sum = [(-4, 3), (-3, 0), (-2, 1), (-1, 2), (0, -1), (1, 4)]`

---

### **步骤 3：遍历相邻差值**

|`i`|`prefix_sum[i-1]`|`prefix_sum[i]`|`diff` 计算|`closest` 更新|`answer`|
|---|---|---|---|---|---|
|1|`(-4,3)`|`(-3,0)`|`|-3 - (-4)|= 1`|
|2|`(-3,0)`|`(-2,1)`|`|-2 - (-3)|= 1`|
|3|`(-2,1)`|`(-1,2)`|`|-1 - (-2)|= 1`|
|4|`(-1,2)`|`(0,-1)`|`|0 - (-1)|= 1`|
|5|`(0,-1)`|`(1,4)`|`|1 - 0|= 1`|

最终返回：

`[1, 3]`

---

## **时间与空间复杂度分析**

3. **时间复杂度**
    
    - **计算前缀和 `O(n)`**
    - **排序 `O(n log n)`**
    - **遍历 `O(n)`**
    - **总复杂度 `O(n log n)`**
4. **空间复杂度**
    
    - **`O(n)`**（存储 `prefix_sum`）

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- **思路**
    - 枚举所有子数组 `nums[i:j]`，计算 `sum(nums[i:j])` 并找最接近 `0` 的。
    - **缺点**：时间复杂度过高。
```python
def subarray_sum_closest(nums):
    n = len(nums)
    closest = float('inf')
    answer = []

    for i in range(n):
        curr_sum = 0
        for j in range(i, n):
            curr_sum += nums[j]
            if abs(curr_sum) < closest:
                closest = abs(curr_sum)
                answer = [i, j]

    return answer

```
- **时间复杂度 `O(n^2)`**
- **适用于数据较小**

---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 排序（最佳解法）**|`O(n log n)`|`O(n)`|**大数据最优**|
|暴力法|`O(n^2)`|`O(1)`|**数据较小时可用**|

🚀 **前缀和 + 排序 `O(n log n)` 是最优解，适用于大规模数据！**