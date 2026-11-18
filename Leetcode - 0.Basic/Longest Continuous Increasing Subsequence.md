Lintcode 397
给定一个整数数组（下标从 `0` 到 `n-1`， `n` 表示整个数组的规模），请找出该数组中的最长上升连续子序列。（最长上升连续子序列可以定义为从右到左或从左到右的序列。）

**样例 1：**
```python
"""
输入：[5, 4, 2, 1, 3]
输出：4
解释：
给定 [5, 4, 2, 1, 3]，其最长上升连续子序列（LICS）为 [5, 4, 2, 1]，返回 4。
```
**样例 2：**
```python
"""
输入：[1, 5, 2, 3, 4]
输出：3
解释：
给定 [1, 5, 2, 3, 4]，其最长上升连续子序列（LICS）为 [2, 3, 4]，返回 3。
```

```python
    def longest_increasing_continuous_subsequence(self, nums):
        if not nums:
            return 0
        longest, incr, desc = 1, 1, 1
        for i in range(1, len(nums)):
            if nums[i] > nums[i - 1]:
                incr += 1
                desc = 1
            elif nums[i] < nums[i - 1]:
                incr = 1
                desc += 1
            else:
                incr = 1
                desc = 1
            longest = max(longest, max(incr, desc))
            
        return longest
```
pass
nums = [1,5,2,3,4]
解釋: 
step1: 從頭到尾遍歷nums, 並新增incr跟desc. 
step2: 如果有連續增加的譬如2->3->4, 則incr連續加1. 如果變成減少則incr變成1


## **LintCode 367：最长上升连续子序列**

---

### **题目描述**

给定一个整数数组 `a`，找到 **最长上升或下降的连续子序列** 的长度。

#### **示例**

`输入: a = [5, 1, 2, 3, 4] 输出: 4`

**解释**：

- **最长上升连续子序列** 是 `[1, 2, 3, 4]`，长度为 `4`。
- **最长下降连续子序列** 是 `[5, 1]`，长度为 `2`。
- 返回 **4**。

---

## **解法：双指针**

### **核心思路**

1. **维护两个变量**
    
    - `incr`：当前**上升**的连续子序列长度。
    - `desc`：当前**下降**的连续子序列长度。
2. **遍历数组**
    
    - **若 `a[i] > a[i-1]`**：上升序列继续增长，`incr += 1`，但 `desc` 归 1。
    - **若 `a[i] < a[i-1]`**：下降序列继续增长，`desc += 1`，但 `incr` 归 1。
    - **若 `a[i] == a[i-1]`**：既不是上升也不是下降，都归 1。
3. **更新最长长度**
    
    - 每次更新 `longest = max(longest, max(incr, desc))`。

---

## **代码解析**
```python
def longest_increasing_continuous_subsequence(self, a):
    if not a:
        return 0

    longest = 1  # 记录最长子序列长度
    incr = 1  # 记录当前上升连续子序列长度
    desc = 1  # 记录当前下降连续子序列长度

    for i in range(1, len(a)):
        if a[i] > a[i - 1]:  # 上升情况
            incr += 1
            desc = 1  # 下降重置
        elif a[i] < a[i - 1]:  # 下降情况
            desc += 1
            incr = 1  # 上升重置
        else:  # 相等情况
            incr = 1
            desc = 1

        longest = max(longest, max(incr, desc))  # 更新最长长度

    return longest

```

---

## **逐步执行分析**

**输入**

python

複製編輯

`a = [5, 1, 2, 3, 4]`

|`i`|`a[i]`|`a[i-1]`|`incr`|`desc`|`longest`|解释|
|---|---|---|---|---|---|---|
|0|5|-|1|1|1|初始值|
|1|1|5|1|2|2|下降 `[5,1]`|
|2|2|1|2|1|2|上升 `[1,2]`|
|3|3|2|3|1|3|上升 `[1,2,3]`|
|4|4|3|4|1|4|上升 `[1,2,3,4]`|

最终返回 `4`。

---

## **时间与空间复杂度分析**

4. **时间复杂度**
    
    - 遍历数组一次，时间复杂度为 **`O(n)`**。
5. **空间复杂度**
    
    - 只使用了几个额外的变量，空间复杂度为 **`O(1)`**。

---

## **其他解法**

### **1. 动态规划（DP）**

- 定义 `dp[i]` 为**以 `a[i]` 结尾**的最长上升（或下降）连续子序列长度。
- 递推关系：
    - `dp_inc[i] = dp_inc[i-1] + 1 if a[i] > a[i-1] else 1`
    - `dp_desc[i] = dp_desc[i-1] + 1 if a[i] < a[i-1] else 1`
- **时间复杂度 `O(n)`，空间复杂度 `O(n)`（可优化为 `O(1)`)**。

---

### **2. 双指针（起点记录）**

- 维护 `start` 指针记录当前上升或下降的起点：
    - **若 `a[i] > a[i-1]`**：继续上升。
    - **若 `a[i] < a[i-1]`**：继续下降。
    - **否则，更新 `start` 位置**。
- 计算 `i - start + 1` 并更新最大长度。

**代码**
```python
def longest_increasing_continuous_subsequence(self, a):
    if not a:
        return 0

    max_length = 1
    start = 0  # 记录当前区间起点
    for i in range(1, len(a)):
        if a[i] == a[i - 1]:
            start = i  # 重新计算
        max_length = max(max_length, i - start + 1)

    return max_length

```
- **时间复杂度 `O(n)`，空间复杂度 `O(1)`**。

---

## **总结**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**双指针（最佳解法）**|`O(n)`|`O(1)`|最优|
|动态规划|`O(n)`|`O(n)`|可优化为 `O(1)`|
|双指针（起点记录）|`O(n)`|`O(1)`|简洁，适用于一般情况|

🚀 **双指针法是最优解，时间 `O(n)`，空间 `O(1)`，适合大规模数据！**