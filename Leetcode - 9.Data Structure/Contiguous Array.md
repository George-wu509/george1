Lintcode 994
给一个二进制数组，找到 0 和 1 数量相等的子数组的最大长度

**样例 1:**
```python
"""
输入: [0,1]
输出: 2
解释: [0, 1] 是具有相等数量的 0 和 1 的最长子数组。
```
**样例 2:**
```python
"""
输入: [0,1,0]
输出: 2
解释: [0, 1] (或者 [1, 0]) 是具有相等数量 0 和 1 的最长子数组。
```


```python
def find_max_length(self, nums: List[int]) -> int:
	map = {0:-1}
	flag = 0
	ans = 0
	for index, num in enumerate(nums):
		if num == 0:
			flag -= 1
		else:
			flag += 1
		if flag in map:
			ans = max(ans, index - map[flag])
		else:
			map[flag] = index
	return ans
```
pass

## **LintCode 994: Contiguous Array（连续数组）**

---

### **题目描述**

给定一个只包含 **0 和 1** 的数组 `nums`，求出 **最长的连续子数组**，使得该子数组中 `0` 和 `1` 的数量相等。

#### **示例**

`输入： nums = [0,1,0,1]  输出： 4`

**解释**

`最长的子数组为 [0,1,0,1]，长度为 4。`

---

## **解法：前缀和 + 哈希表（双指针）**

### **核心思路**

1. **利用前缀和**
    
    - 将 `0` 视为 `-1`，则求**最长的子数组和为 0** 的问题，即： ∑i→jnums[i]=0\sum_{i \to j} nums[i] = 0i→j∑​nums[i]=0
    - 维护一个 `flag` 变量：
        - 遇到 `1` 时，`flag +1`
        - 遇到 `0` 时，`flag -1`
2. **哈希表存储前缀和**
    
    - 维护 `map = {0: -1}`，用于存储**首次出现 `flag` 的索引**。
    - 若 `flag` **之前已出现**，说明 `map[flag]` 到 `index` 之间的子数组和为 `0`，计算长度 `index - map[flag]`。
    - 若 `flag` **未出现**，将当前 `index` 存入 `map`。

---

### **代码解析**
```python
def find_max_length(self, nums: List[int]) -> int:
    map = {0: -1}  # 记录前缀和首次出现的位置
    flag = 0  # 记录 0 和 1 变换的前缀和
    ans = 0  # 记录最长子数组长度

    for index, num in enumerate(nums):
        if num == 0:
            flag -= 1  # 0 视为 -1
        else:
            flag += 1  # 1 视为 +1

        if flag in map:
            ans = max(ans, index - map[flag])  # 找到相同前缀和，计算最大长度
        else:
            map[flag] = index  # 记录该前缀和首次出现的位置

    return ans

```
---

## **逐步执行分析**

**输入**

`nums = [0, 1, 0, 1]`

### **初始化**

|变量|初始值|
|---|---|
|`map`|`{0: -1}`|
|`flag`|`0`|
|`ans`|`0`|

### **遍历 `nums`**

|`index`|`num`|`flag` 更新|`map` 查询|`ans` 更新|`map` 记录|
|---|---|---|---|---|---|
|0|0|`flag = -1`|**未找到**|`ans = 0`|`{-1: 0}`|
|1|1|`flag = 0`|**找到 `0`**，`index - map[0] = 1 - (-1) = 2`|`ans = 2`|`{0: -1, -1: 0}`|
|2|0|`flag = -1`|**找到 `-1`**，`index - map[-1] = 2 - 0 = 2`|`ans = 2`|`{0: -1, -1: 0}`|
|3|1|`flag = 0`|**找到 `0`**，`index - map[0] = 3 - (-1) = 4`|`ans = 4`|`{0: -1, -1: 0}`|

最终 `ans = 4`。

---

## **时间与空间复杂度分析**

1. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次查询和更新 `map` 是 **`O(1)`**，所以总复杂度是 `O(n)`。
2. **空间复杂度**
    
    - **`O(n)`**：`map` 最多存储 `n+1` 个不同的 `flag` 值，因此空间复杂度 `O(n)`。

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- 枚举所有子数组，检查 `0` 和 `1` 是否相等。
- **时间复杂度 `O(n^2)`**，不适合大数据。

```python
def find_max_length(nums):
    ans = 0
    for i in range(len(nums)):
        count_0 = count_1 = 0
        for j in range(i, len(nums)):
            if nums[j] == 0:
                count_0 += 1
            else:
                count_1 += 1
            if count_0 == count_1:
                ans = max(ans, j - i + 1)
    return ans

```
---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 哈希表（最佳解法）**|`O(n)`|`O(n)`|**大数据最优**|
|暴力法|`O(n^2)`|`O(1)`|**小规模数据**|