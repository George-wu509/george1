Lintcode 138
给定一个整数数组，找到和为 0 的子数组。你的代码应该返回满足要求的子数组的起始位置和结束位置

**样例 1:**
```python
"""
输入: [-3, 1, 2, -3, 4]
输出: [0,2] 或 [1,3]	
样例解释： 返回任意一段和为0的区间即可。
```
**样例 2:**
```python
"""
输入: [-3, 1, -4, 2, -3, 4]
输出: [1,5]
```


```python
    def subarray_sum(self, nums):
        prefix_hash = {0: -1}
        prefix_sum = 0
        for i, num in enumerate(nums):
            prefix_sum += num
            if prefix_sum in prefix_hash:
                return prefix_hash[prefix_sum] + 1, i
            prefix_hash[prefix_sum] = i
            
        return -1, -1
```
pass

nums =  [-3, 1, 2, -3, 4]  
解釋:  
step1: 用dict建立一個hash table, 另一個變數prefix_sum是前缀和.  
step2: 從id=0往右, 每個id計算prefix_sum, 並把每個prefix_sum加入hash table  
step3: 每一個step計算新的prefix_sum, 並check是否在hash table裡有跟當前prefix_sum一樣的值  
因為這代表新的id的prefix_sum(new)到以前存的prefix_sum(old)  
prefix_sum(new)-prefix_sum(old)=0  

解釋(II):
隨著for loop, 將新的num加上prefix_sum成新的prefix_sum, 並和之前的prefix_hash比較. 如果有相同的代表prefix_sum[i] - prefix[j] = 0 代表在nums[j+1:i]的和為0

nums = [1, 2, 3, 4]
id =   [0, 1, 2, 3]

prefix_sum = [0, 1, 3, 6, 10]
id         = [0, 1, 2, 3, 4 ]


nums[1]+nums[2] = 2+3 = 5
prefix[3] - prefix[1] = 5

# **LintCode 138: Subarray Sum（子数组和）**

---

## **题目描述**

给定一个整数数组 `nums`，找到一个 **连续的子数组**，使得 **该子数组的和为 `0`**，返回子数组的**起始索引和结束索引**。

#### **示例**

`输入： nums = [1, 2, -3, 3]  输出： (0, 2)`

**解释**

`子数组 [1, 2, -3] 的和为 0，对应索引范围 (0,2)。`

---

## **解法：前缀和 + 哈希表（双指针）**

### **核心思路**

1. **前缀和的概念**
    
    - 计算 `prefix_sum[i]`，表示 `nums[0]` 到 `nums[i]` 的累积和： 
    - 若 `prefix_sum[j] == prefix_sum[i]`（`j < i`），说明 `nums[j+1]` 到 `nums[i]` 的和为 `0`： 
2. **哈希表存储前缀和**
    
    - **键**：前缀和 `prefix_sum`。
    - **值**：该前缀和第一次出现的索引 `index`。
    - **若某个 `prefix_sum` 在哈希表中已存在**，说明存在子数组满足 `sum == 0`，返回索引 `(start, end)`。

---

### **代码解析**
```python
def subarray_sum(self, nums):
    prefix_hash = {0: -1}  # 记录前缀和，初始值表示空数组
    prefix_sum = 0  # 记录当前前缀和

    for i, num in enumerate(nums):
        prefix_sum += num  # 计算当前前缀和

        if prefix_sum in prefix_hash:  # 若前缀和已出现，找到子数组
            return prefix_hash[prefix_sum] + 1, i

        prefix_hash[prefix_sum] = i  # 存储当前前缀和的索引

    return -1, -1  # 无符合条件的子数组

```

---

## **逐步执行分析**

**输入**

`nums = [1, 2, -3, 3]`

---

### **初始化**

|变量|初始值|
|---|---|
|`prefix_hash = {0: -1}`|记录前缀和 `0` 出现于 `-1` 位置|
|`prefix_sum = 0`|当前前缀和|

---

### **遍历 `nums`**

|`i`|`num`|`prefix_sum` 计算|`prefix_hash` 查询|结果|
|---|---|---|---|---|
|0|1|`0 + 1 = 1`|`prefix_hash = {0: -1, 1: 0}`|无匹配|
|1|2|`1 + 2 = 3`|`prefix_hash = {0: -1, 1: 0, 3: 1}`|无匹配|
|2|-3|`3 + (-3) = 0`|`prefix_sum=0` 在 `prefix_hash` 中，找到 `(prefix_hash[0] + 1, i) = (0,2)`|**返回 (0, 2)**|

最终返回：

`(0, 2)`

---

## **时间与空间复杂度分析**

3. **时间复杂度**
    
    - **`O(n)`**：遍历 `nums` 一次，每次 `O(1)` 计算 `prefix_sum` 和 `prefix_hash`。
4. **空间复杂度**
    
    - **`O(n)`**：最坏情况下，哈希表 `prefix_hash` 需要存储 `O(n)` 个前缀和。

---

## **其他解法**

### **1. 暴力法 `O(n^2)`**

- **思路**
    - 枚举所有子数组 `nums[i:j]`，计算和 `sum(nums[i:j])`。
    - **时间复杂度 `O(n^2)`，不适合大数据。**
```python
def subarray_sum(nums):
    n = len(nums)
    
    for i in range(n):
        curr_sum = 0
        for j in range(i, n):
            curr_sum += nums[j]
            if curr_sum == 0:
                return i, j

    return -1, -1

```
---

## **方法比较**

|方法|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|
|**前缀和 + 哈希表（最佳解法）**|`O(n)`|`O(n)`|**适用于大规模数据**|
|暴力法|`O(n^2)`|`O(1)`|**适用于数据较小**|

🚀 **前缀和 `O(n)` 解法是最优，适用于大规模数据！**
