Lintcode 57
给出一个有 `n` 个整数的数组 `S`，在 `S` 中找到三个整数 `a`, `b`, `c`，找到所有使得 `a + b + c = 0` 的三元组。

Examples
```python
样例 1：
输入：
numbers = [2,7,11,15]
输出：
[]
解释：
找不到三元组使得三个数和为0。

样例 2：
输入：
numbers = [-1,0,1,2,-1,-4]
输出：
[[-1, 0, 1],[-1, -1, 2]]
解释：
[-1, 0, 1]和[-1, -1, 2]是符合条件的2个三元组。
```

```python
class Solution:
    """
    @param numbers: Give an array numbers of n integer
    @return: Find all unique triplets in the array which gives the sum of zero.
    """
    def threeSum(self, nums):
        nums = sorted(nums)
        
        results = []
        for i in range(len(nums)):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            self.find_two_sum(nums, i + 1, len(nums) - 1, -nums[i], results)
            
        return results

    def find_two_sum(self, nums, left, right, target, results):
        last_pair = None
        while left < right:
            if nums[left] + nums[right] == target:
                if (nums[left], nums[right]) != last_pair:
                    results.append([-target, nums[left], nums[right]])
                last_pair = (nums[left], nums[right])
                right -= 1
                left += 1
            elif nums[left] + nums[right] > target:
                right -= 1
            else:
                left += 1
```
pass


# **LintCode 57: 3Sum（三数之和）**

---

## **问题描述**

给定一个整数数组 `nums`，找出所有 **不重复** 的 **三元组 `(a, b, c)`**，使得：

a+b+c=0a + b + c = 0a+b+c=0

**注意：**

- **不能包含重复解**。
- **三元组 `(a, b, c)` 必须按升序排列**。

---

## **解法：排序 + 双指针**

### **核心思路**

1. **先对数组进行排序**
    
    - 方便去重处理，保证三元组按**升序排列**。
2. **固定 `nums[i]`，用双指针 `left` 和 `right` 找 `b + c = -a`**
    
    - `i` 遍历 `nums`，每次固定 `nums[i]` 作为 `a`。
    - `left` 指向 `i+1`，`right` 指向 `len(nums)-1`，双指针寻找 `b + c = -a`。
3. **跳过重复元素**
    
    - **`nums[i] == nums[i-1]` 时跳过**，避免 `a` 重复。
    - **`(nums[left], nums[right]) == last_pair` 时跳过**，避免 `(b, c)` 重复。

---

## **执行过程**

### **变量表**

|变量|说明|
|---|---|
|`nums`|经过排序的数组|
|`i`|固定当前 `a` 的索引|
|`left`|指向 `b` 的左指针|
|`right`|指向 `c` 的右指针|
|`target`|需要找到的 `b + c` 的和|
|`results`|存储满足条件的三元组|

---

### **Step 1: 先排序**

假设输入：

python

複製編輯

`nums = [-1, 0, 1, 2, -1, -4]`

排序后：

ini

複製編輯

`nums = [-4, -1, -1, 0, 1, 2]`

---

### **Step 2: 使用三重循环 + 双指针**

#### **第一轮：固定 `i = 0`（`a = -4`），寻找 `b + c = 4`**

|变量|值|
|---|---|
|`i`|0|
|`left`|1|
|`right`|5|
|`target`|4|

- `nums[1] + nums[5] = -1 + 2 = 1`，小于 `4`，右移 `left = 2`。
- `nums[2] + nums[5] = -1 + 2 = 1`，小于 `4`，右移 `left = 3`。
- `nums[3] + nums[5] = 0 + 2 = 2`，小于 `4`，右移 `left = 4`。
- `nums[4] + nums[5] = 1 + 2 = 3`，小于 `4`，右移 `left = 5`，**结束循环**。

---

#### **第二轮：固定 `i = 1`（`a = -1`），寻找 `b + c = 1`**

|变量|值|
|---|---|
|`i`|1|
|`left`|2|
|`right`|5|
|`target`|1|

- `nums[2] + nums[5] = -1 + 2 = 1`，**找到解 `[-1, -1, 2]`**，记录到 `results`。
- `nums[3] + nums[4] = 0 + 1 = 1`，**找到解 `[-1, 0, 1]`**，记录到 `results`。

---

#### **第三轮：固定 `i = 2`（`a = -1`），跳过**

- `nums[2] == nums[1]`，**跳过，避免重复解**。

---

#### **第四轮：固定 `i = 3`（`a = 0`），寻找 `b + c = 0`**

- `nums[4] + nums[5] = 1 + 2 = 3`，大于 `0`，左移 `right = 4`。
- **结束循环**。

---

### **最终结果**

lua

複製編輯

`results = [[-1, -1, 2], [-1, 0, 1]]`

返回：

python

複製編輯

`[[-1, -1, 2], [-1, 0, 1]]`

---

## **时间与空间复杂度分析**

### **时间复杂度**

|操作|复杂度|说明|
|---|---|---|
|**排序 `nums`**|`O(n log n)`|需要排序数组|
|**三重循环遍历 `i, left, right`**|`O(n^2)`|`O(n)` 查找 `b + c`，`O(n)` 固定 `a`|
|**总复杂度**|`O(n^2)`|由于排序占主导，最终复杂度 `O(n^2)`|

### **空间复杂度**

- 仅使用常数额外空间，**`O(1)`**（不计 `results` 输出）。

---

## **其他解法**

### **1. 哈希表（O(n^2)）**

- **思路**
    - 固定 `a = nums[i]`，用哈希表存储 `target - nums[j]`，查找 `c`。
- **时间复杂度**
    - `O(n log n) + O(n^2) = O(n^2)`。

### **2. 递归回溯（O(n^3)）**

- **思路**
    - 递归选择 `a, b, c`，找到和为 `0` 的组合。
- **时间复杂度**
    - `O(n^3)`，适用于小规模数据。

---

## **方法比较**

|方法|思路|时间复杂度|空间复杂度|适用情况|
|---|---|---|---|---|
|**双指针（当前解法）**|**排序 + 固定 `a` + 双指针找 `b, c`**|`O(n^2)`|`O(1)`|**最优解，适用于大数据**|
|**哈希表**|**固定 `a`，哈希表存 `b` 查找 `c`**|`O(n^2)`|`O(n)`|**适用于 `b + c` 变化快的情况**|
|**递归回溯**|**暴力搜索所有三元组**|`O(n^3)`|`O(1)`|**适用于小规模数据**|

---

## **总结**

- **最优解** ✅ **双指针 `O(n^2)`**
- **如果 `b + c` 变化快，可用 `O(n^2)` 哈希表**
- **如果 `n` 适中，可用 `O(n^3)` 递归回溯**

🚀 **"双指针" 是最优解，适用于大规模数据！**