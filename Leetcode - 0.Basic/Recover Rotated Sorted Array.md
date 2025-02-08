Lintcode 39
给定一个**旋转**排序数组，在原地恢复其排序。（升序）

**样例 1：**
输入：
```python
"""```python
数组 = [4,5,1,2,3]
```
输出：
```python
"""
[1,2,3,4,5]
```
解释：
恢复旋转排序数组。

**样例 2：**
输入：
```python
"""
数组 = [6,8,9,1,2]
```
输出：
```python
"""
[1,2,6,8,9]
```
解释：
恢复旋转排序数组。


```python
class Solution:
    """
    @param nums: An integer array
    @return: nothing
    """
    def recover_rotated_sorted_array(self, nums):
        split_position = self.find_split(nums)
        if split_position == len(nums)-1:
            return 
        
        self.swap(nums, 0, split_position)
        self.swap(nums, split_position, len(nums))
        
        nums.reverse()
        return 
        
    def find_split(self, nums):
        # DO NOT use binary search!
        # Binary Search does not work on this prob
        if nums is None or len(nums) < 2:
            return 0
        
        for i in range(1,len(nums)):
            if nums[i] < nums[i-1]:
                return i 
        # return i = len()-1 if it's already a sorted array 
        return i 
            
    def swap(self, nums, start, end):
        if start == end:
            return nums 
        
        left, right = start, end -1  
        while left < right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1 
            right -= 1
```
pass

# **LintCode 39 - Recover Rotated Sorted Array（恢复旋转排序数组）**

## **题目解析**

给定一个 **升序排序的数组 `nums`**，但 **经过某个未知的点进行了旋转**，要求 **恢复原来的升序排序**。  
**示例：**

`输入: [4, 5, 6, 1, 2, 3] 输出: [1, 2, 3, 4, 5, 6]`

### **关键点**

1. 数组是**部分旋转**的，即 `[4, 5, 6, 1, 2, 3]` 由 `[1, 2, 3, 4, 5, 6]` 旋转得到。
2. **不能使用额外空间**，需要 **原地恢复** 数组。
3. **二分查找不可行**，因为数组并非标准升序，而是**局部升序**。

---

## **解法解析**

### **思路**

要恢复 `nums`，我们需要 **找到旋转点并进行恢复**：

4. **找到旋转点（split position）**
    
    - 旋转点是 **数组中第一个递减的位置**，即 `nums[i] > nums[i+1]`。
    - 例如 `[4, 5, 6, 1, 2, 3]` 中，`nums[2] > nums[3]`，所以旋转点 `split = 3`。
5. **使用三次翻转（Reverse）恢复数组**
    
    - **第一步：翻转 `nums[0:split]`**
    - **第二步：翻转 `nums[split:]`**
    - **第三步：翻转整个数组**

---

## **解法步骤**

6. **找到旋转点 `split_position`**
    
    - 遍历数组，找到第一个 `nums[i] > nums[i+1]`，返回 `i+1` 作为 `split_position`。
7. **三次翻转恢复排序**
    
    - 翻转 `nums[0:split]`，即前半部分。
    - 翻转 `nums[split:]`，即后半部分。
    - 翻转整个数组。

---

## **具体举例**

**示例输入**：

python

複製編輯

`nums = [4, 5, 6, 1, 2, 3]`

### **Step 1: 找到旋转点**

遍历 `nums`，发现：

複製編輯

`nums[2] = 6 > nums[3] = 1`

所以 **旋转点 `split_position = 3`**。

---

### **Step 2: 三次翻转**

8. **翻转 `nums[0:3]`**（前半部分 `[4,5,6]`）：
    
    csharp
    
    複製編輯
    
    `[6, 5, 4, 1, 2, 3]`
    
9. **翻转 `nums[3:]`**（后半部分 `[1,2,3]`）：
    
    csharp
    
    複製編輯
    
    `[6, 5, 4, 3, 2, 1]`
    
10. **翻转整个数组**：
    
    csharp
    
    複製編輯
    
    `[1, 2, 3, 4, 5, 6]`
    

---

## **时间与空间复杂度分析**

### **时间复杂度**

11. **寻找旋转点：O(n)**
12. **三次翻转：O(n)**
    - 每次翻转最多 `O(n)`。
    - 总共 **3 次翻转**，仍为 **O(n)**。
13. **总时间复杂度：O(n)**

### **空间复杂度**

- **原地修改**，仅使用 **常数额外空间 `O(1)`**。

---

## **其他解法**

### **1. 直接 `sort()` 排序**

- **思路**：
    - 直接使用 `nums.sort()` 进行排序。
- **时间复杂度**：O(n log n)（不如 O(n) 方案）
- **空间复杂度**：O(1)

### **2. 遍历插入法**

- **思路**：
    - 找到 `split_position` 后，将前半部分 **移动到数组末尾**。
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)（额外存储）

---

## **LintCode 相关题目**

|**题号**|**题目名称**|**难度**|**核心技术**|
|---|---|---|---|
|**LintCode 39**|Recover Rotated Sorted Array|⭐⭐⭐|**旋转数组恢复**|
|**LintCode 39**|Search in Rotated Sorted Array|⭐⭐⭐|**二分查找旋转数组**|
|**LintCode 31**|Next Permutation|⭐⭐⭐|**数组翻转**|
|**LintCode 179**|Rotate Array|⭐⭐⭐|**三次翻转**|
|**LintCode 39**|Find Minimum in Rotated Sorted Array|⭐⭐|**二分查找最小值**|

---

## **总结**

14. **最优解法**：
    
    - **三次翻转法 `O(n)`**
    - **时间 O(n)**，**空间 O(1)**，原地恢复。
15. **其他解法**
    
    - **`sort()` 排序（O(n log n)）**
    - **遍历插入法（O(n) + 额外空间 O(n)）**
16. **这类题目涉及**
    
    - **数组旋转**
    - **双指针 / 翻转技巧**
    - **二分查找（若查找某个元素）**

🚀 **三次翻转法是最优解，适用于所有**「恢复旋转数组」**的问题！**