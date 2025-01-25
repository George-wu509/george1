给出一个整数数组 `nums` 和一个整数 `k`。划分数组（即移动数组 `nums` 中的元素），使得：

- 所有小于`k`的元素移到左边
- 所有大于等于`k`的元素移到右边

返回数组划分的位置，即数组中第一个位置 `i`，满足 `nums[i]` 大于等于 `k`。


**样例 1：**
输入：
```python
nums = []
k = 9
```
输出：
```python
0
```
解释：
空数组，输出0

**样例 2：**
输入：
```python
nums = [3,2,2,1]
k = 2
```
输出：
```python
1
```
解释：
真实的数组为[1,2,2,3].所以返回 1

```python
class Solution:
    """
    @param nums: The integer array you should partition
    @param k: An integer
    @return: The index after partition
    """
    def partition_array(self, nums, k):
        left, right = 0, len(nums) - 1
        
        while left <= right:
            while left <= right and nums[left] < k:
                left += 1
            while left <= right and nums[right] >= k:
                right -= 1
            if left <= right:
                nums[left], nums[right] = nums[right], nums[left]
                left += 1
                right -= 1
        
        return left
```
pass



在解决 LintCode 31 (分割数组) 问题时，代码中使用了双指针和二分查找的结合，外部 `while left <= right` 和内部两个 `while` 循环分别承担不同的职责。

让我们通过代码结构和两个具体例子详细解释为什么要这样设计，以及它们在解决问题中的作用。

---

### **代码的基本逻辑**

```python
def partitionArray(nums, k):
    if not nums:
        return 0

    left, right = 0, len(nums) - 1

    while left <= right:
        while left <= right and nums[left] < k:
            left += 1
        while left <= right and nums[right] >= k:
            right -= 1

        if left <= right:
            nums[left], nums[right] = nums[right], nums[left]
            left += 1
            right -= 1

    return left

```

#### **关键逻辑：**

1. **外部循环 (`while left <= right`)**：
    
    - 确保指针 `left` 和 `right` 在数组中有效。
    - 每次交换操作后重新检查是否需要进一步分割。
2. **内部两个 `while` 循环**：
    
    - **`while left <= right and nums[left] < k:`**
        - 找到 `left` 指针的位置，指向第一个不小于 kkk 的元素。
    - **`while left <= right and nums[right] >= k:`**
        - 找到 `right` 指针的位置，指向第一个小于 kkk 的元素。
3. **交换逻辑**：
    
    - 当 `left` 和 `right` 分别找到需要交换的元素时，将它们交换，并继续缩小搜索范围。

---

### **为什么需要内部两个 `while` 循环？**

内部两个 `while` 循环的目的是让 `left` 和 `right` 指针分别移动到正确的位置。以下是两个例子说明这种设计的必要性：

---

### **例子 1：简单分割，数组中已经有部分元素满足条件**

输入：

python

複製程式碼

`nums = [1, 2, 3, 4, 5] k = 3`

#### 运行步骤：

1. 初始状态：
    - `left = 0`, `right = 4`, `nums = [1, 2, 3, 4, 5]`
2. 第一次进入外部 `while`：
    - **内部第一个 `while`：** 从左到右，`nums[0] = 1 < 3`, `nums[1] = 2 < 3`，因此 `left = 2` 停止。
    - **内部第二个 `while`：** 从右到左，`nums[4] = 5 >= 3`, `nums[3] = 4 >= 3`，因此 `right = 2` 停止。
3. **退出外部 `while`：** 此时 `left = 2`, `right = 2`，指针重叠，结束循环。

**结果：**

- 所有小于 kkk 的元素（1, 2）已经在数组左边，`partitionIndex = 2`。

---

### **例子 2：需要交换元素**

输入：

python

複製程式碼

`nums = [5, 2, 4, 1, 3] k = 3`

#### 运行步骤：

1. 初始状态：
    
    - `left = 0`, `right = 4`, `nums = [5, 2, 4, 1, 3]`
2. 第一次进入外部 `while`：
    
    - **内部第一个 `while`：** `nums[0] = 5 >= 3`，停止，`left = 0`。
    - **内部第二个 `while`：** `nums[4] = 3 >= 3`，`nums[3] = 1 < 3`，停止，`right = 3`。
    - 交换 `nums[0]` 和 `nums[3]`，结果：`nums = [1, 2, 4, 5, 3]`，`left = 1`, `right = 2`。
3. 第二次进入外部 `while`：
    
    - **内部第一个 `while`：** `nums[1] = 2 < 3`，停止，`left = 2`。
    - **内部第二个 `while`：** `nums[2] = 4 >= 3`，停止，`right = 1`。
    - 此时 `left > right`，退出循环。

**结果：**

- 数组分割为两部分：`[1, 2]` 和 `[4, 5, 3]`，`partitionIndex = 2`。

---

### **为什么需要外部 `while left <= right`？**

外部循环控制指针的全局移动，确保所有元素都被检查和分割：

- 如果没有这个循环，交换后的元素不会被重新检查，可能导致结果不正确。

---

### **复杂度分析**

- **时间复杂度：**
    - 每次移动指针或交换操作最多执行 O(n)O(n)O(n) 次，因此总时间复杂度为 O(n)O(n)O(n)。
- **空间复杂度：**
    - 原地操作，无需额外空间，空间复杂度为 O(1)O(1)O(1)。

---

### **总结**

- **`while left <= right`：** 控制整体的检查和分割流程。
- **`while left <= right and nums[left] < k` / `while left <= right and nums[right] >= k`：** 找到需要交换的左右指针位置。

通过这种设计，算法能够高效地对数组进行分割并确保正确性。

4o

  

O