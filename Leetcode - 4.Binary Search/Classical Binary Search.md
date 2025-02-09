Lintcode 457
在一个排序数组中找一个数，返回该数出现的任意位置，如果不存在，返回 `-1`。

**样例 1：**
```python
"""
输入：nums = [1,2,2,4,5,5], target = 2
输出：1 或者 2
```
**样例 2：**
```python
"""
输入：nums = [1,2,2,4,5,5], target = 6
输出：-1
```

```python
"""
    def binarySearch(self, nums, target):
        if not nums:
            return -1

        start, end = 0, len(nums) - 1

        while start + 1 < end:

            mid = (start + end) // 2

            if nums[mid] < target:
                start = mid
            elif nums[mid] == target:
                end = mid
            else:
                end = mid

        if nums[start] == target:
            return start
        if nums[end] == target:
            return end

        return -1

```
pass



在二分查找中，`while start + 1 < end` 的目的是防止无限循环，并确保当循环结束时，`start` 和 `end` 紧邻（即 `start` 与 `end` 相差 1）。这对于检查目标元素是否在数组中至关重要。

使用 `while start < end` 会导致某些情况下算法无法正确收敛到目标值，或者漏检目标值。以下是两个具体的例子，说明在 `start < end` 的情况下可能会失败的场景：

---

### 例子 1：数组中没有目标值

输入：
`nums = [1, 2, 4, 5, 6] target = 3`

分析：

1. 初始值：`start = 0`, `end = 4`
2. 如果使用 `while start < end`：
    - 第一次计算 `mid = (0 + 4) // 2 = 2`，`nums[mid] = 4 > target`，所以 `end = mid = 2`
    - 第二次计算 `mid = (0 + 2) // 2 = 1`，`nums[mid] = 2 < target`，所以 `start = mid = 1`
    - 此时 `start = 1`，`end = 2`，`start < end` 条件依然成立，继续循环：
        - 计算 `mid = (1 + 2) // 2 = 1`，`nums[mid] = 2`，`start` 再次更新为 1，导致死循环。

**原因**：在某些情况下，`start` 和 `end` 没有正确地收敛到紧邻状态，循环无法退出。

---

### 例子 2：目标值位于边界

输入：

`nums = [1, 3, 5, 7, 9] target = 7`

分析：

1. 初始值：`start = 0`, `end = 4`
2. 如果使用 `while start < end`：
    - 第一次计算 `mid = (0 + 4) // 2 = 2`，`nums[mid] = 5 < target`，所以 `start = mid = 2`
    - 第二次计算 `mid = (2 + 4) // 2 = 3`，`nums[mid] = 7 == target`，`end = mid = 3`
    - 此时 `start = 2`，`end = 3`，`start < end` 条件依然成立，继续循环：
        - 计算 `mid = (2 + 3) // 2 = 2`，`nums[mid] = 5`，`start` 再次更新为 2，导致 `start` 无法收敛到正确的边界值 `3`。

**结果**：由于循环条件没有正确收敛，目标值可能被漏检。

---

### 为什么 `while start + 1 < end` 能避免这些问题？

- **快速收敛**：`while start + 1 < end` 确保 `start` 和 `end` 最后相差 1，退出循环时只需检查两个元素即可。
- **避免死循环**：当 `start + 1 == end` 时，循环自然退出，不会陷入死循环。
- **边界检查明确**：退出循环后，`nums[start]` 和 `nums[end]` 是两个唯一可能包含目标值的位置，逻辑清晰。

总结：`while start + 1 < end` 是为了保证算法的正确性和收敛性，而 `while start < end` 在某些情况下会导致死循环或遗漏目标值，因此不适合在这种二分查找逻辑中使用。

