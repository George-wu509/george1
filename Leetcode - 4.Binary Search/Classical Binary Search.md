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

一些問題
```
在[Binary Search], if not nums: return -1是否可以換成if not nums: return? 另外if not nums是否是很好的寫法可以取代很多條件檢查譬如if nums == None....也幫我整理一些簡潔或推薦的條件檢查寫法. 接下來為何要使用while left + 1 < right而不是while left < right不是while left <= right等
```
**回傳 `None` vs `-1`** 如果找不到目標值，回傳 `-1` 能明確表示這是一個「無效的索引」。如果你將 `if not nums: return -1` 改成 `return`，它會輸出 `None`。如果呼叫此函式的外層程式碼嘗試使用這個結果（例如 `nums[result]` 或是 `result + 1`），Python 會拋出 `TypeError: unsupported operand type` 的型別錯誤。

**`if not nums:` 的寫法** 沒錯，`if not nums:` 是 PEP 8 (Python 官方程式碼風格指南) 推薦檢查空序列的標準寫法。在 Python 裡，空列表 `[]`、空字串 `""`、數字 `0` 以及 `None` 在布林判斷時，都會被自動視為 `False`。

**為何使用 `while left + 1 < right:`** 這個特定的模板是為了徹底根除死迴圈的問題。在傳統的 `while left < right` 二分搜尋中，如果 `left = 0`，`right = 1`，此時計算出的 `mid = 0 + (1-0)//2` 為 `0`，接著執行 `left = mid` (亦即 `left = 0`)，指標根本沒有移動！迴圈就會永遠卡在這裡。 透過使用 `left + 1 < right`，迴圈會精準地在 `left` 和 `right` 相鄰時（例如索引 0 和 1）終止。你永遠不會計算出一個讓指標卡住的 `mid`。跳出迴圈後，只需要補上簡單的後處理檢查即可：`if nums[left] == target` 以及 `if nums[right] == target`。

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

