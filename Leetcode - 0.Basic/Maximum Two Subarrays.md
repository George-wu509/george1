Lintcode 42
给定一个整数数组，找出两个 _不重叠_ 子数组使得它们的和最大。  每个子数组的数字在数组中的位置应该是连续的。  返回最大的和。

**样例 1：**
输入：
```python
#nums = [1, 3, -1, 2, -1, 2]
```
输出：
```python
#7
```
解释：
最大的子数组为 [1, 3] 和 [2, -1, 2] 或者 [1, 3, -1, 2] 和 [2].  

**样例 2：**
输入：
```python
#nums = [5,4]
```
输出：
```python
#9
```
解释：
最大的子数组为 [5] 和 [4].


```python

def max_two_sub_arrays(nums):
    """
    找出兩個不重疊子陣列的最大和。

    Args:
        nums: 一個整數陣列。

    Returns:
        兩個不重疊子陣列的最大和。
    """

    n = len(nums)
    if n < 2:
        return 0  # 或者拋出異常，視題目要求而定

    # 1. 計算前綴最大和
    left_max = [0] * n  # left_max[i] 表示到 i 為止的最大子陣列和
    max_so_far = -float('inf')
    current_max = 0
    for i in range(n):
        current_max = max(nums[i], current_max + nums[i])  
        max_so_far = max(max_so_far, current_max)
        left_max[i] = max_so_far   # left_max:到每個位置i為止，整個陣列中最大的子陣列和
	# left_max = [1,4,4,5,5,6]

    # 2. 計算後綴最大和
    right_max = [0] * n  # right_max[i] 表示從 i 開始到結尾的最大子陣列和
    max_so_far = -float('inf')
    current_max = 0
    for i in range(n - 1, -1, -1):
        current_max = max(nums[i], current_max + nums[i])
        max_so_far = max(max_so_far, current_max)
        right_max[i] = max_so_far
	# right_max = [6,5,3,3,2,2]

    # 3. 合併結果
    max_total = -float('inf')
    for i in range(n - 1):
        max_total = max(max_total, left_max[i] + right_max[i + 1])

    return max_total
```
pass
解釋:
step1:  for loop遍歷nums計算




**問題描述 (Lintcode 42):**

給定一個整數陣列 `nums`，我們要找出兩個**不重疊**的連續子陣列，使得這兩個子陣列的和加起來最大。你需要返回這個最大的和。

**解題思路:**

核心思想是：

1. **計算前綴最大和:** 我們從左到右遍歷 `nums`，計算到每個位置 `i` 為止，以 `i` 結尾的最大子陣列和。同時，我們也記錄到每個位置 `i` 為止，整個陣列中最大的子陣列和。
2. **計算後綴最大和:** 我們從右到左遍歷 `nums`，計算從每個位置 `i` 開始到結尾的最大子陣列和。
3. **合併結果:** 最後，我們遍歷 `nums`，對於每個位置 `i`，將 "到 `i` 為止的最大子陣列和" 和 "從 `i+1` 開始到結尾的最大子陣列和" 相加，並取所有這些和的最大值。

**程式碼解釋:**

1. **`maxTwoSubArrays(nums)` 函式:**
    - 接收一個整數陣列 `nums` 作為輸入。
    - 處理邊界情況：如果 `nums` 長度小於 2，則返回 0 或拋出異常。
    - **前綴最大和計算:**
        - `left_max` 陣列儲存到每個位置 `i` 為止的最大子陣列和。
        - 使用 Kadane's Algorithm 計算最大子陣列和 ( `current_max` 和 `max_so_far` )。
    - **後綴最大和計算:**
        - `right_max` 陣列儲存從每個位置 `i` 開始到結尾的最大子陣列和。
        - 同樣使用 Kadane's Algorithm，但從右向左計算。
    - **合併結果:**
        - 遍歷 `nums`，將 `left_max[i]` (到 `i` 為止的最大和) 和 `right_max[i+1]` (從 `i+1` 開始的最大和) 相加。
        - 更新 `max_total` 以保持最大的和。
    - 返回 `max_total`。

**用 `nums = [1, 3, -1, 2, -1, 2]` 舉例:**

1. **計算 `left_max`:**
    
    - `current_max` 和 `max_so_far` 會不斷更新。
    - `left_max` 的計算過程：
        - `i = 0`: `current_max = 1`, `max_so_far = 1`, `left_max[0] = 1`
        - `i = 1`: `current_max = 4`, `max_so_far = 4`, `left_max[1] = 4`
        - `i = 2`: `current_max = 3`, `max_so_far = 4`, `left_max[2] = 4`
        - `i = 3`: `current_max = 5`, `max_so_far = 5`, `left_max[3] = 5`
        - `i = 4`: `current_max = 4`, `max_so_far = 5`, `left_max[4] = 5`
        - `i = 5`: `current_max = 6`, `max_so_far = 6`, `left_max[5] = 6`
    - 所以，`left_max = [1, 4, 4, 5, 5, 6]`
2. **計算 `right_max`:**
    
    - 從右向左計算。
    - `right_max` 的計算過程：
        - `i = 5`: `current_max = 2`, `max_so_far = 2`, `right_max[5] = 2`
        - `i = 4`: `current_max = 1`, `max_so_far = 2`, `right_max[4] = 2`
        - `i = 3`: `current_max = 3`, `max_so_far = 3`, `right_max[3] = 3`
        - `i = 2`: `current_max = 2`, `max_so_far = 3`, `right_max[2] = 3`
        - `i = 1`: `current_max = 4`, `max_so_far = 4`, `right_max[1] = 4`
        - `i = 0`: `current_max = 5`, `max_so_far = 5`, `right_max[0] = 5`
    - 所以，`right_max = [5, 4, 3, 3, 2, 2]`
3. **合併結果:**
    
    - 遍歷 `nums` 並計算 `left_max[i] + right_max[i+1]`：
        - `i = 0`: `left_max[0] + right_max[1] = 1 + 4 = 5`
        - `i = 1`: `left_max[1] + right_max[2] = 4 + 3 = 7`
        - `i = 2`: `left_max[2] + right_max[3] = 4 + 3 = 7`
        - `i = 3`: `left_max[3] + right_max[4] = 5 + 2 = 7`
        - `i = 4`: `left_max[4] + right_max[5] = 5 + 2 = 7`
    - `max_total` 會是 7。

因此，對於 `nums = [1, 3, -1, 2, -1, 2]`，兩個不重疊子陣列的最大和是 7。