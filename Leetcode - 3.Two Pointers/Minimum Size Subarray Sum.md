
406
给定一个由 `n` 个正整数组成的数组和一个正整数 `s` ，请找出该数组中满足其和 ≥ s 的最小长度子数组。如果无解，则返回 -1。

**样例 1:**
```python
输入: [2,3,1,2,4,3], s = 7
输出: 2
解释: 子数组 [4,3] 是该条件下的最小长度子数组。
```
**样例 2:**
```python
输入: [1, 2, 3, 4, 5], s = 100
输出: -1
```

前綴和解法
```python
class Solution:
    def minSubArrayLen_hashmap(self, s: int, nums: list[int]) -> int:
        """
        找出数组中和大于或等于 s 的最小连续子数组长度 (使用哈希表优化前缀和)。

        Args:
            s: 目标和。
            nums: 输入的整数数组。

        Returns:
            最小连续子数组的长度，如果不存在则返回 0。
        """
        n = len(nums)
        if n == 0:
            return 0

        prefix_sum = 0
        prefix_sum_index = {0: -1}  # 存储前缀和及其对应的最小索引
        min_len = float('inf')

        for i in range(n):
            prefix_sum += nums[i]

            # 檢查是否存在一個前綴和 pre_sum，使得 prefix_sum - pre_sum >= s
            # 即 pre_sum <= prefix_sum - s
            for pre_sum, index in prefix_sum_index.items():
                if pre_sum <= prefix_sum - s:
                    min_len = min(min_len, i - index)

            # 更新當前前綴和的最小索引
            if prefix_sum not in prefix_sum_index:
                prefix_sum_index[prefix_sum] = i

        return min_len if min_len != float('inf') else 0
```
**時間複雜度分析：**

- **計算前綴和和遍歷數組：** 仍然是 O(n)。
- **哈希表操作：**
    - 在每次迭代中，我們可能會遍歷 `prefix_sum_index` 這個哈希表。在最壞的情況下，這個哈希表的大小可能達到 O(n)。
    - 因此，內層的 `for pre_sum, index in prefix_sum_index.items():` 循環在最壞情況下可能需要 O(n) 的時間。
    - 這使得總體的時間複雜度在最壞情況下仍然是 O(n2)。

**空間複雜度分析：**

- **哈希表 `prefix_sum_index`：** 在最壞的情況下，哈希表可能會儲存所有可能的前綴和及其索引，因此空間複雜度是 O(n)。
- **其他變數：** 佔用常數空間 O(1)。

**為什麼這個解法在平均情況下可能更好？**

雖然最壞情況下的時間複雜度仍然是 O(n2)，但在實際情況中，哈希表 `prefix_sum_index` 的大小通常不會一直保持在 O(n)。如果數組中的前綴和重複出現，哈希表的大小會小於 n。此外，一旦我們找到一個較小的滿足條件的子數組長度，`min_len` 會減小，這可能會在後續的迭代中更快地找到更小的解，或者使得滿足 `pre_sum <= prefix_sum - s` 的 `pre_sum` 的數量減少，從而減少內層迴圈的迭代次數。

**更優的解法：滑動窗口 (Two Pointers)**

值得注意的是，對於這個問題，存在一個時間複雜度為 O(n) 的更優解法，即使用**滑動窗口**技術。滑動窗口通過維護一個連續的子數組（窗口），並根據當前窗口的和與目標和 `s` 的比較，動態地調整窗口的左右邊界，從而避免了不必要的重複計算。


```python
    def minimum_Size(self, nums, s):
        if nums is None or len(nums) == 0:
            return -1

        n = len(nums)
        minLength = n + 1
        sum = 0
        j = 0
        for i in range(n):
            while j < n and sum < s:
                sum += nums[j]
                j += 1
            if sum >= s:
                minLength = min(minLength, j - i)

            sum -= nums[i]
            
        if minLength == n + 1:
            return -1
            
        return minLength
```
pass

解釋:
用雙指針, 第一個指針i是左邊, 第二個指針j是右邊
1. 第一個指針i從左(id=0)一步步移到右. 從一開始i在0的位置, 第二個指針j 也從id=0開始慢慢往右直到並計算sum直到sum>s. 紀錄長度=minLength
2. 第一個指針i往右移一步(id=1), sum減掉= id=0的值, 然後第二個指針j 從剛剛j停留的位置繼續往右移直到sum>s. 紀錄長度=minLength = min(minLength, j-i)
3. 繼續直到第一個指針i到最右端, minLength就是最小長度
