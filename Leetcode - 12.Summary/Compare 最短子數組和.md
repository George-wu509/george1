
|                              |     |
| ---------------------------- | --- |
| [[#### 比較兩個數組的最短子數組符合和 > k]] |     |
|                              |     |
#### 比較兩個數組的最短子數組符合和 > k
```
請比較這兩題Lintcode題目(Lintcode 406 vs Lintcode 1507)跟解法的差別
```

| **解法類別**       | **LintCode 406 (全為正數 ≥0)**                                                                                                     | **LintCode 1507 (包含負數)**                                                                                              |
| -------------- | ------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------- |
| **雙指針 / 滑動窗口** | **【最佳解法】**<br>• 時間：$O(N)$<br>• 空間：$O(1)$<br><br>• 說明：單調性成立（右移必增、左移必減），實現簡單且空間最省。                                               | **【不能用（解答錯誤）】**<br>• 時間：$O(N)$<br>• 空間：$O(1)$<br><br>• 說明：**邏輯失效**。負數破壞單調性，右界會盲目右移至末尾，無法回溯尋找更短子數組。                    |
| **前綴和 + 哈希表**  | **【不可用 / 會超時 (TLE)】**<br>• 時間：$O(N^2)$<br>• 空間：$O(N)$<br><br>• 說明：哈希表僅支援 $O(1)$ **精確匹配**（和等於 $K$）。面對不等式範圍（和 $\ge S$）需遍歷所有 Key。 | **【不可用 / 會超時 (TLE)】**<br>• 時間：$O(N^2)$<br>• 空間：$O(N)$<br><br>• 說明：**效能失效**。無法有效率執行不等式範圍查詢 $\le P[j] - K$，時間複雜度退化至平方級。 |
| **前綴和 + 單調隊列** | **【可行（非最佳）】**<br>• 時間：$O(N)$<br>• 空間：$O(N)$<br><br>• 說明：答案正確，但相比雙指針多消耗了 $O(N)$ 的隊列空間，屬於過度設計（Overkill）。                         | **【最佳解法】**<br>• 時間：$O(N)$<br>• 空間：$O(N)$<br><br>• 說明：前綴和轉化題目後，利用單調隊列進行「隊頭淘汰」與「隊尾支配剪枝」，是唯一的 $O(N)$ 最優解。                |

### 核心結論總結

1. **LintCode 406 的關鍵在於「全正數」**：因為數值具備單調性，雙指針左右邊界只需要各自向右掃描一次，就能以 **$O(N)$ 時間與 $O(1)$ 空間** 找到最優解。
2. **LintCode 1507 的關鍵在於「含負數 + 不等式」**：
    - 雙指針因為負數破壞單調性而**直接失效**。
    - 哈希表因為無法高效處理不等式 $\ge K$ 的範圍查詢而**超時**。
    - 只有**前綴和 + 單調隊列**能透過雙端剪枝，在 $O(N)$ 時間內完美解決。

Leetcode 406 [[Minimum Size Subarray Sum]]
给定一个由 `n` 个正整数组成的数组和一个正整数 `s` ，请找出该数组中满足其和 ≥ s 的最小长度子数组
最佳解法:  雙指針 / 滑動窗口
```python
class Solution:

    def minimum_size(self, nums: List[int], s: int) -> int:
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
解釋: 
用雙指針(滑動視窗), 第一個指針i是左邊(i是左指針用for loop), 第二個指針j是右邊
1. 第一個指針i從左(id=0)一步步for loop移到右. 從一開始i在0的位置, 第二個指針j 也從id=0開始慢慢往右直到並計算sum直到sum>s. 紀錄長度=minLength
2. 第一個指針i往右移一步(id=1), sum減掉= id=0的值, 然後第二個指針j 從剛剛j停留的位置繼續往右移直到sum>s. 紀錄長度=minLength = min(minLength, j-i)
3. 繼續直到第一個指針i到最右端, minLength就是最小長度


Leetcode 1507  [[Shortest Subarray with Sum at Least K]]
返回 `A` 的最短的非空连续子数组的**长度**，该子数组的和至少为 `K`
最佳解法:  前綴和 + 單調隊列
```python
from collections import deque
def shortest_subarray(self, a, k):
	mono_queue = deque([(0, -1)])
	shortest = float('inf')
	prefix_sum = 0
	for end in range(len(a)):
		prefix_sum += a[end]
		# pop left
		while mono_queue and prefix_sum - mono_queue[0][0] >= k:
			min_prefix_sum, index = mono_queue.popleft()
			shortest = min(shortest, end - index)
		# push right
		while mono_queue and mono_queue[-1][0] >= prefix_sum:
			mono_queue.pop()
		mono_queue.append((prefix_sum, end))
	
	if shortest == float('inf'):
		return -1
	return shortest
```
解釋: 
使用完全相同的結構描述 LeetCode 1507 的解法步驟：
1. **初始化 (Initialization)**
    - 初始化雙端隊列 `queue = [(0, -1)]` 存放 `(prefix_sum, index)` 元組。
    - 設定當前前綴和 `prefix_sum = 0`，全局最小長度 `min_len = infinity`。
2. **右邊界掃描與狀態更新 (Expand Right & Accumulate)**
    - 遍歷右邊界 `right`（0 到 $N-1$），將 `nums[right]` 累加至 `prefix_sum`。
3. **雙端剪枝與隊列維護 (Dual-ended Pruning & Queue Maintenance)**
    - **隊頭檢查（更新答案）**：當 `queue` 非空且 `prefix_sum - queue[0].sum >= K` 時：
        - 更新 `min_len = min(min_len, right - queue[0].index)`。
        - 彈出隊頭 `queue.popleft()`（該隊頭對後續更右側的位置而言長度必然更長，已無保留價值）。
    - **隊尾檢查（維護單調性）**：當 `queue` 非空且 `queue[-1].sum >= prefix_sum` 時：
        - 彈出隊尾 `queue.pop()`（當前點前綴和更小且位置更靠右，已完全支配隊尾元素）。
    - 將當前 `(prefix_sum, right)` 推入隊尾。
4. **返回結果 (Return)**
    - 遍歷結束後，若 `min_len` 未被更新則返回 `-1`，否則返回 `min_len`。


### 为什么 LintCode 406 的解法不能用于 1507？

在 **LintCode 406** 中，因为所有数字都是正数：
1. 当 `sum < s` 时，右指针 `j` 必须右移才能让 `sum` 变大。
2. 当 `sum >= s` 时，左指针 `i` 右移缩小窗口，`sum` 必然变小。

但在 **LintCode 1507** 中存在负数，这种单调性被破坏。
- 例如：`nums = [2, -1, 2], K = 3`
- 当窗口包含 `[2, -1]` 时，和为 `1 < 3`。
- 如果按照滑动窗口逻辑，右指针右移加入 `2`，窗口为 `[2, -1, 2]`，和为 `3 >= 3`，长度为 `3`。
- 但实际上最短的有效子数组是末尾的 `[2, -1, 2]` 吗？不是，如果看后半部分，其实 `[2]` 只有 2 不够，但原数组中后两项 `[-1, 2]` 和只有 1。如果数组是 `[-1, 4]`，最短子数组只是 `[4]`（长度 1），但滑动窗口无法正确贪心地收缩左边界。

### LintCode 1507 的正确解法：前缀和 + 单调队列

要解决含负数的问题，需要引入**前缀和数组 `P`**，其中 $P[i]$ 表示前 $i$ 项的和。 问题转化为：找到一对 $(i, j)$，满足 $j > i$ 且 $P[j] - P[i] \ge K$，同时使长度 $j - i$ 最小。

我们维护一个存储前缀和下标的**单调递增双端队列 `deque`**：
```python
from collections import deque
from typing import List

class Solution:
    def shortestSubarray(self, nums: List[int], k: int) -> int:
        n = len(nums)
        # 1. 计算前缀和数组 P
        P = [0] * (n + 1)
        for i in range(n):
            P[i + 1] = P[i] + nums[i]
            
        min_len = n + 1
        q = deque()  # 存储前缀和下标，保持对应的前缀和单调递增
        
        for j in range(n + 1):
            # 优化 1 (弹队头)：如果当前前缀和 P[j] - P[q[0]] >= k
            # 说明找到一个可行解 j - q[0]，更新 min_len
            # 之后更靠右的 j' 即使能与 q[0] 满足条件，长度也必然大于当前的解，所以 q[0] 弹出
            while q and P[j] - P[q[0]] >= k:
                min_len = min(min_len, j - q.popleft())
                
            # 优化 2 (弹队尾)：如果当前前缀和 P[j] <= P[q[-1]]
            # 说明当前点 j 的前缀和更小，且位置更靠右（更容易凑出更大的差值且长度更短）
            # 队尾的 q[-1] 已经完全被 j 替代，从队尾弹出
            while q and P[j] <= P[q[-1]]:
                q.pop()
                
            q.append(j)
            
        return min_len if min_len <= n else -1
```
