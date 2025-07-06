
Lintcode 406
给定一个由 `n` 个正整数组成的数组和一个正整数 `s` ，请找出该数组中满足其和 ≥ s 的最小长度子数组。如果无解，则返回 -1。

比較:  Lintcode 1507  Shortest Subarray with Sum at Least K

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

**Method1(雙指針): 是最優解，具有 O(N) 的時間複雜度和 O(1) 的空間複雜度**

Method2(前綴和和哈希表):因為它通常用於查找固定和的子數組。如果要求是找到和等於 s 的子數組，哈希表才會發揮其最大優勢。

Method3(單調隊列):通常用於解決滑動窗口中的最大值/最小值問題，例如「滑動窗口最大值」或者「找到一個區間，使得區間內的最小值滿足某個條件」。對於求「和」的問題，單調隊列的直接應用較少

nums= [2,3,1,2,4,3], s = 7
雙指針解法
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
pass
解釋: 
step1: i是左指針用for loop, j是右指針 



前綴和解法 - 會超時
```python
class Solution:
    def minimum_size(self, nums: List[int], s: int) -> int:
        if not nums:
	        return -1

        prefix_sum = 0
        prefix_hash = {0: -1}  # 存储前缀和及其对应的最小索引
        min_len = float('inf')

        for i in range(n):
            prefix_sum += nums[i]

            # 檢查是否存在一個前綴和 pre_sum，使得 prefix_sum - pre_sum >= s
            # 即 pre_sum <= prefix_sum - s
            for pre_sum, index in prefix_hash.items():
                if pre_sum <= prefix_sum - s:
                    min_len = min(min_len, i - index)

            # 更新當前前綴和的最小索引
            if prefix_sum not in prefix_hash:
                prefix_hash[prefix_sum] = i

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


### Method 1: `minSubArrayLen_hashmap` (基於哈希表的方法)

**核心思想：** 這種方法使用了前綴和和哈希表。它遍歷數組，計算每個位置的前綴和。對於每個前綴和 `prefix_sum`，它會在哈希表中查找是否存在一個 `pre_sum`，使得 `prefix_sum - pre_sum >= s`。如果存在，這意味著從 `pre_sum` 對應的索引到當前索引的子數組的和滿足條件。它試圖找到滿足這個條件的最小長度。

**優點：**

- **概念相對直觀：** 對於熟悉前綴和和哈希表的開發者來說，理解其查找邏輯相對容易。

**缺點：**

- **時間複雜度高：** 在內層循環中，它會遍歷 `prefix_sum_index` 中的所有元素。在最壞的情況下，哈希表中可能包含 `n` 個元素，導致時間複雜度達到 O(n2)。這對於較大的輸入會導致超時。
- **無法處理負數：** 雖然 LintCode Minimum Size Subarray Sum 的題目通常假設數組元素為正數，但如果數組中包含負數，`prefix_sum_index` 的設計可能無法找到正確的最小長度，因為 `pre_sum <= prefix_sum - s` 這個條件可能不夠精確。當處理負數時，需要更複雜的邏輯來維護一個單調遞增的前綴和。

### Method 2: `shortest_subarray` (基於單調隊列的方法)

**核心思想：** 這種方法也使用了前綴和，但它引入了一個**單調隊列 (monotonic queue)**。隊列中存儲的是 `(prefix_sum, index)` 對，並且保證隊列中的 `prefix_sum` 是單調遞增的。

- **彈出左側 (pop left):** 當 `prefix_sum - mono_queue[0][0] >= k` 時，表示我們找到了一個滿足條件的子數組，並且 `mono_queue[0][0]` 是隊列中最小的前綴和，它能形成最短的子數組。所以我們彈出隊列頭部並更新 `shortest`。
- **彈出右側 (pop right):** 當 `mono_queue[-1][0] >= prefix_sum` 時，表示當前的前綴和 `prefix_sum` 比隊列尾部的前綴和更小或相等，但它的索引更大。這意味著隊列尾部的前綴和已經沒有用處了（因為我們總是傾向於更小的前綴和來獲得更短的子數組），所以我們將其彈出以維護隊列的單調性。
- **推入右側 (push right):** 將當前 `(prefix_sum, end)` 推入隊列。

**優點：**

- **時間複雜度優化：** 由於每個元素最多被推入和彈出隊列一次，所以整體時間複雜度為 O(n)。這是非常高效的。
- **能夠處理負數：** 單調隊列的設計可以正確處理包含負數的數組。這是因為它維護了一個單調遞增的前綴和序列，使得 `prefix_sum - mono_queue[0][0]` 能夠找到最小的合法長度。

**缺點：**

- **理解複雜：** 單調隊列的概念對於初學者來說可能比較難以理解和實現。其維護邏輯需要仔細推敲。

### 總結比較

|特性|Method 1: `minSubArrayLen_hashmap` (哈希表)|Method 2: `shortest_subarray` (單調隊列)|
|:--|:--|:--|
|**時間複雜度**|O(n2) (最壞情況)|O(n)|
|**空間複雜度**|O(n) (哈希表)|O(n) (隊列)|
|**是否處理負數**|通常不行|可以|
|**理解難度**|相對較低|相對較高|
|**效率**|較低 (對於大數據集可能超時)|較高 (推薦用於競爭性編程)|

匯出到試算表

### 結論

對於 LintCode Minimum Size Subarray Sum 這樣的問題（通常假設數組元素為正），兩種方法理論上都能得到正確結果，但 **Method 2 (單調隊列)** 在效率上遠遠優於 **Method 1 (哈希表)**。特別是當數組包含負數時，單調隊列方法是更通用和正確的解法。

在實際應用和競爭性編程中，**Method 2 是更推薦的解法**，因為它的時間複雜度更優，並且能夠處理更複雜的輸入情況。Method 1 雖然在概念上簡單，但其效率瓶頸使其不適合處理大規模數據。
