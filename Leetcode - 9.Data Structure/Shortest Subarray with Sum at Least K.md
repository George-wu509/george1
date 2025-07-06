Lintcode 1507
返回 `A` 的最短的非空连续子数组的**长度**，该子数组的和至少为 `K` 。
如果没有和至少为 `K` 的非空子数组，返回 `-1` 。

比較:  Lintcode 406  Minimum Size Subarray Sum


**样例 1:**
```python
输入：[2,3,1,2,4,3], s = 7
输出：1
```

**样例 2:**
```python
"""
输入：A = [1,2], K = 4
输出：-1
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




前綴和解法 + 單調隊列
a= [2,3,1,2,4,3], k = 7
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
pass



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





## 最短子陣列和 (Shortest Subarray Sum) 解釋

這段程式碼的目標是找到陣列 `a` 中，其元素總和大於或等於 `k` 的**最短連續子陣列**的長度。如果沒有這樣的子陣列，則返回 -1。

程式碼使用了一個**單調佇列 (monotonic queue)** 來有效地追蹤前綴和 (prefix sums)。單調佇列的特性是，佇列中的元素按照特定的順序排列（在這個例子中是遞增）。

我們將以 `a = [2, 3, 1, 2, 4, 3]` 和 `k = 7` 為例，一步步詳細解釋程式碼的執行過程。

---

### 變數初始化

在開始迴圈之前，我們初始化以下變數：

- `mono_queue = deque([(0, -1)])`: 這是我們的單調佇列。它儲存 `(前綴和, 索引)` 對。初始時，我們加入 `(0, -1)`。`0` 代表空子陣列的前綴和，`-1` 是一個虛擬的索引，表示在陣列開始之前。這個初始值很重要，因為它允許我們處理從陣列開頭開始的子陣列。
- `shortest = float('inf')`: 用來儲存找到的最短子陣列長度。我們將其初始化為無限大，因為我們希望找到最小值。
- `prefix_sum = 0`: 用來累積當前的前綴和。

---

### 迴圈遍歷陣列 `a`

程式碼會遍歷陣列 `a` 中的每個元素，從索引 0 到 `len(a) - 1`。

#### 迭代 `end = 0` (元素 `a[0] = 2`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `0` 變為 `0 + a[0] = 2`。
2. **`pop left` 操作**: 檢查單調佇列的左側。
    
    - 條件是 `mono_queue` 不為空**且** `prefix_sum - mono_queue[0][0] >= k`。
    - 目前 `mono_queue = deque([(0, -1)])`。
    - `prefix_sum - mono_queue[0][0]` 等於 `2 - 0 = 2`。
    - `2 >= 7` 為 `False`。
    - 所以 `pop left` 迴圈不會執行。
3. **`push right` 操作**: 檢查單調佇列的右側，維持單調性（前綴和遞增）。
    
    - 條件是 `mono_queue` 不為空**且** `mono_queue[-1][0] >= prefix_sum`。
    - `mono_queue[-1][0]` 是 `0`。
    - `0 >= 2` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(prefix_sum, end)`，即 `(2, 0)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(0, -1), (2, 0)])`。

---

#### 迭代 `end = 1` (元素 `a[1] = 3`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `2` 變為 `2 + a[1] = 5`。
2. **`pop left` 操作**:
    
    - 目前 `mono_queue = deque([(0, -1), (2, 0)])`。
    - `mono_queue[0][0]` 是 `0`。
    - `prefix_sum - mono_queue[0][0]` 等於 `5 - 0 = 5`。
    - `5 >= 7` 為 `False`。
    - 所以 `pop left` 迴圈不會執行。
3. **`push right` 操作**:
    
    - `mono_queue[-1][0]` 是 `2`。
    - `2 >= 5` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(5, 1)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(0, -1), (2, 0), (5, 1)])`。

---

#### 迭代 `end = 2` (元素 `a[2] = 1`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `5` 變為 `5 + a[2] = 6`。
2. **`pop left` 操作**:
    
    - `prefix_sum - mono_queue[0][0]` 等於 `6 - 0 = 6`。
    - `6 >= 7` 為 `False`。
    - 所以 `pop left` 迴圈不會執行。
3. **`push right` 操作**:
    
    - `mono_queue[-1][0]` 是 `5`。
    - `5 >= 6` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(6, 2)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(0, -1), (2, 0), (5, 1), (6, 2)])`。

---

#### 迭代 `end = 3` (元素 `a[3] = 2`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `6` 變為 `6 + a[3] = 8`。
2. **`pop left` 操作**:
    
    - 目前 `mono_queue = deque([(0, -1), (2, 0), (5, 1), (6, 2)])`。
    - **第一次檢查**:
        - `mono_queue[0][0]` 是 `0`。
        - `prefix_sum - mono_queue[0][0]` 等於 `8 - 0 = 8`。
        - `8 >= 7` 為 `True`。
        - 符合條件，所以執行 `popleft()`。`min_prefix_sum = 0`, `index = -1`。
        - 更新 `shortest = min(shortest, end - index)`。
        - `shortest = min(inf, 3 - (-1)) = min(inf, 4) = 4`。
        - `mono_queue` 現在是 `deque([(2, 0), (5, 1), (6, 2)])`。
    - **第二次檢查**:
        - `mono_queue[0][0]` 是 `2`。
        - `prefix_sum - mono_queue[0][0]` 等於 `8 - 2 = 6`。
        - `6 >= 7` 為 `False`。
        - 不符合條件，`pop left` 迴圈結束。
3. **`push right` 操作**:
    
    - `mono_queue[-1][0]` 是 `6`。
    - `6 >= 8` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(8, 3)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(2, 0), (5, 1), (6, 2), (8, 3)])`。

---

#### 迭代 `end = 4` (元素 `a[4] = 4`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `8` 變為 `8 + a[4] = 12`。
2. **`pop left` 操作**:
    
    - 目前 `mono_queue = deque([(2, 0), (5, 1), (6, 2), (8, 3)])`。
    - **第一次檢查**:
        - `mono_queue[0][0]` 是 `2`。
        - `prefix_sum - mono_queue[0][0]` 等於 `12 - 2 = 10`。
        - `10 >= 7` 為 `True`。
        - 符合條件，執行 `popleft()`。`min_prefix_sum = 2`, `index = 0`。
        - 更新 `shortest = min(shortest, end - index)`。
        - `shortest = min(4, 4 - 0) = min(4, 4) = 4`。
        - `mono_queue` 現在是 `deque([(5, 1), (6, 2), (8, 3)])`。
    - **第二次檢查**:
        - `mono_queue[0][0]` 是 `5`。
        - `prefix_sum - mono_queue[0][0]` 等於 `12 - 5 = 7`。
        - `7 >= 7` 為 `True`。
        - 符合條件，執行 `popleft()`。`min_prefix_sum = 5`, `index = 1`。
        - 更新 `shortest = min(shortest, end - index)`。
        - `shortest = min(4, 4 - 1) = min(4, 3) = 3`。
        - `mono_queue` 現在是 `deque([(6, 2), (8, 3)])`。
    - **第三次檢查**:
        - `mono_queue[0][0]` 是 `6`。
        - `prefix_sum - mono_queue[0][0]` 等於 `12 - 6 = 6`。
        - `6 >= 7` 為 `False`。
        - 不符合條件，`pop left` 迴圈結束。
3. **`push right` 操作**:
    
    - `mono_queue[-1][0]` 是 `8`。
    - `8 >= 12` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(12, 4)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(6, 2), (8, 3), (12, 4)])`。

---

#### 迭代 `end = 5` (元素 `a[5] = 3`)

1. **更新 `prefix_sum`**:
    
    - `prefix_sum` 從 `12` 變為 `12 + a[5] = 15`。
2. **`pop left` 操作**:
    
    - 目前 `mono_queue = deque([(6, 2), (8, 3), (12, 4)])`。
    - **第一次檢查**:
        - `mono_queue[0][0]` 是 `6`。
        - `prefix_sum - mono_queue[0][0]` 等於 `15 - 6 = 9`。
        - `9 >= 7` 為 `True`。
        - 符合條件，執行 `popleft()`。`min_prefix_sum = 6`, `index = 2`。
        - 更新 `shortest = min(shortest, end - index)`。
        - `shortest = min(3, 5 - 2) = min(3, 3) = 3`。
        - `mono_queue` 現在是 `deque([(8, 3), (12, 4)])`。
    - **第二次檢查**:
        - `mono_queue[0][0]` 是 `8`。
        - `prefix_sum - mono_queue[0][0]` 等於 `15 - 8 = 7`。
        - `7 >= 7` 為 `True`。
        - 符合條件，執行 `popleft()`。`min_prefix_sum = 8`, `index = 3`。
        - 更新 `shortest = min(shortest, end - index)`。
        - `shortest = min(3, 5 - 3) = min(3, 2) = 2`。
        - `mono_queue` 現在是 `deque([(12, 4)])`。
    - **第三次檢查**:
        - `mono_queue[0][0]` 是 `12`。
        - `prefix_sum - mono_queue[0][0]` 等於 `15 - 12 = 3`。
        - `3 >= 7` 為 `False`。
        - 不符合條件，`pop left` 迴圈結束。
3. **`push right` 操作**:
    
    - `mono_queue[-1][0]` 是 `12`。
    - `12 >= 15` 為 `False`。
    - 所以 `push right` 迴圈不會執行。
4. **添加到單調佇列**:
    
    - 將 `(15, 5)` 添加到 `mono_queue` 中。
    - `mono_queue` 現在是 `deque([(12, 4), (15, 5)])`。

---

### 迴圈結束後的處理

迴圈結束後，`shortest` 的值是 `2`。

- `if shortest == float('inf'):`
    
    - `2 == float('inf')` 為 `False`。
- 返回 `shortest`。
    

---

### 最終結果

對於 `a = [2, 3, 1, 2, 4, 3]` 和 `k = 7`，最短子陣列的長度是 **2**。

**解釋最短子陣列：** 當 `end = 5` 時，`prefix_sum = 15`。 我們從 `mono_queue` 中彈出了 `(8, 3)`。這意味著 `prefix_sum` (15) 減去 `8` (`prefix_sum` 到索引 3 的和) 等於 `7`，而 `7 >= k` (7)。這個子陣列是 `a[4:6]`，即 `[4, 3]`，其和為 `7`。 這個子陣列的長度是 `end - index = 5 - 3 = 2`。這就是我們找到的最短子陣列長度。

---

### 總結

這個演算法的關鍵在於**單調佇列**。

- **`pop left` 操作**：當 `prefix_sum - mono_queue[0][0] >= k` 時，表示從 `mono_queue[0][0]` 對應的索引到當前 `end` 索引的子陣列和 `>= k`。我們嘗試將這個子陣列的長度與 `shortest` 比較，並從佇列左側移除這個元素，因為任何包含這個 `mono_queue[0][0]` 的更長的子陣列（即從 `mono_queue[0][0]` 開始到後面 `end` 的子陣列）都不會比這個子陣列更短，我們只關心最短的。
    
- **`push right` 操作**：當 `mono_queue[-1][0] >= prefix_sum` 時，表示佇列最右側的前綴和大於或等於當前的前綴和。這意味著佇列最右側的元素對於未來尋找滿足條件的子陣列是「冗餘」的。因為我們總是希望從**盡可能小的前綴和**開始構建子陣列，這樣才能得到最短的長度。如果後面有更大的前綴和，它不可能提供一個比當前 `prefix_sum` 更短的有效子陣列，因為它的起點更靠右，而值卻更大，所以移除它以保持佇列的單調性（遞增）。
    

這個方法的時間複雜度是 O(N)，因為每個元素最多被推入和彈出佇列一次。