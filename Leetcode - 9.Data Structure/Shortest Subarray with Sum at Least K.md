Lintcode 1507
返回 `A` 的最短的非空连续子数组的**长度**，该子数组的和至少为 `K` 。如果没有和至少为 `K` 的非空子数组，返回 `-1` 。

比較:  Lintcode 406  Minimum Size Subarray Sum
[[Compare 最短子數組和]]

|                                            |     |
| ------------------------------------------ | --- |
| [[#### 題目描述跟三種解法描述]]                       |     |
| [[#### 最短子陣列和 (Shortest Subarray Sum) 解釋]] |     |
| [[#### 前綴和解法 + 單調隊列解法5個問題]]                |     |
|                                            |     |

#### 題目描述跟三種解法描述
**样例 1:**
```python
输入：[2,3,1,2,4,3], s = 7
输出：2
```

**样例 2:**
```python
"""
输入：A = [1,2], K = 4
输出：-1
```

Method1(雙指針): 不能使用, 因為滑動窗口要有單調性

Method2(前綴和+哈希表):因為它通常用於查找固定和的子數組。如果要求是找到和等於 s 的子數組，哈希表才會發揮其最大優勢。

Method3(前綴和+單調隊列):通常用於解決滑動窗口中的最大值/最小值問題，例如「滑動窗口最大值」或者「找到一個區間，使得區間內的最小值滿足某個條件」。對於求「和」的問題，單調隊列的直接應用較少. **標準最優解法**（時間複雜度 $O(N)$，空間複雜度 $O(N)$）

Method3:前綴和解法 + 單調隊列 (最佳解法)
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
解釋:
子數組 `a[i...j]` 的和可以表示為前綴和之差：$P[j + 1] - P[i] \ge K$。目標是找到下標差 $(j + 1) - i$ 最小的組合. 巧思在於使用 `deque([(0, -1)])` 直接儲存 `(前綴和, 下標)` 元組，省去了顯式建立前綴和數組 `P` 的空間分配，邏輯非常精簡

單調隊列維護一個**前綴和嚴格單調遞增**的雙端隊列，依靠兩個剪枝原則：
1. **隊頭彈出 (`popleft`) —— 尋找最佳解與淘汰舊下標**
    - 當 `當前前綴和 - 隊頭前綴和 >= K` 時，說明找到了一個符合條件的子數組。
    - **為什麼能 popleft？** 因為我們要求的是**最短**子數組。對於後續更靠右的 `end`，就算能與當前隊頭匹配，長度也必然比現在更長。因此該隊頭已發揮最大價值，可以直接廢棄。
2. **隊尾彈出 (`pop`) —— 維護單調性與支配性剪枝 (Dominance)**
    - 當 `隊尾前綴和 >= 當前前綴和` 時，彈出隊尾。
    - **為什麼能 pop？** 當前的 `prefix_sum` 不僅數值更小（更容易湊出 $\ge K$ 的差值），而且位置更靠右（與未來位置構成的子數組更短）。此時隊尾元素在「數值」與「位置」上都被當前元素完全支配（Dominated），隊尾元素再無保留價值。

### Method 3: `shortest_subarray` (基於單調隊列的方法)

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


### 結論

對於 LintCode Minimum Size Subarray Sum 這樣的問題（通常假設數組元素為正），兩種方法理論上都能得到正確結果，但 **Method 2 (單調隊列)** 在效率上遠遠優於 **Method 1 (哈希表)**。特別是當數組包含負數時，單調隊列方法是更通用和正確的解法。

在實際應用和競爭性編程中，**Method 2 是更推薦的解法**，因為它的時間複雜度更優，並且能夠處理更複雜的輸入情況。Method 1 雖然在概念上簡單，但其效率瓶頸使其不適合處理大規模數據。





#### 最短子陣列和 (Shortest Subarray Sum) 解釋
```

```

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


#### 前綴和解法 + 單調隊列解法5個問題
```
關於以下用前綴和解法 + 單調隊列求解Shortest Subarray with Sum at Least K問題(舉例: a=[2,3,1,2,4,3], k = 7), 請分析我以下的理解是否正確:

1. prefix_sum += a[end]在這一行的用意是在用for loop遍歷a慢慢加入裡面的id, 計算到這個id的pre sum. 

2. 而根據mono_queue.append((prefix_sum, end)), mono_queue在for loop裡每次一定會append(這個元素的pre sum, id). 所以在這一題mono_queue就是要跟pre sum比對去計算出Subarray的sum

3. while mono_queue and prefix_sum - mono_queue[0][0] >= k這一行就是為了計算Subarray的sum找出sum>k, 但為何之後要直接popleft mono_queue的最左邊元素? 是不是為了比對看有沒有比shortest更短?

4. 在這while迴圈內怎麼維持mono_queue是單調隊列(哪幾行code, 為何)? 是這一段嗎? while mono_queue and mono_queue[-1][0] >= prefix_sum:
mono_queue.pop()

5. 因為我只看到在for迴圈裡面有兩個while迴圈, 第一個是為了做pop left, 第二個是為了做push right. 我還是不太理解 單調隊列在這一題的功用, 根據我的理解單調隊列可以計算在滑動窗口的最大值, 可是這一題不是這樣.

```
你的理解**大方向非常精準（前 4 點有 90% 以上完全正確）**，最核心的困惑集中在第 3 點的「為何丟掉隊頭」以及第 5 點的「單調隊列與滑動窗口最大值的關聯」。以下逐點為你驗證與補充關鍵邏輯：

### 1. `prefix_sum += a[end]` 的用意
```
1. prefix_sum += a[end]在這一行的用意是在用for loop遍歷a慢慢加入裡面的id, 計算到這個id的pre sum. 
```
**完全正確。** 這行程式碼在遍歷過程中動態累加前綴和，代表從下標 `0` 加到當前 `end` 的總和（數學上的 $P[\text{end}+1]$）。這樣做省去了額外開一個陣列 `P` 的空間。

### 2. `mono_queue.
append((prefix_sum, end))` 與隊列功用
```
2. 而根據mono_queue.append((prefix_sum, end)), mono_queue在for loop裡每次一定會append(這個元素的pre sum, id). 所以在這一題mono_queue就是要跟pre sum比對去計算出Subarray的sum
```
**完全正確。**`mono_queue` 裡存放的是過去所有「還有機會成為最短子數組起點（前一位）」的 `(前綴和, 下標)`。 因為任意連續子數組 `a[index + 1 ... end]` 的總和，就等於：
$$\text{當前前綴和 } (\text{prefix\_sum}) - \text{過去某個前綴和 } (\text{mono\_queue}[i][0])$$
所以隊列存這些資料，正是為了讓後面的 `end` 來做減法、湊出子數組總和。

### 3. 為何滿足條件後要直接 `popleft()`？
```
3. while mono_queue and prefix_sum - mono_queue[0][0] >= k這一行就是為了計算Subarray的sum找出sum>k, 但為何之後要直接popleft mono_queue的最左邊元素? 是不是為了比對看有沒有比shortest更短?
```
**前半句完全正確，後半句「比對有沒有更短」只是原因之一，更關鍵的原因是「貪心淘汰」。**

當 `prefix_sum - mono_queue[0][0] >= k` 成立時，`popleft()` 同時達成兩個目的：

1. **在當前 `end` 嘗試找更短的解（你猜對的部分）**：隊頭 `mono_queue[0]` 彈出後，原本排在第二位的 `mono_queue[1]` 變成新隊頭。因為它的下標 `index` 更靠右，如果它也能滿足條件，算出來的 `end - index` 就會比剛剛的 `shortest` 更短。
    
2. **該隊頭對「未來的 `end`」已經徹底沒用了（最關鍵原因）**：假設這個舊起點在當前 `end` 已經成功湊出總和 `>= k`（長度為 $L$）。那麼對於未來更後面的 `end_future` 來說，就算也能跟這個舊起點湊出 `>= k`，算出来的長度 `end_future - index` **必定大於 $L$**。既然題目求的是**最短**，這個舊起點已經達成了它這輩子最短的紀錄，直接丟掉絕不會錯過最佳解。
    

### 4. 哪幾行維持單調隊列？
```
4. 在這while迴圈內怎麼維持mono_queue是單調隊列(哪幾行code, 為何)? 是這一段嗎? while mono_queue and mono_queue[-1][0] >= prefix_sum:
mono_queue.pop()
```
**完全正確！** 就是你框出來的這段（搭配緊接著的 `append`）：
```Python
while mono_queue and mono_queue[-1][0] >= prefix_sum:
    mono_queue.pop()
mono_queue.append((prefix_sum, end))
```

這段程式碼強制讓 `mono_queue` 裡的前綴和從左到右保持**嚴格單調遞增**（`隊頭前綴和 < ... < 隊尾前綴和`）。

**背後的「雙重碾壓（支配）」邏輯：** 如果隊尾元素的前綴和 `>=` 當前的 `prefix_sum`，代表當前的 `end` 不僅**前綴和更小**（未來別人減去它時，差值更大、更容易 `>= k`），而且**位置更靠右**（未來算出來的子數組長度更短）。隊尾那個「數值又大、位置又靠左」的舊元素在各方面都輸給當前點，永遠不可能成為最佳解，所以直接從隊尾淘汰。

### 5. 破除迷思：為什麼「單調隊列」能用在這一題？
```
5. 因為我只看到在for迴圈裡面有兩個while迴圈, 第一個是為了做pop left, 第二個是為了做push right. 我還是不太理解 單調隊列在這一題的功用, 根據我的理解單調隊列可以計算在滑動窗口的最大值, 可是這一題不是這樣.
```
你對單調隊列的印象沒錯，它最經典的題目是「滑動窗口最大值」，但**單調隊列的真正本質不是「求窗口極值」，而是「動態維護有潛力的候選集合，把不可能成為最優解的選項提前踢掉」**。

|                                                                                                           |     |
| --------------------------------------------------------------------------------------------------------- | --- |
| 第一個 `while` 迴圈（`popleft`）負責**結算最佳解並淘汰已用盡價值的舊起點**：一旦隊頭能與當前位置湊出和 $\ge k$ 就更新最短長度並將其移除，因為它對未來的右邊界只會形成更長的子數組。 |     |
| 第二個 `while` 迴圈（`pop`）則負責**維護前綴和單調遞增並進行支配剪枝**：將隊尾那些「前綴和更大、位置又更靠左」的劣勢候選點全部踢除，確保隊列只保留最有潛力湊出最短距離的起點。          |     |

在這一題中，我們希望公式 `當前前綴和 - 舊前綴和 >= k` 越容易成立越好，且長度越短越好，這代表我們需要找的舊元素必須同時具備兩個特質：
- **前綴和盡量小**（減數越小，差值越大，越容易 `>= k`）
- **下標盡量大 / 靠右**（起點越靠右，子數組越短）

透過第 4 點的維護，隊列裡留下了所有「既小又靠右」的精英候選人，而且它們從左到右排好隊
- **最左邊（`mono_queue[0]`）** 是目前所有候選人裡**前綴和最小**的（最容易讓 `prefix_sum - mono_queue[0][0] >= k` 達標）。
- **越往右邊** 前綴和越大，但**下標也越靠右**（一旦達標，長度會更短）。

這就是為什麼只要每次拿 `prefix_sum` 去跟**隊頭 `mono_queue[0]`** 比較就夠了——因為連全場最小的隊頭都湊不到 `k` 的話，隊列後面那些前綴和更大的元素就更不可能湊到了！

> **補充你舉的例子 `a = [2, 3, 1, 2, 4, 3], k = 7`：**
> 
> 因為這個例子的數字**全部都是正數**，前綴和 `[0, 2, 5, 6, 8, 12, 15]` 本身就已經是嚴格遞增的，所以第 4 點的 `while` 迴圈在這個例子中一次都不會觸發（這也是為什麼全正數時直接用雙指針滑動窗口就能解）。只有當陣列出現**負數**（例如 `a = [2, -1, 3]`）導致前綴和往下掉時，第 4 點的隊尾彈出才會真正發揮威力。