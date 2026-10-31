
|                              |     |
| ---------------------------- | --- |
| [[#### 比較兩個數組的最短子數組符合和 > k]] |     |
|                              |     |
#### 比較兩個數組的最短子數組符合和 > k
```
比較兩個數組的最短子數組符合和 > k
```

## 一、406 vs 1507 完整比較

| 比較項目        | LintCode 406              | LintCode 1507                         |
| ----------- | ------------------------- | ------------------------------------- |
| 題目          | Minimum Size Subarray Sum | Shortest Subarray with Sum at Least K |
| 目標          | 最短連續子陣列，其和 ≥ K            | 最短連續子陣列，其和 ≥ K                        |
| 陣列限制        | 全部正整數                     | 可包含正數、負數、0                            |
| 核心差異        | Window Sum 隨右指標擴張而增加      | Window Sum 可能增加或減少                    |
| 最佳演算法       | Sliding Window            | Prefix Sum + Monotonic Deque          |
| 資料結構        | Two Pointers              | Deque                                 |
| 時間複雜度       | O(n)                      | O(n)                                  |
| 空間複雜度       | O(1)                      | O(n)                                  |
| 對應 LeetCode | 209                       | 862                                   |

## 二、為什麼 406 可以使用 Sliding Window？

406 的關鍵是所有元素都大於 0。

```
nums = [1, 1, 1, 2]s = 3
```

你提供的程式碼使用兩個指標：

- `i`：左邊界（Window Start）
    
- `j`：右邊界的下一個位置（Window End Exclusive）
    
- `sum`：目前 Window 內的元素總和
    

Sliding Window 動態示範

操作 1 / 7

目標 ≥ 3

0

# 1

1

# 1

2

# 1

3

# 2

i = 0, j = 0

sum = 0

初始狀態，Window 為空

上一步下一步

綠色格子代表目前 Window，即 nums[i:j]。

406 可以使用 Sliding Window 的數學原因：

當所有 `nums[x] > 0`：

\[ \begin{aligned} \text{加入右邊元素}&\Rightarrow \text{sum 增加}\\ \text{移除左邊元素}&\Rightarrow \text{sum 減少} \end{aligned} \]

所以可以有把握地移動指標，而不會漏掉更短的合法子陣列。

你提供的程式碼中：

```
while j < n and sum < s:    sum += nums[j]    j += 1
```

表示不斷擴張右邊界，直到總和達到 `s`。

```
if sum >= s:    minLength = min(minLength, j - i)sum -= nums[i]
```

表示記錄答案，然後縮小左邊界。

雖然有 `for` 與 `while` 兩層迴圈，但 `i` 與 `j` 都只會往右移動最多 n 次，所以整體是 O(n)，不是 O(n²)。

## 三、為什麼相同程式碼無法解 1507？

1507 允許出現負數，造成 Sliding Window 原本依賴的單調性消失。

我們用一個很簡單的反例：

```
nums = [-1, 2, -1]k = 2# 正確答案：1# 因為 nums[1:2] = [2]，總和 2
```

如果執行你原本的程式碼：

|i|j|sum（判斷時）|結果|
|---|---|---|---|
|0|3|0|0 < 2，找不到|
|1|3|1|1 < 2，找不到|
|2|3|-1|-1 < 2，找不到|

最後回傳 `-1`，但正確答案明明是 `1`。

原因是原本的程式碼會一直把 `j` 往右推，沒有找到答案就一路走到陣列尾端。然而在負數陣列中，右邊的 `-1` 會讓總和下降，因此不能因為較長的 Window 不合格，就認為裡面較短的 Window 也不合格。

例如：

```
[-1, 2, -1]  sum = 0  ❌

     [2,-1]  sum = 1  ❌

     [2]     sum = 2  ✅
```

對於 406，縮小 Window 只會讓總和下降；但對於 1507，縮小 Window 反而可能讓總和上升。

這正是兩題不能直接共用 Sliding Window 解法的根本原因。

## 四、1507 為什麼需要 Prefix Sum + Monotonic Deque？

### Step 1：Prefix Sum 將區間總和轉換為相減

定義：

\[ P[j]=\sum_{t=0}^{j-1}nums[t] \]

那麼：

\[ sum(nums[i:j])=P[j]-P[i] \]

題目要求：

\[ P[j]-P[i]\ge K \]

同時希望子陣列長度：

\[ j-i \]

越小越好。

換句話說，對每一個右端點 `j`，我們需要找到適合的左端點 `i`。

### Step 2：Deque 儲存值得保留的左端點

最重要的觀察是：

假設 `i1 < i2`，而且

\[ P[i_1]\ge P[i_2] \]

那麼對未來任何相同的右端點 `j` 而言，`i2` 都比 `i1` 更有優勢：

- `P[i2]` 更小或相等，所以更容易滿足總和 ≥ K。
    
- `i2` 更靠右，所以產生的子陣列更短。
    

因此 `i1` 可以直接淘汰。

這就產生了 Monotonic Deque：維持前綴和由小到大排列的候選索引佇列。

### Step 3：Deque 的兩個關鍵操作

|操作|判斷條件|原因|
|---|---|---|
|`popleft()`|`P[j] - P[q[0]] >= K`|找到合法子陣列，更新最短長度。這個起點對未來更右側的終點不可能得到更短答案|
|`pop()`|`P[j] <= P[q[-1]]`|新的起點更靠右、前綴和又更小或相等，舊起點已被支配|

兩個操作合起來，讓 Deque 裡只留下對未來有用的候選左端點。

## 五、1507 正確 Python 解法

```
from collections import dequefrom typing import Listclass Solution:    def minimum_size(self, nums: List[int], s: int) -> int:        if not nums:            return -1        n = len(nums)        prefix = [0] * (n + 1)        # Step 1: Prefix Sum        for i in range(n):            prefix[i + 1] = prefix[i] + nums[i]        # Step 2: Monotonic Deque        q = deque()        minLength = n + 1        for j in range(n + 1):            # 取得符合 sum >= s 的最短長度            while q and prefix[j] - prefix[q[0]] >= s:                i = q.popleft()                minLength = min(minLength, j - i)            # 維持 Prefix Sum 單調遞增            while q and prefix[j] <= prefix[q[-1]]:                q.pop()            # 儲存當前 Prefix Sum 的 index            q.append(j)        return -1 if minLength == n + 1 else minLength
```

### 實際執行 1507：`nums = [-1, 2, -1], K = 2`

Prefix Sum：

```
nums   = [-1,  2, -1]prefix = [ 0, -1,  1,  0]index  = [ 0,  1,  2,  3]
```

Monotonic Deque 執行過程

j = 0

P[0]

## 0

P[1]

## -1

P[2]

## 1

P[3]

## 0

Deque（儲存索引）

### [0]

最短長度

## ∞

將 index 0 加入 Deque，作為候選起點。

上一步下一步

綠色代表目前仍保留在 Deque 的候選起點。

注意 `Deque` 並不是直接存放原本 `nums` 的值，而是存放 `prefix` 的索引。

### 時間複雜度為什麼也是 O(n)？

雖然裡面有兩個 `while`：

```
while q and prefix[j] - prefix[q[0]] >= s:    ...while q and prefix[j] <= prefix[q[-1]]:    ...
```

每個索引最多：

- `append()` 一次。
    
- 被 `popleft()` 或 `pop()` 移除一次。
    

因此所有 Deque 操作總計仍然是 O(n)，加上建立 Prefix Sum 也是 O(n)。

總時間 O(n)，總空間 O(n)。

## 六、總結：面試如何快速判斷使用哪個演算法？

|判定條件|選擇|複雜度|經典例題|
|---|---|---|---|
|最短子陣列和 ≥ K，所有元素為正數或非負數|Sliding Window / Two Pointers|Time O(n), Space O(1)|LintCode 406 / LeetCode 209|
|最短子陣列和 ≥ K，可能有負數|Prefix Sum + Monotonic Deque|Time O(n), Space O(n)|LintCode 1507 / LeetCode 862|

你也可以直接記這個規則：

> 正數陣列 + 最短子陣列和 ≥ K → Sliding Window。
> 
> 包含負數 + 最短子陣列和 ≥ K → Prefix Sum + Monotonic Deque。

1507 的 Monotonic Deque 還和你前面研究的「非負前綴和（Non-Negative Prefix Sum）」有關聯：兩者都使用 Prefix Sum 與 Deque，但 1507 主要是維護有優勢的候選起點，而環形非負前綴和問題則主要使用滑動區間最小前綴和來檢查每個起點是否有效。
