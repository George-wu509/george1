
資料串流中的第 K 大元素 Kth Largest Element in a Stream

LeetCode 703 設計一個找到資料串流（Data Stream）中第 `k` 大元素的類別（Class）。請注意，它是排序後的第 `k` 大元素，而**不是**第 `k` 個不同的元素。

請實作 `KthLargest` 類別：

- `KthLargest(int k, int[] nums)` 使用整數 `k` 和整數串流 `nums` 初始化物件。
- `int add(int val)` 將整數 `val` 插入資料串流 `nums` 中，並回傳目前資料串流中第 `k` 大的元素。


_(註：題目保證每次呼叫 `add` 時，資料串流中至少已經有 `k` 個元素。)_

**样例 1：** 输入：
```Python
["KthLargest", "add", "add", "add", "add", "add"]
[[3, [4, 5, 8, 2]], [3], [5], [10], [9], [4]]
```

输出：
```Python
[null, 4, 5, 5, 8, 8]
```

解释：
```Python
kthLargest = KthLargest(3, [4, 5, 8, 2])
kthLargest.add(3)   # 串流為 [2, 3, 4, 5, 8]，第 3 大為 4
kthLargest.add(5)   # 串流為 [2, 3, 4, 5, 5, 8]，第 3 大為 5
kthLargest.add(10)  # 串流為 [2, 3, 4, 5, 5, 8, 10]，第 3 大為 5
kthLargest.add(9)   # 串流為 [2, 3, 4, 5, 5, 8, 9, 10]，第 3 大為 8
kthLargest.add(4)   # 串流為 [2, 3, 4, 4, 5, 5, 8, 9, 10]，第 3 大為 8
```

**样例 2：** 输入：
```Python
["KthLargest", "add", "add", "add"]
[[1, []], [-3], [-2], [-4]]
```

输出：
```Python
[null, -3, -2, -2]
```

解释： 初始陣列為空，要找第 `1` 大元素：

- 加入 `-3` -> 串流 `[-3]`，第 1 大為 `-3`
- 加入 `-2` -> 串流 `[-3, -2]`，第 1 大為 `-2`
- 加入 `-4` -> 串流 `[-4, -3, -2]`，第 1 大仍為 `-2`

**样例 3：** 输入：
```Python
["KthLargest", "add", "add"]
[[2, [5, 5]], [5], [6]]
```

输出：
```Python
[null, 5, 5]
```

解释： 重複元素也算入排名：加入 `5` 後為 `[5, 5, 5]`（第 2 大為 `5`）；加入 `6` 後為 `[5, 5, 5, 6]`（第 2 大仍為 `5`）。

```Python
# 方法一：固定大小為 K 的最小堆（Min-Heap）- 面試最佳解
import heapq
from typing import List

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        self.minHeap = nums
        # 1. 原地將 nums 轉換為最小堆，時間 O(N)
        heapq.heapify(self.minHeap)
        # 2. 只保留「最大的 K 個元素」，把其餘較小的元素全部彈出淘汰
        while len(self.minHeap) > k:
            heapq.heappop(self.minHeap)

    def add(self, val: int) -> int:
        # 3. 將新元素推入堆中
        heapq.heappush(self.minHeap, val)
        # 4. 若元素超過 K 個，彈出最小值（即第 K+1 大的元素）
        if len(self.minHeap) > self.k:
            heapq.heappop(self.minHeap)
        # 5. 此時堆中留下的恰好是全域前 K 大元素，堆頂（最小值）就是「第 K 大」
        return self.minHeap[0]
```

```Python
# 方法二：有序陣列 + 二分搜尋（Binary Search / bisect）
import bisect
from typing import List

class KthLargest:
    def __init__(self, k: int, nums: List[int]):
        self.k = k
        # 先將初始陣列由小到大排序
        self.stream = sorted(nums)

    def add(self, val: int) -> int:
        # 利用二分搜尋找到插入位置並保持有序，查找 O(log N)，插入 O(N)
        bisect.insort(self.stream, val)
        # 由小到大排序後，倒數第 k 個元素即為「第 k 大」
        return self.stream[-self.k]
```

解釋: step1: 反向思考「找第 K 大」：如果我們手上只留下**目前為止最大的 K 個數字**，那麼這 K 個數字裡面**最小的那一個**，就是全部數字中的「第 K 大」。 step2: Python 的 `heapq` 預設就是**最小堆（Min-Heap）**，堆頂 `minHeap[0]` 永遠是堆裡面的最小值，剛好完美符合「快速取得前 K 大之中的最小值」。 step3: 在 `__init__` 初始化時，用 `heapq.heapify(nums)` 建堆，並用 `while len(self.minHeap) > k:` 把多餘的較小元素全部 `heappop` 丟掉，確保堆的大小不超過 `k`。 step4: 每次呼叫 `add(val)` 時，先把 `val` 塞進 `minHeap`；如果塞完後長度變成 `k + 1`，就把新的最小值 `heappop` 淘汰掉。 step5: 最後直接回傳 `self.minHeap[0]` 即可（不需要把堆頂彈出，因為之後還要繼續比較）。

`k = 3, nums = [4, 5, 8, 2]`，依序呼叫 `add(3)`, `add(5)`, `add(10)` -> 依序回傳 `4, 5, 5`

1. `__init__`：`[4, 5, 8, 2]` 建堆後彈出最小的 `2` -> `minHeap = [4, 5, 8]`（保留前 3 大，堆頂為 `4`）
    
2. `add(3)`：推入 `3` 變成 `[3, 4, 5, 8]` -> 超過 3 個，彈出最小值 `3` -> `minHeap = [4, 5, 8]` -> 回傳堆頂 `4`
    
3. `add(5)`：推入 `5` 變成 `[4, 5, 5, 8]` -> 超過 3 個，彈出最小值 `4` -> `minHeap = [5, 5, 8]` -> 回傳堆頂 `5`
    
4. `add(10)`：推入 `10` 變成 `[5, 5, 8, 10]` -> 超過 3 個，彈出最小值 `5` -> `minHeap = [5, 8, 10]` -> 回傳堆頂 `5`
    

解釋: 資料串流（Stream）的特性是**資料源源不絕進來、無法預知總長度，且只會新增、不會刪除**。

任何比「目前第 K 大」還要小的數字，永遠不可能在未來變成第 K 大，因此**維護一個容量上限為 `K` 的最小堆作為「前 K 大榮譽榜」**，就能直接丟棄所有歷史包袱，每次查詢都是 `O(1)`。

### **LeetCode 703 - Kth Largest Element in a Stream（資料串流中的第 K 大元素）**

這道題目是 **最小堆（Min-Heap / Priority Queue）** 應用的入門經典題（NeetCode 150 必考題）。許多初學者看到「求第 K **大**」會直覺想用「最大堆（Max-Heap）」，但這題恰恰相反——**求第 K 大，要用「最小堆」**。

## **解法解析**

這道題的核心在於理解「為什麼求第 K 大反而要用最小堆」**以及**「如何處理無限增長的資料串流」。

### **思路**

1. **為什麼不用最大堆（Max-Heap）？**
    
    - 如果把所有元素都放進最大堆，雖然堆頂是第 `1` 大，但要拿到第 `k` 大，每次 `add()` 都必須連續 `heappop` 彈出 `k - 1` 次，讀完之後還要再全部 `heappush` 塞回去，單次時間複雜度高達 `O(k log N)`，且空間會隨資料流無限膨脹到 `O(N)`。
        
2. **「前 K 大守門員」模型（大小為 `k` 的最小堆）**
    
    - 想像一場比賽只錄取前 `k` 名（排行榜只有 `k` 個名額）。
        
    - 排行榜的最後一名（第 `k` 名）就是錄取門檻！
        
    - 只要維護一個大小為 `k` 的**最小堆**，堆頂 `minHeap[0]` 就是這 `k` 個強者中的最弱者（即第 `k` 大）。
        
    - 當新選手 `val` 進場時：
        
        - 如果 `val` 比門檻還弱（`val <= minHeap[0]`），他連排行榜都進不去，直接被淘汰。
            
        - 如果 `val` 比門檻強（`val > minHeap[0]`），他擠進排行榜，原本的最後一名 `minHeap[0]` 被踢出，新的最後一名成為新的第 `k` 大。
            
3. **代碼執行流程**
    
    - **初始化 `__init__`**：將初始陣列 `nums` 原地 `heapify` 成最小堆，並用 `while len(self.minHeap) > k` 修剪至最多只剩 `k` 個元素。
        
    - **新增元素 `add(val)`**：執行 `heappush(self.minHeap, val)`；若堆大小超過 `k`，則執行 `heappop(self.minHeap)` 移除最小值，最後回傳 `self.minHeap[0]`。
        

## **具體舉例**

假設初始化 `k = 3`, `nums = [4, 5, 8, 2]`，接著依序呼叫 `add(3)`, `add(5)`, `add(10)`, `add(9)`, `add(4)`，內部狀態變化如下：

|**操作步驟**|**推入後的堆元素（概念排序）**|**是否超過 k=3 並淘汰最小值**|**最終 minHeap（前 3 大）**|**回傳值 minHeap[0]**|
|---|---|---|---|---|
|`__init__(3, [4,5,8,2])`|`[2, 4, 5, 8]`|淘汰 `2`|`[4, 5, 8]`|`null`|
|`add(3)`|`[3, 4, 5, 8]`|淘汰 `3`|`[4, 5, 8]`|`4`|
|`add(5)`|`[4, 5, 5, 8]`|淘汰 `4`|`[5, 5, 8]`|`5`|
|`add(10)`|`[5, 5, 8, 10]`|淘汰 `5`|`[5, 8, 10]`|`5`|
|`add(9)`|`[5, 8, 9, 10]`|淘汰 `5`|`[8, 9, 10]`|`8`|
|`add(4)`|`[4, 8, 9, 10]`|淘汰 `4`|`[8, 9, 10]`|`8`|

## **時間與空間複雜度分析**

設 `N` 為初始陣列 `nums` 的長度，`M` 為後續呼叫 `add()` 的次數，`k` 為目標排名。

- **初始化 `__init__`**：
    
    - **時間複雜度：O(N log N)**（精確為 `O(N + (N - k) log N)`）
        
        - `heapq.heapify(nums)` 需要 `O(N)` 時間；若 `N > k`，最多需彈出 `N - k` 次元素，每次 `heappop` 為 `O(log N)`。
            
- **單次新增 `add(val)`**：
    
    - **時間複雜度：O(log k)**
        
        - 堆的大小永遠被限制在最多 `k + 1` 個元素，因此 `heappush` 和 `heappop` 的時間複雜度嚴格控制在 **O(log k)**（與資料串流總長度無關！），讀取堆頂 `minHeap[0]` 為 `O(1)`。
            
- **空間複雜度：O(k)**
    
    - 堆中最多只保留 `k` 個（push 瞬間 `k + 1` 個）元素，其餘歷史資料全部釋放，非常適合處理海量資料串流。
        

## **其他解法**

### **1. `heapq.heapreplace` 優化版（微調常數時間）**

- **思路**：
    
    - 在 `add(val)` 中，當 `len(self.minHeap) == self.k` 且 `val > self.minHeap[0]` 時，使用 `heapq.heapreplace(self.minHeap, val)`（先彈出堆頂再推入新值，僅需執行一次向下調整 `sift-down`），省去先 `heappush` 再 `heappop` 的兩次堆調整開銷。
        
- **時間複雜度**：單次 `add` 為 `O(log k)`（實際執行速度比 push + pop 快約一倍）
    
- **空間複雜度**：`O(k)`
    

### **2. 有序陣列 + 二分搜尋（`bisect.insort`）**

- **思路**：
    
    - 初始化時將 `nums` 排序，每次 `add(val)` 用二分搜尋找到插入點並插入陣列，直接回傳倒數第 `k` 個元素 `stream[-k]`。
        
- **時間複雜度**：單次 `add` 為 `O(N + M)`（雖然二分查找只要 `O(log N)`，但陣列中間插入元素需要搬移記憶體 `O(N)`，且空間會隨資料流無限增長至 `O(N + M)`，效能遠不如最小堆）。
    
- **空間複雜度**：`O(N + M)`
    

## **總結**

- **面試最佳解**：**大小為 `K` 的最小堆（Min-Heap）**，時間 `O(log k)`，空間 `O(k)`。
    
- **核心記憶口訣**：**「找第 K 大，用最小堆；找第 K 小，用最大堆」**（永遠用相反性質的堆來當門檻守門員）。
    
- **常見陷阱**：初始化時 `nums` 的長度可能**小於 `k`**（例如範例 2 中 `k = 1, nums = []`），因此 `add(val)` 中必須寫 `if len(self.minHeap) > self.k:` 來判斷是否彈出，而不能無條件直接 `heappop`。
    

## **Top-K 與資料串流（Heap / QuickSelect）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 703**|Kth Largest Element in a Stream|**動態資料串流**中持續查詢第 `K` 大元素|**固定大小為 `K` 的 Min-Heap**，單次更新 `O(log K)` 空間 `O(K)`|
|**LeetCode 215**|Kth Largest Element in an Array|**靜態無序陣列**中找出第 `K` 大元素|**快速選擇（QuickSelect）平均 `O(N)`**，或大小為 `K` 的 Min-Heap `O(N log K)`|
|**LeetCode 347**|Top K Frequent Elements（前 K 個高頻元素）|找出陣列中出現頻率最高的 `K` 個元素|**桶排序（Bucket Sort）最佳 `O(N)`**，或 HashMap + 大小為 `K` 的 Min-Heap|
|**LeetCode 973**|K Closest Points to Origin（最接近原點的 K 個點）|找出距離原點 `(0, 0)` 最近的 `K` 個座標|**大小為 `K` 的 Max-Heap**（淘汰距離較遠的點），時間 `O(N log K)`|
|**LeetCode 295**|Find Median from Data Stream（資料串流的中位數）|**動態資料串流**中隨時查詢中位數（動態的第 `N/2` 大）|**對頂堆（Max-Heap 存較小一半 + Min-Heap 存較大一半）**，查詢 `O(1)`|
|**LeetCode 1046**|Last Stone Weight（最後一塊石頭的重量）|每次取出**最重的兩塊石頭**相撞，模擬直到剩一塊|**Max-Heap 模擬**，每次彈出兩個最大值並推回差值，時間 `O(N log N)`|
|**LeetCode 378**|Kth Smallest Element in a Sorted Matrix|行列皆遞增的 `N × N` 矩陣中找第 `K` 小元素|**值域二分搜尋 `O(N log(max-min))`**，或 Min-Heap 多路歸併 `O(K log N)`|
|**LeetCode 692**|Top K Frequent Words（前 K 個高頻單字）|找出頻率最高的 `K` 個單字（同頻率按字典序排列）|**HashMap + 自訂比較器的 Min-Heap**，時間 `O(N log K)`|

### **進階延伸題：陣列中的第 K 個最大元素（LeetCode 215）**

如果題目不是「資料串流」，而是一次性給你完整的靜態陣列 `nums` 求第 `k` 大元素，除了用同樣的 `Min-Heap` 之外，面試官通常還會追問如何在 **`O(N)` 平均時間**內解決（即 **QuickSelect 快速選擇演算法**）：

Python

```
import random
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        # 第 k 大元素，在由小到大排序後的目標索引為 len(nums) - k
        target_idx = len(nums) - k

        def quickselect(left: int, right: int) -> int:
            # 隨機挑選基準值 pivot，避免最壞情況 O(N^2)
            pivot_idx = random.randint(left, right)
            pivot = nums[pivot_idx]
            nums[pivot_idx], nums[right] = nums[right], nums[pivot_idx]

            # 將所有小於 pivot 的元素移到左半邊
            store_idx = left
            for i in range(left, right):
                if nums[i] < pivot:
                    nums[store_idx], nums[i] = nums[i], nums[store_idx]
                    store_idx += 1
            
            # 將 pivot 歸位到最終正確排名位置
            nums[store_idx], nums[right] = nums[right], nums[store_idx]

            # 根據 pivot 位置決定往左還是往右縮小範圍
            if store_idx == target_idx:
                return nums[store_idx]
            elif store_idx < target_idx:
                return quickselect(store_idx + 1, right)
            else:
                return quickselect(left, store_idx - 1)

        return quickselect(0, len(nums) - 1)
```