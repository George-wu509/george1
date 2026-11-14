
包含每個查詢的最小區間 Minimum Interval to Include Each Query

LeetCode 1851 給你一個二維整數陣列 `intervals`，其中 `intervals[i] = [left_i, right_i]` 表示第 `i` 個區間開始於 `left_i` 、結束於 `right_i`（包含兩端點）。區間的**長度**定義為區間中包含的整數數目，更正式地表達為 `right_i - left_i + 1`。

再給你一個整數陣列 `queries`。第 `j` 個查詢的答案是滿足 `left_i <= queries[j] <= right_i` 的**長度最小區間 `i` 的長度**。如果不存在這樣的區間，那麼答案是 `-1`。

以陣列形式回傳對應查詢的所有答案。

**样例 1：** 输入：
```Python
intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]
```

输出：
```Python
[3,3,1,4]
```

解释： 查詢處理如下：

- `Query = 2`：區間 `[2,4]` 是包含 2 的最小區間，長度為 `4 - 2 + 1 = 3`。
- `Query = 3`：區間 `[2,4]` 是包含 3 的最小區間，長度為 `4 - 2 + 1 = 3`。
- `Query = 4`：區間 `[4,4]` 是包含 4 的最小區間，長度為 `4 - 4 + 1 = 1`。
- `Query = 5`：區間 `[3,6]` 是包含 5 的最小區間，長度為 `6 - 3 + 1 = 4`。

**样例 2：** 输入：
```Python
intervals = [[2,3],[2,5],[1,8],[20,25]], queries = [2,19,5,22]
```

输出：
```Python
[2,-1,4,6]
```

解释： 查詢處理如下：

- `Query = 2`：區間 `[2,3]` 是包含 2 的最小區間，長度為 `3 - 2 + 1 = 2`。
- `Query = 19`：不存在包含 19 的區間，答案為 `-1`。
- `Query = 5`：區間 `[2,5]` 是包含 5 的最小區間，長度為 `5 - 2 + 1 = 4`。
- `Query = 22`：區間 `[20,25]` 是包含 22 的最小區間，長度為 `25 - 20 + 1 = 6`。

**样例 3：** 输入：
```Python
intervals = [[1,5],[1,5]], queries = [1,5,6]
```

输出：
```Python
[5,5,-1]
```

解释： 區間邊界為閉區間（包含端點 `1` 與 `5`），且允許存在完全相同的重複區間；當查詢值 `6` 超出所有區間右端點時回傳 `-1`。

```Python
# 方法一：離線查詢 + 最小堆（Sorting + Min-Heap）- 面試最佳解
import heapq
from typing import List

class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        # 1. 將區間依照「左端點 left」由小到大排序
        intervals.sort()
        
        minHeap = []  # 存放 (區間長度, 右端點 right)，依長度由小到大排序
        res = {}      # 用雜湊表記錄每個查詢值的答案（同時自動處理重複查詢）
        i = 0         # 指向 intervals 的掃描指標
        
        # 2. 將查詢按數值由小到大排序進行「離線處理」
        for q in sorted(queries):
            # 3. 將所有「左端點 <= q」的區間加入最小堆（代表這些區間已經開始覆蓋到 q）
            while i < len(intervals) and intervals[i][0] <= q:
                l, r = intervals[i]
                heapq.heappush(minHeap, (r - l + 1, r))
                i += 1
            
            # 4. 惰性刪除：將堆頂中「右端點 < q」的過期區間彈出（它們已經無法覆蓋 q 及後續更大的查詢）
            while minHeap and minHeap[0][1] < q:
                heapq.heappop(minHeap)
            
            # 5. 此時堆頂即為「包含 q 且長度最小」的合法區間
            res[q] = minHeap[0][0] if minHeap else -1
            
        # 6. 依照原始 queries 的順序還原答案
        return [res[q] for q in queries]
```


```Python
# 方法二：區間按長度排序 + 並查集（Union-Find 跳躍染色法）
from bisect import bisect_left, bisect_right
from typing import List

class Solution:
    def minInterval(self, intervals: List[List[int]], queries: List[int]) -> List[int]:
        # 1. 將 queries 排序並去重，建立查詢座標軸
        sorted_q = sorted(set(queries))
        m = len(sorted_q)
        # parent[idx] 指向下一個「尚未被填上答案」的查詢索引
        parent = list(range(m + 1))

        def find(x: int) -> int:
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        # 2. 將區間依照「長度 (r - l + 1)」由小到大排序（越短的區間優先佔領查詢點）
        intervals.sort(key=lambda x: x[1] - x[0])
        ans_map = {q: -1 for q in sorted_q}

        # 3. 遍歷每個區間，利用並查集直接跳過已經被更短區間覆蓋的查詢點
        for l, r in intervals:
            left_idx = bisect_left(sorted_q, l)
            right_idx = bisect_right(sorted_q, r) - 1
            idx = find(left_idx)
            while idx <= right_idx:
                ans_map[sorted_q[idx]] = r - l + 1
                parent[idx] = find(idx + 1)  # 標記該查詢點已找到最小區間，指向下一個未處理點
                idx = find(idx)

        return [ans_map[q] for q in queries]
```

解釋: step1: 題目不要求即時互動（Online），因此我們可以採用離線查詢（Offline Query）策略：把 `intervals` 按左端點升序排列，同時把 `queries` 也由小到大排序，讓查詢點像一條「掃描線」從左往右推進。 step2: 對於當前的查詢值 `q`，所有滿足 `left <= q` 的區間都有資格包含 `q`。用指標 `i` 單調往右掃描，把符合條件的區間以 `(區間長度, 右端點)` 的格式推入 `minHeap`。 step3: 因為 `minHeap` 依照第一個元素「區間長度」排序，堆頂永遠是目前候選區間中最短的那一個。 step4: 檢查堆頂區間是否已經過期：如果堆頂的 `right < q`，代表這個區間在 `q` 的左邊就已經結束了，無法覆蓋 `q`（也絕對無法覆蓋之後更大的 `q`），直接 `heappop` 丟掉，直到堆頂合法為止。 step5: 經過「加入左端點合格者」與「剔除右端點過期者」兩步過濾後，留在堆頂的元素一定同時滿足 `left <= q <= right` 且長度最小，直接記錄 `res[q] = minHeap[0][0]`。

`intervals = [[1,4],[2,4],[3,6],[4,4]], queries = [2,3,4,5]` -> `result = [3,3,1,4]`

1. `q = 2`：推入 `[1,4]`（長 4）與 `[2,4]`（長 3）-> 堆頂 `(3, 4)` 合法 -> `res[2] = 3`
    
2. `q = 3`：推入 `[3,6]`（長 4）-> 堆頂仍為 `(3, 4)` 合法 -> `res[3] = 3`
    
3. `q = 4`：推入 `[4,4]`（長 1）-> 堆頂變為 `(1, 4)` 合法 -> `res[4] = 1`
    
4. `q = 5`：無新區間可推入；檢查堆頂 `(1, 4)`、 `(3, 4)`、`(4, 4)` 的右端點皆小於 5，全部彈出 -> 堆頂剩 `(4, 6)` 合法 -> `res[5] = 4`
    

解釋: 如果對每個查詢都暴力掃描所有區間，時間複雜度是 `O(N × Q)` 會超時。

透過「雙重排序 + 最小堆惰性刪除」，每個區間最多只會進堆一次、出堆一次，將查詢過程均攤為線性掃描與對數級堆操作。

### **LeetCode 1851 - Minimum Interval to Include Each Query（包含每個查詢的最小區間）**

這道題目是 **離線查詢（Offline Query）**、**掃描線思想（Sweep Line）** 與 **最小堆（Min-Heap）** 結合的經典 Hard 題（NeetCode 150 區間篇壓軸題）。掌握這題的「單調指標入堆 + 過期元素惰性出堆」模板，就能解決一大類動態區間查詢問題。

## **解法解析**

這道題如果按照 `queries` 原本的亂序逐一處理，每次都得重新檢查所有區間；一旦我們打破順序限制，把查詢由小到大排列，問題就會出現強烈的單調性。

### **思路**

1. **為什麼要「排序 + 離線處理」？**
    
    - 一個區間 `[left, right]` 能包含查詢點 `q`，必須同時滿足兩個條件：`left <= q` 且 `right >= q`。
        
    - 当我們將 `intervals` 按 `left` 升序排列，並將 `queries` 升序排列後：
        
        - 隨著 `q` 越來越大，滿足 `left <= q` 的區間只會**越來越多**（指標 `i` 只需單向往右推進，絕不回頭）。
            
        - 同時，一旦某個區間的 `right < q`，它在當前這一步失效了，在未來所有更大的 `q'` 中也**永遠不可能再有效**，因此可以永久刪除。
            
2. **為什麼最小堆只需要檢查「堆頂」是否過期（惰性刪除 Lazy Deletion）？**
    
    - 堆裡面可能混雜著其他已經過期、但還沒浮到堆頂的較長區間，我們需要特地去堆中間把它們挖出來刪掉嗎？
        
    - **完全不需要！** 因為我們只在乎「長度最小」的那個合法區間。只要堆頂元素的 `right >= q`，它就一定是所有合法區間中最短的；那些藏在堆底的過期長區間根本不影響答案，等未來它們浮到堆頂時再順手 `heappop` 掉即可。
        
3. **代碼執行流程**
    
    - 對 `intervals` 執行 `intervals.sort()`。
        
    - 使用雜湊表 `res = {}` 儲存每個不重複查詢值的結果，避免題目給出重複查詢（例如 `queries = [4, 4, 4]`）時重複計算。
        
    - 遍歷 `sorted(queries)`，先用 `while` 把左端點合格的區間壓入 `minHeap`，再用 `while` 把堆頂右端點過期的區間彈出，最後讀取 `minHeap[0][0]`。
        

## **具體舉例**

假設 `intervals = [[2,3], [2,5], [1,8], [20,25]]`，`queries = [2, 19, 5, 22]`。 先排序得到：

- `intervals = [[1,8], [2,3], [2,5], [20,25]]`
    
- `sorted(queries) = [2, 5, 19, 22]`
    

掃描線推進過程如下：

|**當前 q**|**入堆區間 (left <= q)**|**入堆後 minHeap (長度, 右端點)**|**出堆過期區間 (right < q)**|**最終 minHeap**|**res[q]**|
|---|---|---|---|---|---|
|`2`|`[1,8]`, `[2,3]`, `[2,5]`|`[(2,3), (8,8), (4,5)]`|無（堆頂 `3 >= 2`）|`[(2,3), (8,8), (4,5)]`|`2`|
|`5`|無|`[(2,3), (8,8), (4,5)]`|彈出 `(2,3)`（因為 `3 < 5`）|`[(4,5), (8,8)]`|`4`|
|`19`|無|`[(4,5), (8,8)]`|彈出 `(4,5)` 與 `(8,8)`（皆 `< 19`）|`[]`|`-1`|
|`22`|`[20,25]`|`[(6,25)]`|無（堆頂 `25 >= 22`）|`[(6,25)]`|`6`|

最後按照原始查詢順序 `[2, 19, 5, 22]` 查表還原，回傳 `[2, -1, 4, 6]`。

## **時間與空間複雜度分析**

設 `N` 為 `intervals` 的長度，`Q` 為 `queries` 的長度。

- **時間複雜度：O(N log N + Q log Q)**
    
    - 對 `intervals` 排序需要 `O(N log N)`，對 `queries` 排序需要 `O(Q log Q)`。
        
    - 在雙重 `while` 迴圈中，每個區間最多被 `heappush` 一次、`heappop` 一次，單次堆積操作時間為 `O(log N)`，因此所有堆操作總耗時為 `O(N log N)`。
        
    - 總時間複雜度為 **O(N log N + Q log Q)**。
        
- **空間複雜度：O(N + Q)**
    
    - `minHeap` 最多同時儲存 `N` 個區間，佔用 `O(N)` 空間。
        
    - 雜湊表 `res` 與排序後的查詢陣列佔用 `O(Q)` 空間。
        

## **其他解法**

### **1. 區間按長度排序 + 並查集（Union-Find 跳躍覆蓋法）**

- **思路**：
    
    - 反過來從「區間長度」出發：先把所有區間依照長度 `r - l + 1` **由小到大排序**。
        
    - 這代表越短的區間優先權越高——一旦某個查詢點 `q` 被較短的區間覆蓋了，之後任何較長的區間都**不可能提供更好的答案**。
        
    - 將所有查詢點排序並去重，利用二分搜尋（`bisect_left` / `bisect_right`）定位區間 `[l, r]` 涵蓋的查詢索引範圍，並透過**並查集（Union-Find）的 `parent` 陣列**把已經填過答案的查詢點串聯起來，實現 `O(1)` 均攤跳過已處理的點。
        
- **時間複雜度**：`O(N log N + Q log Q)`（每個查詢點最多被賦值並合併一次）
    
- **空間複雜度**：`O(Q)`（並查集大小僅與查詢數量 `Q` 有關）
    

### **2. 暴力法（Brute Force）**

- **思路**：
    
    - 雙層迴圈，外層遍歷每個 `query`，內層遍歷所有 `intervals`，檢查 `left <= query <= right` 並取最小長度。
        
- **時間複雜度**：`O(N × Q)`（當 `N, Q <= 10^5` 時會超過時間限制 TLE）
    
- **空間複雜度**：`O(1)`
    

## **總結**

- **面試最佳解**：**離線查詢 + 最小堆（Sorting + Min-Heap）**，思路清晰、程式碼簡短且不易寫錯，時間複雜度 `O(N log N + Q log Q)`。
    
- **核心記憶口訣**：**「區間按左排、查詢升序來；左合格入堆、右過期惰性刪」**。
    
- **常見陷阱**：
    
    1. **忘記還原原始查詢順序**：題目要求回傳的陣列必須對應原本 `queries` 的順序，直接遍歷 `sorted(queries)` 收集結果會導致順序錯亂，務必透過雜湊表 `res[q]` 映射或連同原始索引 `(q, idx)` 一起排序。
        
    2. **重複查詢的處理**：如果選擇儲存 `(q, idx)` 而非雜湊表 `res[q]`，遇到兩個相同的 `q` 時，第一個 `q` 已經把合格區間入堆、過期區間彈出，第二個相同的 `q` 會直接讀取同一個 `minHeap[0]`，兩者都能正確運作，但使用 `res[q]` 字典更簡潔。
        

## **區間查詢與離線掃描線（Intervals / Heap / Sweep Line）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 1851**|Minimum Interval to Include Each Query|查詢包含點 `q` 的**最小長度區間**|**離線查詢 + 區間按左端點排序 + Min-Heap 維護長度與右端點**|
|**LeetCode 2251**|Number of Flowers in Full Bloom（花期內花的數目）|查詢時間點 `t` 同時盛開的**區間總數**|**離線查詢 + Min-Heap**，或 **起點陣列與終點陣列分別二分搜尋**|
|**LeetCode 253**|Meeting Rooms II（會議室 II）|計算所有區間中**最大同時重疊數**（最少會議室）|**按開始時間排序 + Min-Heap 維護最早結束時間**，或差分掃描線|
|**LeetCode 56**|Merge Intervals（合併區間）|將所有重疊的區間合併成不重疊區間列表|**按左端點 `left` 升序排列**，比較當前左端點與上一個右端點|
|**LeetCode 57**|Insert Interval（插入區間）|在已排序且無重疊的區間列表中插入新區間並合併|**三段式線性掃描**（左側無重疊、中間重疊取最大聯集、右側無重疊）|
|**LeetCode 435**|Non-overlapping Intervals（無重疊區間）|移除最少區間數量使剩餘區間互不重疊|**貪心按右端點 `right` 升序排列**，優先保留結束早的區間|
|**LeetCode 759**|Employee Free Time（員工空閒時間）|找出所有員工工作區間聯集之外的**共同空閒區間**|**展開所有區間按左端點排序合併（同 LC 56）**，相鄰合併區間的縫隙即答案|
|**LeetCode 218**|The Skyline Problem（天際線問題）|求所有建築物區間輪廓的**最高外邊界關鍵點**|**掃描線事件排序 + Max-Heap 惰性刪除（或 SortedList）**|

### **進階延伸題：花期內花的數目（LeetCode 2251）**

這題與 `LC 1851` 幾乎是雙胞胎題型！差別只在於 `LC 1851` 要找覆蓋查詢點 `q` 的「最小區間長度」**，而 `LC 2251` 要找覆蓋查詢點 `person` 的**「有效區間個數」（用完全相同的離線查詢 + Min-Heap 模板即可秒殺）：

Python

```
import heapq
from typing import List

class Solution:
    def fullBloomFlowers(self, flowers: List[List[int]], people: List[int]) -> List[int]:
        # 1. 花期區間按開始時間排序
        flowers.sort()
        minHeap = []  # 存放正在盛開的花的「結束時間 end」
        res = {}
        i = 0
        
        # 2. 離線處理：訪客按到達時間由小到大排序
        for p in sorted(people):
            # 將所有「開始時間 <= p」的花加入堆中
            while i < len(flowers) and flowers[i][0] <= p:
                heapq.heappush(minHeap, flowers[i][1])
                i += 1
            
            # 惰性刪除：將「結束時間 < p」（已經凋謝）的花彈出
            while minHeap and minHeap[0] < p:
                heapq.heappop(minHeap)
                
            # 此時堆中剩下的元素數量，就是當前時間點 p 正在盛開的花數！
            res[p] = len(minHeap)
            
        return [res[p] for p in people]
```