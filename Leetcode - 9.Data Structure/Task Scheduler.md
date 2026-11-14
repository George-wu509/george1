
任務排程器 Task Scheduler

LeetCode 621 (LintCode 945) 給你一個用字元陣列 `tasks` 表示的 CPU 需要執行的任務列表，其中每個字母表示一種不同種類的任務。任務可以以**任意順序**執行，並且每個任務都可以在 **1 個單位時間**內執行完。在任何一個單位時間，CPU 可以完成一個任務，或者處於待命（Idle）狀態。

然而，兩個**相同種類**的任務之間必須有長度為整數 `n` 的**冷卻時間**，因此至少有連續 `n` 個單位時間 CPU 在執行不同的任務，或者在待命狀態。

你需要計算完成所有任務所需要的**最短時間**。

**样例 1：** 输入：
```Python
tasks = ["A","A","A","B","B","B"], n = 2
```

输出：
```Python
8
```

解释： 一種可能的執行順序為：`A -> B -> (待命) -> A -> B -> (待命) -> A -> B` 在本範例中，兩個相同類型的任務之間必須間隔長度為 `n = 2` 的冷卻時間，因此至少需要 8 個單位時間。

**样例 2：** 输入：
```Python
tasks = ["A","C","A","B","D","B"], n = 1
```

输出：
```Python
6
```

解释： 一種可能的執行順序為：`A -> B -> C -> D -> A -> B` 由於 `n = 1`，只要任意兩個相同任務不緊鄰即可，所有任務皆可無縫穿插，無需任何待命時間，共 6 個單位時間。

**样例 3：** 输入：
```Python
tasks = ["A","A","A","B","B","B"], n = 3
```

输出：

```Python
10
```

解释： 一種可能的執行順序為：`A -> B -> (待命) -> (待命) -> A -> B -> (待命) -> (待命) -> A -> B` 共需要 10 個單位時間。

```Python
# 方法一：貪心公式計算（桶子思想 / 數學最佳解）- 時間 O(N)，空間 O(1)
from collections import Counter
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        # 1. 統計每種任務出現的次數
        counts = Counter(tasks)
        max_freq = max(counts.values())  # 出現次數最多的任務頻率
        
        # 2. 計算有多少種任務並列為「最高頻率」
        max_count = sum(1 for freq in counts.values() if freq == max_freq)
        
        # 3. 套用公式：(最高頻率 - 1) 組完整區塊 * 每組長度 (n + 1) + 最後一排並列最高頻的任務數
        # 若其他零散任務太多，把空格全塞滿還溢出來，答案就是 len(tasks) 本身
        formula_time = (max_freq - 1) * (n + 1) + max_count
        return max(len(tasks), formula_time)
```


```Python
# 方法二：Max-Heap + 佇列（模擬法）- 直觀好理解，時間 O(N)，空間 O(1)
import heapq
from collections import Counter, deque
from typing import List

class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counts = Counter(tasks)
        # Python 只有最小堆，將頻率轉負數模擬最大堆（每次優先執行剩餘次數最多的任務）
        maxHeap = [-cnt for cnt in counts.values()]
        heapq.heapify(maxHeap)
        
        time = 0
        q = deque()  # 存放正在冷卻中的任務：[剩餘負頻率, 解鎖可用時間點]
        
        while maxHeap or q:
            time += 1
            if maxHeap:
                # 取出目前剩餘次數最多的任務執行（負數 + 1 代表剩餘次數減 1）
                cnt = 1 + heapq.heappop(maxHeap)
                if cnt != 0:
                    # 還沒做完，放進冷卻佇列，在 time + n 時刻才能重新執行
                    q.append([cnt, time + n])
            
            # 檢查冷卻佇列最前端的任務是否已經冷卻完畢
            if q and q[0][1] == time:
                heapq.heappush(maxHeap, q.popleft()[0])
                
        return time
```

解釋: step1: 核心貪心策略：**永遠優先安排「剩餘次數最多」的任務**，因為它們最容易受冷卻時間限制、最難消化完。 step2: 統計所有任務頻率，找出最大頻率 `max_freq`（決定總共需要切分出幾個時間區塊）。 step3: 先排好前 `max_freq - 1` 輪，每一輪的固定寬度為 `n + 1`（包含 1 個最高頻任務本身 + `n` 個冷卻空位）。 step4: 最後一輪不需要再補後面的冷卻時間，只需要放上所有頻率等於 `max_freq` 的任務，共 `max_count` 個，得出公式 `(max_freq - 1) * (n + 1) + max_count`。 step5: 若任務種類極多、冷卻時間 `n` 很短，把所有冷卻空位塞滿後還有剩餘任務，此時完全不會產生 `Idle`（待命），最短時間直接等於任務總數 `len(tasks)`，故取 `max(len(tasks), formula_time)`。

`tasks = ["A","A","A","B","B","B"], n = 2` -> `result = 8`

1. 統計頻率：`A: 3, B: 3`，所以 `max_freq = 3`
    
2. 並列最高頻的任務有 A 和 B 兩種，所以 `max_count = 2`
    
3. 前 `3 - 1 = 2` 輪架構：`[A, B, _] -> [A, B, _]`，佔用 `2 * (2 + 1) = 6` 單位時間
    
4. 最後一輪收尾：`[A, B]`，佔用 `max_count = 2` 單位時間
    
5. `formula_time = 6 + 2 = 8`，`max(6, 8) = 8`
    

解釋: 把最高頻任務當作「骨架」，每隔 `n` 個空位放一個。

其他任務負責去填補中間的 `Idle` 空位；**空位填不滿時，總時間由骨架決定；空位全填滿還有剩時，總時間就等於任務總數 `len(tasks)`**。

### **LeetCode 621 - Task Scheduler（任務排程器）**

這道題目是 **貪心演算法（Greedy）** 與 **優先佇列（Max-Heap）** 的經典代表題（Blind 75 & NeetCode 150 必考題）。面試中通常建議先講出「最大堆 + 冷卻佇列」的模擬思路展現工程直覺，再給出「桶子排列公式」的 `O(1)` 空間數學最佳解。

## **解法解析**

這道題的瓶頸永遠取決於**出現次數最多的任務**。只要把最高頻率的任務安排妥當，其餘低頻任務都能穿插進空隙中。

### **思路**

1. **建立「桶子（Bucket）」排列模型（貪心公式法）**
    
    - 假設任務 `A` 出現最多次，共有 `max_freq` 次，冷卻時間為 `n`。
        
    - 為了讓總時間最短，我們把這 `max_freq` 個 `A` 盡量緊密排列，但兩兩之間必須隔開 `n` 個位置：
        
        - 第 1 組：`A` + `[n 個空位]`
            
        - 第 2 組：`A` + `[n 個空位]`
            
        - ...
            
        - 第 `max_freq - 1` 組：`A` + `[n 個空位]`
            
        - 第 `max_freq` 組（最後一排）：`A`（後面不需要再等冷卻！）
            
    - 前 `max_freq - 1` 組共有 `(max_freq - 1)` 個區塊，每個區塊長度為 `n + 1`。
        
2. **處理並列最高頻的任務（`max_count`）**
    
    - 如果有另一種任務 `B` 的出現次數也等於 `max_freq`，那 `B` 也會跟著 `A` 一起出現在最後一排的尾巴。
        
    - 因此最後一排的長度就是所有頻率等於 `max_freq` 的任務種類數 `max_count`。
        
    - 由此推導出公式：`formula_time = (max_freq - 1) * (n + 1) + max_count`。
        
3. **為什麼要取 `max(len(tasks), formula_time)`？**
    
    - 當任務種類非常豐富時（例如 `tasks = ["A","A","B","B","C","C","D","D"]`, `n = 1`），`formula_time = (2 - 1) * (1 + 1) + 4 = 6`，但實際有 `8` 個任務。
        
    - 這代表中間的冷卻空位早就被塞爆了，我們只需要把多出來的任務往每組後面繼續接下去（`[A, B, C, D] -> [A, B, C, D]`），完全不會破壞冷卻限制，且 **0 待命時間**，所以總時間就是 `len(tasks)`。
        

## **具體舉例**

假設 `tasks = ["A", "A", "A", "B", "B", "C"]`，`n = 2`，我們同時對照「方法二：Max-Heap + 冷卻佇列」在每個時間點的狀態變化：

- 初始頻率：`A: 3, B: 2, C: 1` → `maxHeap = [-3(A), -2(B), -1(C)]`
    

|**time**|**執行任務（Pop Max-Heap）**|**maxHeap 剩餘狀態**|**冷卻佇列 q [剩餘負頻率, 解鎖時間]**|**本輪結束時解鎖檢查**|
|---|---|---|---|---|
|`1`|執行 `A`（剩 2 次）|`[-2(B), -1(C)]`|`[[-2(A), 3]]`|無|
|`2`|執行 `B`（剩 1 次）|`[-1(C)]`|`[[-2(A), 3], [-1(B), 4]]`|無|
|`3`|執行 `C`（剩 0 次）|`[]`|`[[-2(A), 3], [-1(B), 4]]`|`A` 在 `time=3` 解鎖，推回 `maxHeap`|
|`4`|執行 `A`（剩 1 次）|`[]`|`[[-1(B), 4], [-1(A), 6]]`|`B` 在 `time=4` 解鎖，推回 `maxHeap`|
|`5`|執行 `B`（剩 0 次）|`[]`|`[[-1(A), 6]]`|無|
|`6`|**待命（Idle）**（堆為空）|`[]`|`[[-1(A), 6]]`|`A` 在 `time=6` 解鎖，推回 `maxHeap`|
|`7`|執行 `A`（剩 0 次）|`[]`|`[]`|堆與佇列皆空，結束！|

最終花費 `7` 個單位時間（驗證公式：`(3 - 1) * (2 + 1) + 1 = 7`，完全一致）。

## **時間與空間複雜度分析**

設 `N` 為 `tasks` 陣列的總長度，`K` 為不同任務的種類數（本題任務為大寫英文字母 `A-Z`，故 $K \le 26$）。

- **方法一（貪心公式法）**：
    
    - **時間複雜度：O(N)**
        
        - 遍歷一次 `tasks` 統計頻率需 `O(N)`，接著遍歷最多 26 個字母求最大值需 `O(K) = O(1)`。
            
    - **空間複雜度：O(1)**
        
        - 雜湊表 `Counter` 最多只儲存 26 個英文字母的計數，屬於常數空間 `O(1)`。
            
- **方法二（Max-Heap + 冷卻佇列模擬法）**：
    
    - **時間複雜度：O(N × n)**（在無待命情況下為 `O(N log K)`）
        
        - 每次堆操作 `heappush` / `heappop` 為 `O(log K)`，因為 $K \le 26$ 視為 `O(1)`；最壞情況下（全都是同一種任務，如 `["A","A","A"]`, `n = 100`），總模擬時間步數為 `O(N * n)`。
            
    - **空間複雜度：O(1)**
        
        - `maxHeap` 與 `deque` 最多同時存放 `K` 個不同種類的任務，空間複雜度同樣為 `O(1)`。
            

## **其他解法**

### **1. 陣列排序 + 計算閒置槽位（Idle Slots）**

- **思路**：
    
    - 將 26 個字母的頻率存入長度 26 的陣列 `freq` 並由小到大排序。
        
    - 最高頻任務佔據 `freq[25]`，產生的最大閒置槽位數為 `idle_slots = (freq[25] - 1) * n`。
        
    - 從 `freq[24]` 往下遍歷其他任務，用它們去扣除 `idle_slots`：`idle_slots -= min(freq[25] - 1, freq[i])`（注意每個任務最多只能填 `freq[25] - 1` 個槽位）。
        
    - 最後若 `idle_slots > 0`，回傳 `len(tasks) + idle_slots`；否則回傳 `len(tasks)`。
        
- **時間複雜度**：`O(N)`（排序固定 26 個元素為 `O(26 log 26) = O(1)`）
    
- **空間複雜度**：`O(1)`
    

## **總結**

- **面試最佳策略**：
    
    1. 先提出 **Max-Heap + Deque 模擬法**，展現你對「優先處理高頻任務」與「冷卻狀態轉移」的資料結構掌握度（且當面試官 Follow-up 要求**印出實際任務排程順序**時，只有模擬法能做到！）。
        
    2. 接著補充 **貪心公式法** `max(len(tasks), (max_freq - 1) * (n + 1) + max_count)`，將時間複雜度優化至與冷卻時間 `n` 完全無關的 `O(N)`。
        
- **常見陷阱**：忘記取 `max(len(tasks), formula_time)`，導致當任務種類很多、沒有任何 `Idle` 時間時，算出的結果小於任務總數。
    

## **貪心排程與優先佇列（Heap / Greedy）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 621**|Task Scheduler（任務排程器）|**帶相同任務冷卻時間 `n` 的最短完成時間**|**貪心桶子公式** `O(N)`，或 **Max-Heap + 冷卻佇列** 模擬|
|**LeetCode 358**|Rearrange String k Distance Apart|**重排字串使相同字元至少間隔 `k` 距離**|**Max-Heap + 固定長度 `k` 的冷卻佇列**（LC 621 輸出具體字串版）|
|**LeetCode 767**|Reorganize String（重構字串）|**重排字串使相鄰字元不相同**（即 `k = 2` 特例）|**奇偶下標交替填入最高頻字元**，或 Max-Heap 每次取前兩高頻|
|**LeetCode 2365**|Task Scheduler II（任務排程器 II）|**必須按陣列固定順序執行任務**，求最短天數|**HashMap 記錄每種任務上次執行的時間點**，遇冷卻直接快轉時間 `O(N)`|
|**LeetCode 435**|Non-overlapping Intervals（無重疊區間）|**區間排程問題**，移除最少區間使剩餘不重疊|**貪心按「結束時間 `end` 升序」排序**，優先保留早結束的區間|
|**LeetCode 253**|Meeting Rooms II（會議室 II）|**計算同時進行的會議最大重疊數**（最少會議室）|**按開始時間排序 + Min-Heap 維護最早結束時間**，或上下車掃描線法|
|**LeetCode 1353**|Maximum Number of Events That Can Be Attended|每天只能參加一場有時效的會議，求最多參加幾場|**按天數推進 + Min-Heap 貪心參加「最早結束」的可用會議**|
|**LeetCode 1834**|Single-Threaded CPU（單執行緒 CPU）|**非搶佔式最短作業優先（SJF）CPU 排程**|**按到達時間排序 + Min-Heap 維護 `(執行時間, 任務索引)`**|

### **進階延伸題：重構字串（LeetCode 767）**

這題本質上就是 `Task Scheduler` 當冷卻時間 `n = 1` 時，要求**真正構造出合法排程字串**的版本（無解時回傳 `""`）：

Python

```
import heapq
from collections import Counter

class Solution:
    def reorganizeString(self, s: str) -> str:
        counts = Counter(s)
        # 若某字元出現次數超過一半（無條件進位），必定無法隔開
        if max(counts.values()) > (len(s) + 1) // 2:
            return ""
        
        maxHeap = [(-cnt, char) for char, cnt in counts.items()]
        heapq.heapify(maxHeap)
        
        prev_cnt, prev_char = 0, ""
        res = []
        
        while maxHeap:
            cnt, char = heapq.heappop(maxHeap)
            res.append(char)
            
            # 上一輪冷卻完畢的字元，現在可以重新放回堆中
            if prev_cnt < 0:
                heapq.heappush(maxHeap, (prev_cnt, prev_char))
                
            # 更新當前字元為「冷卻中」（剩餘負頻率 + 1）
            prev_cnt, prev_char = cnt + 1, char
            
        return "".join(res)
```