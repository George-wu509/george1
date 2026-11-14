
LeetCode 981 / LintCode 1877 設計一個基於時間的鍵值資料結構，該結構可以在不同時間戳記儲存對應同一個鍵的多個值，並針對特定時間戳記檢索鍵對應的值。

實作 `TimeMap` 類別：

- `TimeMap()` 初始化資料結構物件。
    
- `void set(String key, String value, int timestamp)` 儲存鍵 `key`、值 `value`，以及給定的時間戳記 `timestamp`。
    
- `String get(String key, int timestamp)`
    
    - 返回先前呼叫 `set(key, value, timestamp_prev)` 所儲存的值，其中 `timestamp_prev <= timestamp`。
        
    - 如果有多個這樣的值，則返回對應最大的 `timestamp_prev` 的那個值。
        
    - 如果沒有值，則返回空字串 `""`。
        

**提示條件**：所有的 `set` 操作中的時間戳記 `timestamp` 都是**嚴格遞增**的。

**範例 1：** 輸入：
```Python
["TimeMap", "set", "get", "get", "set", "get", "get"]
[[], ["foo", "bar", 1], ["foo", 1], ["foo", 3], ["foo", "bar2", 4], ["foo", 4], ["foo", 5]]
```

輸出：
```Python
[None, None, "bar", "bar", None, "bar2", "bar2"]
```

解釋：
```Python
timeMap = TimeMap()
timeMap.set("foo", "bar", 1)  # 儲存鍵 "foo" 和值 "bar"，時間戳記 timestamp = 1
timeMap.get("foo", 1)         # 返回 "bar"
timeMap.get("foo", 3)         # 返回 "bar"，因為在時間戳記 3 和 2 處沒有對應 "foo" 的值，唯一的值位於時間戳記 1 處（即 "bar"）
timeMap.set("foo", "bar2", 4) # 儲存鍵 "foo" 和值 "bar2"，時間戳記 timestamp = 4
timeMap.get("foo", 4)         # 返回 "bar2"
timeMap.get("foo", 5)         # 返回 "bar2"
```

**範例 2：** 輸入：
```Python
["TimeMap", "set", "set", "get", "get", "get", "get", "get"]
[[], ["love", "high", 10], ["love", "low", 20], ["love", 5], ["love", 10], ["love", 15], ["love", 20], ["love", 25]]
```

輸出：
```Python
[None, None, None, "", "high", "high", "low", "low"]
```

解釋： 時間戳記 `5` 小於所有已存入的時間戳記（最小為 `10`），故返回 `""`；時間戳記 `15` 介於 `10` 與 `20` 之間，返回不大於 `15` 的最大時間戳記 `10` 對應的值 `"high"`。

### 解法一：雜湊表 + 手寫二分搜尋（Optimal - `set` $O(1)$ / `get` $O(\log n)$）

```Python
from collections import defaultdict

class TimeMap:

    def __init__(self):
        # key -> list of (timestamp, value)
        self.store = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        # 題目保證 timestamp 嚴格遞增，直接 append 即可維持列表有序
        self.store[key].append((timestamp, value))

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.store:
            return ""
        
        values = self.store[key]
        left, right = 0, len(values) - 1
        res = ""
        
        # 二分搜尋尋找小於等於 timestamp 的最後一個位置（上界）
        while left <= right:
            mid = left + (right - left) // 2
            if values[mid][0] <= timestamp:
                res = values[mid][1]  # 記錄當前合法的值，並繼續往右尋找更接近的時間戳記
                left = mid + 1
            else:
                right = mid - 1       # 時間戳記太大，往左縮小範圍
                
        return res
```


### 解法二：雜湊表 + Python 內建 `bisect` 模組（簡潔寫法 - `set` $O(1)$ / `get` $O(\log n)$）

```Python
from collections import defaultdict
import bisect

class TimeMap:

    def __init__(self):
        # 將 timestamps 與 values 分開儲存，方便直接使用 bisect_right
        self.timestamps = defaultdict(list)
        self.values = defaultdict(list)

    def set(self, key: str, value: str, timestamp: int) -> None:
        self.timestamps[key].append(timestamp)
        self.values[key].append(value)

    def get(self, key: str, timestamp: int) -> str:
        if key not in self.timestamps:
            return ""
        
        # bisect_right 找到第一個嚴格大於 timestamp 的索引，減 1 即為 <= timestamp 的最大索引
        idx = bisect.bisect_right(self.timestamps[key], timestamp) - 1
        if idx >= 0:
            return self.values[key][idx]
        return ""
```

解釋:

- `step1`: 使用 `defaultdict(list)` 建立雜湊表，讓每個 `key` 對應一個儲存 `(timestamp, value)` 元組的列表。
    
- `step2`: 呼叫 `set` 時，因為題目明確條件「所有的 `timestamp` 都是**嚴格遞增**的」，直接 `append` 到列表尾端，列表天然就是按 `timestamp` 升序排列，無需額外排序。
    
- `step3`: 呼叫 `get(key, timestamp)` 時，先檢查 `key` 是否存在；若不存在直接回傳 `""`。
    
- `step4`: 以 `store["love"] = [(10, "high"), (20, "low")]`，查詢 `get("love", 15)` 為例，初始 `left = 0`, `right = 1`, `res = ""`。
    
- `step5`: 第 1 輪二分：`mid = (0 + 1) // 2 = 0`，`values[0][0] = 10 <= 15` 成立！更新 `res = "high"`，並將 `left` 右移為 `1`（嘗試看看有沒有更接近 `15` 的合法時間戳記）。
    
- `step6`: 第 2 輪二分：`mid = (1 + 1) // 2 = 1`，`values[1][0] = 20 > 15` 超過目標時間戳記！將 `right` 左移為 `0`。此時 `left > right` 結束迴圈，回傳 `res = "high"`。
    

解釋: 這道題結合了 **雜湊表（Hash Table）** 與 **二分搜尋上界（Upper Bound Binary Search）**。

雜湊表負責在 $O(1)$ 時間內定位到目標 `key` 的歷史紀錄列表；而利用時間戳記嚴格遞增的特性，在該列表中透過二分搜尋找出滿足 `timestamp_prev <= timestamp` 的最右側元素。

### **LeetCode 981 / LintCode 1877 - Time Based Key-Value Store（基於時間的鍵值儲存）**

這道題目是 **資料結構設計（Design） + 二分搜尋** 的經典題型，也是 NeetCode 150 的必考題。在實際工程中，這就像是簡易版的時序資料庫（Time-Series Database）或版本控制系統（Git 根據時間點檢索歷史版本）。

## **解法解析**

這道題可以拆解為 **「按鍵分組儲存」** 與 **「有序陣列找前驅（Floor / Upper Bound）」** 兩個部分。

### **思路**

1. **選擇合適的底層容器**
    
    - 我們需要快速透過 `key` 找到所有歷史紀錄，因此外層使用 **雜湊表（Dictionary / HashMap）**。
        
    - 每個 `key` 對應多個 `(timestamp, value)` 紀錄。由於題目保證每次 `set` 傳入的 `timestamp` 嚴格遞增，使用 **動態陣列（List）** 依序追加即可保持時間戳記有序，且支援 $O(1)$ 隨機索引存取，完美契合二分搜尋。
        
2. **二分搜尋的目標定義（找小於等於 `target` 的最大值）**
    
    - 傳統二分搜尋是找「等於 `target` 的元素」，但本題允許返回比查詢時間更早的最近一筆紀錄。
        
    - 因此我們的搜尋目標是：**在有序列表中，找到最後一個滿足 `values[mid][0] <= timestamp` 的位置**。
        
    - 每次遇到 `values[mid][0] <= timestamp` 時：
        
        - 先將 `values[mid][1]` 存入候選答案 `res`（因為它可能是最終答案）。
            
        - 接著令 `left = mid + 1`，繼續往右半邊試探是否還有時間戳記更大、但仍不超過 `timestamp` 的紀錄。
            
    - 若 `values[mid][0] > timestamp`：
        
        - 該紀錄發生在查詢時間之後（未來時間），不合法，令 `right = mid - 1` 往左縮小範圍。
            
3. **無合法紀錄的邊界處理**
    
    - 如果所有存入的時間戳記都大於查詢的 `timestamp`（例如存入 `10, 20`，查詢 `5`），`values[mid][0] <= timestamp` 從未被觸發，`res` 維持初始值 `""` 並直接返回。
        

## **具體舉例**

假設依序執行 `set("love", "high", 10)`、`set("love", "low", 20)`，目前 `store["love"] = [(10, "high"), (20, "low")]`，執行不同 `get` 查詢的變化如下：

|**查詢操作**|**left, right 初始**|**mid 檢查過程**|**最終 res**|**說明**|
|---|---|---|---|---|
|`get("love", 5)`|`0, 1`|`mid=0` (`10 > 5`) $\rightarrow$ `right=-1` 結束|`""`|所有時間戳記皆大於 `5`，無效查詢|
|`get("love", 15)`|`0, 1`|`mid=0` (`10 <= 15`, `res="high"`, `left=1`) $\rightarrow$ `mid=1` (`20 > 15`, `right=0`)|`"high"`|`10` 為不超過 `15` 的最大時間戳記|
|`get("love", 20)`|`0, 1`|`mid=0` (`10 <= 20`, `res="high"`, `left=1`) $\rightarrow$ `mid=1` (`20 <= 20`, `res="low"`, `left=2`)|`"low"`|精準命中 `20`，覆蓋舊答案並回傳 `"low"`|

## **時間與空間複雜度分析**

- **時間複雜度**：
    
    - `set` 操作：**$O(1)$**（平攤時間複雜度，直接向列表尾端 `append` 元素）。
        
    - `get` 操作：**$O(\log n)$**，其中 $n$ 為該 `key` 呼叫 `set` 的總次數。雜湊表查詢為 $O(1)$，在長度為 $n$ 的有序列表中進行二分搜尋耗時 $O(\log n)$。
        
- **空間複雜度：$O(N)$**
    
    - 其中 $N$ 為所有 `set` 操作的總呼叫次數，雜湊表與列表需要儲存每一筆 `(timestamp, value)` 紀錄。
        

## **其他解法**

### **1. 雜湊表 + 平衡二元搜尋樹（TreeMap / SortedList）**

- **思路**：
    
    - 若題目**沒有**保證 `set` 的 `timestamp` 嚴格遞增（即可能亂序插入歷史時間戳記），單純使用 List 的 `append` 會破壞有序性（插入變成 $O(n)$）。
        
    - 此時在 Java/C++ 中可使用 `TreeMap`（紅黑樹）的 `floorEntry(timestamp)`，在 Python 中可使用 `sortedcontainers.SortedDict`，讓亂序插入與查詢都維持在 $O(\log n)$。
        
- **時間複雜度**：`set` 為 $O(\log n)$，`get` 為 $O(\log n)$
    
- **空間複雜度**：$O(N)$
    

### **2. 雜湊表 + 線性倒序掃描**

- **思路**：
    
    - 存入時直接 `append`，查詢時從列表尾端（最新時間）往前逐一比對，遇到第一個 `<= timestamp` 的元素即返回。雖然在查詢最新資料時很快，但查詢早期資料時會退化。
        
- **時間複雜度**：`set` 為 $O(1)$，`get` 最壞為 $O(n)$
    
- **空間複雜度**：$O(N)$
    

## **總結**

- **面試最優解**：**HashMap + 陣列二分搜尋**，`set` 時間 $O(1)$、`get` 時間 $O(\log n)$、空間 $O(N)$。
    
- **面試加分點（Follow-up 應對）**：面試官常會追問「**如果 `set` 的 `timestamp` 不是嚴格遞增的怎麼辦？**」此時可以回答將內層資料結構從動態陣列替換為**平衡二元搜尋樹（如 `TreeMap` 或 `SortedDict`）**，即可在 $O(\log n)$ 內維持有序插入。
    
- **易錯細節**：
    
    1. 在 Python 中使用 `bisect_right` 時，回傳的是「第一個**大於** `timestamp` 的位置」，因此務必記得減 `1` 才是「小於等於 `timestamp` 的最大位置」，且必須檢查 `idx >= 0`。
        
    2. 不要在 `get` 時直接把整個 `(timestamp, "")` 丟進 `bisect_right` 與 `(timestamp, value)` 比較，因為元組比較第二項字串時，任何非空字串都會大於 `""`，導致相同時間戳記被誤判（建議將 `timestamps` 獨立存成純整數列表，或自訂 key / 手寫二分）。
        

## **Blind 75 + NeetCode 150 二分搜尋與設計類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 981**|Time Based Key-Value Store（基於時間的鍵值儲存）|**設計題 + 二分搜尋上界**|**HashMap 存時間戳記列表，查詢時二分找 `<= timestamp` 的最大值，時間 $O(\log n)$**|
|**LeetCode 704**|Binary Search（二分查找）|**基礎 1D 有序陣列二分**|**標準左右閉區間 `[left, right]` 二分搜尋，時間 $O(\log n)$**|
|**LeetCode 34**|Find First and Last Position of Element in Sorted Array|**二分搜尋左右邊界**|**兩次二分搜尋分別找第一個 `>= target` 與最後一個 `<= target` 的索引**|
|**LeetCode 74**|Search a 2D Matrix（搜索二維矩陣）|**2D 轉 1D 有序陣列二分**|**將二維座標映射為一維索引 `mid // cols, mid % cols`，時間 $O(\log(m \cdot n))$**|
|**LeetCode 875**|Koko Eating Bananas（愛吃香蕉的珂珂）|**答案空間二分（Binary Search on Answer）**|**二分猜測吃香蕉速度 `k`，用 `check(k)` 驗證總時間，時間 $O(n \log M)$**|
|**LeetCode 153**|Find Minimum in Rotated Sorted Array（尋找旋轉排序陣列中的最小值）|**旋轉有序陣列二分**|**比較 `nums[mid]` 與 `nums[right]` 判斷最小值所在半區，時間 $O(\log n)$**|
|**LeetCode 33**|Search in Rotated Sorted Array（搜索旋轉排序陣列）|**旋轉有序陣列二分**|**每次切分必有一半完全有序，判斷 `target` 是否在有序區間內，時間 $O(\log n)$**|
|**LeetCode 146**|LRU Cache（LRU 快取機制）|**經典設計題（HashMap + 雙向鏈表）**|**HashMap 實現 $O(1)$ 查找，雙向鏈表實現 $O(1)$ 移動與淘汰最久未使用節點**|

### 延伸變形題：快照陣列（LeetCode 1146 - Snapshot Array）

實作一個支援下列介面的類別 `SnapshotArray`（本質上是針對每個陣列索引維護一個基於 `snap_id` 時間戳記的歷史版本列表）：

- `SnapshotArray(int length)`：初始化一個與指定長度相等的類陣列資料結構，初始每個元素都等於 `0`。
    
- `void set(index, val)`：會將指定索引 `index` 處的元素設定為 `val`。
    
- `int snap()`：獲取該陣列的快照，並返回總快照的次數 `snap_id`（從 `0` 開始計算）。
    
- `int get(index, snap_id)`：根據指定的 `snap_id` 選擇快照，並返回該快照指定索引 `index` 的值。
    

Python

```
import bisect

class SnapshotArray:

    def __init__(self, length: int):
        self.snap_id = 0
        # 每個索引獨立維護一個歷史紀錄列表，初始在 snap_id = 0 時值為 0
        self.history = [[(0, 0)] for _ in range(length)]

    def set(self, index: int, val: int) -> None:
        # 若當前 snap_id 已經有修改紀錄，直接覆蓋最後一筆；否則新增一筆版本紀錄
        if self.history[index][-1][0] == self.snap_id:
            self.history[index][-1] = (self.snap_id, val)
        else:
            self.history[index].append((self.snap_id, val))

    def snap(self) -> int:
        self.snap_id += 1
        return self.snap_id - 1

    def get(self, index: int, snap_id: int) -> int:
        # 尋找 <= snap_id 的最後一個版本（用 float('inf') 確保元組比較第二項時不會誤判）
        records = self.history[index]
        idx = bisect.bisect_right(records, (snap_id, float('inf'))) - 1
        return records[idx][1]
```