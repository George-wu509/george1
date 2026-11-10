Lintcode839  合并两个排序的区间列表，返回一个合并后的列表

比較: [[Compare - Merge two Intervals]]

**样例1**
```python
"""
输入:  list1 = [(1,2),(3,4)] and list2 = [(2,3),(5,6)]
输出: [(1,4),(5,6)]
解释:
(1,2),(2,3),(3,4) --> (1,4)
(5,6) --> (5,6)
```
**样例2**

```python
"""
输入: list1 = [(1,2),(3,4)] 和 list2 = [(4,5),(6,7)]
输出: [(1,2),(3,5),(6,7)]
解释:
(1,2) --> (1,2)
(3,4),(4,5) --> (3,5)
(6,7) --> (6,7)
```



```python
class Solution:
    """
    @param list1: one of the given list
    @param list2: another list
    @return: the new sorted list of interval
    """
    def merge_two_interval(self, list1, list2):
        i, j = 0, 0
        intervals = []
        while i < len(list1) and j < len(list2):
            if list1[i].start < list2[j].start:
                self.push_back(intervals, list1[i])
                i += 1
            else:
                self.push_back(intervals, list2[j])
                j += 1
        while i < len(list1):
            self.push_back(intervals, list1[i])
            i += 1
        while j < len(list2):
            self.push_back(intervals, list2[j])
            j += 1
        
        return intervals
        
    def push_back(self, intervals, interval):
        if not intervals:
            intervals.append(interval)
            return
        
        last_interval = intervals[-1]
        if last_interval.end < interval.start:
            intervals.append(interval)
            return
        
        intervals[-1].end = max(intervals[-1].end, interval.end)
```
pass
解釋:



`Merge Two Sorted Interval Lists` 的核心思維結合了 **雙指標（Two Pointers）** 與 **動態區間合併（Interval Merging）**：因為兩個輸入串列各自已經按起始時間排序，我們可以像歸併排序（Merge Sort）一樣逐一比較雙指標指向的區間，並在放入結果集時即時處理重疊或相連的區間。

#### 核心圖解與概念

針對範例：
- `list1 = [(1,2), (3,4)]`
- `list2 = [(2,3), (5,6)]`
##### 區間軸示意圖
```
時間軸 :  1   2   3   4   5   6
-----------------------------------
list1  :  [---]   [---]            (1,2) 與 (3,4)
list2  :      [---]       [---]    (2,3) 與 (5,6)
-----------------------------------
最終結果: [-------]   [---]        (1,4) 與 (5,6)
```

- **關鍵判斷點**：
    
    1. `(1,2)` 與 `(2,3)` 在點 `2` 端點相連（`2 >= 2`），合併為 `(1,3)`。
    2. `(1,3)` 與 `(3,4)` 在點 `3` 端點相連（`3 >= 3`），繼續延伸合併為 `(1,4)`。
    3. `(1,4)` 與 `(5,6)` 中間有間隔（`4 < 5`），無法合併，獨立成為新區間。

## 範例具體執行追蹤 (Step-by-Step Trace)

初始狀態：`i = 0`, `j = 0`, `intervals = []`

|**步驟**|**當前比較**|**較小 start 的區間**|**push_back 操作與邏輯判斷**|**結果集 intervals 狀態**|**指標變更**|
|---|---|---|---|---|---|
|**1**|`list1[0]=(1,2)` vs `list2[0]=(2,3)`|`(1,2)`|`intervals` 為空，直接加入 `(1,2)`|`[(1,2)]`|`i = 1`|
|**2**|`list1[1]=(3,4)` vs `list2[0]=(2,3)`|`(2,3)`|末尾 `(1,2)` 的 `end(2) < start(2)` 為 **False**（接觸重疊）<br><br>  <br><br>合併：`end = max(2, 3) = 3`|`[(1,3)]`|`j = 1`|
|**3**|`list1[1]=(3,4)` vs `list2[1]=(5,6)`|`(3,4)`|末尾 `(1,3)` 的 `end(3) < start(3)` 為 **False**（接觸重疊）<br><br>  <br><br>合併：`end = max(3, 4) = 4`|`[(1,4)]`|`i = 2` (清空 list1)|
|**4**|`list2[1]=(5,6)` (剩餘處理)|`(5,6)`|末尾 `(1,4)` 的 `end(4) < start(5)` 為 **True**（無重疊）<br><br>  <br><br>直接加入 `(5,6)`|`[(1,4), (5,6)]`|`j = 2` (清空 list2)|

## 程式碼詳細解釋

### 1. `merge_two_interval` 主流程

Python

```
def merge_two_interval(self, list1, list2):
    i, j = 0, 0
    intervals = []
    
    # 雙指標同步比對兩串列
    while i < len(list1) and j < len(list2):
        # 永遠選擇 start 較小的區間先放入結果集（確保時間軸推進順序）
        if list1[i].start < list2[j].start:
            self.push_back(intervals, list1[i])
            i += 1
        else:
            self.push_back(intervals, list2[j])
            j += 1
            
    # 若 list1 還有剩餘，依序處理
    while i < len(list1):
        self.push_back(intervals, list1[i])
        i += 1
        
    # 若 list2 還有剩餘，依序處理
    while j < len(list2):
        self.push_back(intervals, list2[j])
        j += 1
        
    return intervals
```

- `i` 與 `j` 分別追蹤 `list1` 與 `list2` 的目前索引。
    
- 第一個 `while` 迴圈保證按 `start` 從小到大的順序取出區間。
    
- 後續兩個 `while` 迴圈確保某一個串列走完後，另一個串列剩餘的區間也能全數被推進結果集。
    

### 2. `push_back` 合併輔助函式

Python

```
def push_back(self, intervals, interval):
    # 情況 A：結果集為空，直接加入第一個區間
    if not intervals:
        intervals.append(interval)
        return
        
    last_interval = intervals[-1]
    
    # 情況 B：無重疊 (前一個區間的 end < 當前區間的 start)
    # 例如：(1,4) 與 (5,6)，4 < 5 -> 無重疊，直接 append
    if last_interval.end < interval.start:
        intervals.append(interval)
        return
        
    # 情況 C：存在重疊或端點相連 (last_interval.end >= interval.start)
    # 例如：(1,2) 與 (2,3)，2 >= 2 -> 重疊，更新前一個區間的 end 為 max(2, 3) = 3
    intervals[-1].end = max(intervals[-1].end, interval.end)
```

- **判斷條件 `last_interval.end < interval.start`** 是全演算法的核心：
    
    - 若為 `True`：代表兩區間完全分離，直接存入。
        
    - 若為 `False`：代表兩區間交集或相連，只需要將前一個區間的右界延伸至 `max(last.end, current.end)` 即可。
        

## 複雜度分析

- **時間複雜度**：$\mathcal{O}(N + M)$
    
    - 其中 $N$ 為 `list1` 的長度，$M$ 為 `list2` 的長度。雙指標只遍歷每個區間一次。
        
- **空間複雜度**：$\mathcal{O}(1)$（不計入回傳值所需的結果陣列空間）
    
    - 僅使用常數個指標變數。