
|                                                |     |
| ---------------------------------------------- | --- |
| [[#### 比較兩個合併重疊區間問題]]                          |     |
| [[#### Merge Two Sorted Interval有更好的解法用heap嗎]] |     |

#### 比較兩個合併重疊區間問題
```
A. Merge Two Sorted Interval Lists
合并两个排序的区间列表，返回一个合并后的列表
list1 = [ [1,2],[3,4] ]  
list2 = [ [2,3],[5,6] ]  

-> [ [1,4],[5,6] ]


B. Merge Intervals
给定一组区间，合并所有重叠的区间
[(1,3),(2,6),(8,10),(15,18)]

-> [(1,6),(8,10),(15,18)]
```

| **比較項目**  | **A. Merge Two Sorted Interval Lists** | **B. Merge Intervals**               |
| --------- | -------------------------------------- | ------------------------------------ |
| **輸入結構**  | 兩個**已分別排序**的區間列表 (`list1`, `list2`)    | 單一個**未排序**的區間列表 (`intervals`)        |
| **核心挑戰**  | 雙路合併 (Two-way Merge) + 區間重疊合併          | 全局排序 (Sorting) + 區間重疊合併              |
| **最佳演算法** | **雙指標 (Two Pointers)**                 | **排序 (Sorting) + 單次線性掃描**            |
| **時間複雜度** | $O(N + M)$                             | $O(N \log N)$（瓶頸在排序）                 |
| **空間複雜度** | $O(1)$（不計輸出結果陣列）                       | $O(N)$ 或 $O(\log N)$（取決於 Sorting 實作） |

## A. Merge Two Sorted Interval Lists

### 解法 Idea

由於兩條列表各自已經按起點 `start` 升序排列，直接使用 **雙指標 `i` 與 `j`** 指向兩列表的當前元素：

1. **選擇較小起點**：比較 `list1[i]` 與 `list2[j]` 的 `start`，優先挑出起點較小的區間。
2. **重疊處理 (`push_back`)**：
    - 若 `intervals` 為空，直接存入。
    - 若最後一個已存入區間的終點 `last.end` 小於當前區間起點 `curr.start`（無重疊），直接追加。
    - 若有重疊 (`last.end >= curr.start`)，則更新最後區間的終點 `last.end = max(last.end, curr.end)`。

### Code 追蹤與圖解

以 `list1 = [[1,2],[3,4]]`, `list2 = [[2,3],[5,6]]` 為例：
```
初始狀態：
list1: [1,2] -> [3,4]  (i=0)
list2: [2,3] -> [5,6]  (j=0)

[Step 1] 比較 list1[0](1,2) 與 list2[0](2,3)
         1 < 2 -> 挑選 list1[0], i 移至 1
         intervals: [[1,2]]

[Step 2] 比較 list1[1](3,4) 與 list2[0](2,3)
         2 < 3 -> 挑選 list2[0], j 移至 1
         與末項 [1,2] 比較: 2 <= 2 (重疊) -> 合併為 [1, max(2,3)] = [1,3]
         intervals: [[1,3]]

[Step 3] 比較 list1[1](3,4) 與 list2[1](5,6)
         3 < 5 -> 挑選 list1[1], i 移至 2 (list1 結束)
         與末項 [1,3] 比較: 3 <= 3 (重疊) -> 合併為 [1, max(3,4)] = [1,4]
         intervals: [[1,4]]

[Step 4] 處理剩餘 list2[1](5,6), j 移至 2
         與末項 [1,4] 比較: 4 < 5 (無重疊) -> 追加 [5,6]
         intervals: [[1,4], [5,6]]
```

## B. Merge Intervals

### 解法 Idea

輸入只有一個雜亂無章的單一列表。
1. **排序**：按區間起點 `x.start` 升序排序 $O(N \log N)$。排序後，所有潛在重疊的區間會相鄰排列。
2. **單次掃描**：遍歷排序後的區間，維護結果列表 `result`：
    - 當前區間與 `result[-1]` 無重疊（`result[-1].end < interval.start`）$\rightarrow$ 追加到 `result`。
    - 當前區間與 `result[-1]` 重疊 $\rightarrow$ 合併終點 `result[-1].end = max(result[-1].end, interval.end)`。

### Code 追蹤與圖解

以例子 `[(1,3), (2,6), (8,10), (15,18)]` 為例：
```
已排序輸入：[(1,3), (2,6), (8,10), (15,18)]

[Step 1] 處理 (1,3):
         result 為空 -> 追加 (1,3)
         result: [(1,3)]

[Step 2] 處理 (2,6):
         result[-1].end (3) >= interval.start (2) -> 重疊！
         更新終點: max(3, 6) = 6
         result: [(1,6)]

[Step 3] 處理 (8,10):
         result[-1].end (6) < interval.start (8) -> 無重疊
         追加 (8,10)
         result: [(1,6), (8,10)]

[Step 4] 處理 (15,18):
         result[-1].end (10) < interval.start (15) -> 無重疊
         追加 (15,18)
         result: [(1,6), (8,10), (15,18)]
```

## 關鍵釐清：為什麼第一題常被提及「最小堆」，而第二題不用？

### 1. 第一題：雙指標 vs 最小堆 (Min-Heap)

- **兩條已排序列表 (Merge 2 Sorted Lists)**：只需 **雙指標 (Two Pointers)** 即可在 $O(N+M)$ 時間完成，**不需要最小堆**。
- **$K$ 條已排序列表 (Merge K Sorted Lists)**： 如果問題擴展為合併 **$K$ 個** 已排序區間列表：
    - 每次需要從 $K$ 個列表中找出當前起點最小的區間。
    - 若每次用迴圈找最小，每次時間為 $O(K)$，總時間 $O(N \cdot K)$。
    - 若將 $K$ 個列表的頭節點放入**最小堆 (Min-Heap)**，每次取最小值與插入下一節點只需 $O(\log K)$，總時間優化為 $O(N \log K)$。
    - **結論**：最小堆是用於 **多路合併 (Multi-way Merge)**，而非雙路合併。

```python
class Solution:
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

### 2. 第二題：為什麼完全不需要最小堆？

- **第二題的輸入未排序**。演算法的第一步使用了全域排序 `sorted(intervals, key=lambda x: x.start)`，直接將整個列表的起點一次性調整為單調遞增。
- 排序後，數據流方向固定（只會從左到右掃描），每次合併只需要關心 **「前一個已合併區間 `result[-1]`」** 與 **「當前區間」**。
- 不需要動態獲取全域的最小值，因此建立 Min-Heap 不僅無法優化複雜度（仍需 $O(N \log N)$），反而會增加額外的 Data Structure 開銷。

```python
    def merge(self, intervals):
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result
```




#### Merge Two Sorted Interval有更好的解法用heap嗎
```
那leetcode A. Merge Two Sorted Interval有更好的解法用heap or min/max heap的嗎? 如果請說明解法及寫出python code, 以及為何更好
```
對 **只合併 2 個已排序列表** 而言，使用 最小堆（Min-Heap）**不會更好**，反而更慢且多餘；但如果問題擴展到 **$K$ 個已排序列表（Merge $K$ Sorted Interval Lists）**，最小堆就是最佳解。

## 核心結論：為什麼 2 個列表用 Heap 沒有更好？

|**比較維度**|**雙指標 (Two Pointers)**|**最小堆 (Min-Heap, K=2)**|
|---|---|---|
|**時間複雜度**|$O(N + M)$|$O((N + M) \log 2) = O(N + M)$|
|**空間複雜度**|$O(1)$|$O(2) = O(1)$（需維護大小為 2 的 Heap）|
|**常數開銷 (Overhead)**|**極低**（僅做一次 `if` 數字大小比較）|**較高**（包含 heap push/pop 樹調整開銷）|
|**程式碼簡潔度**|直觀易讀|需要維護指標與堆維護狀態|

- **只有 2 個列表**：雙指標比較兩個數字大小只需要一次 CPU 比對；而 Min-Heap 就算大小只有 2，每次取出與插入仍需執行 Heapify（維護堆結構），多出了不必要的記憶體與計算開銷。
    
- **擴展到 $K$ 個列表 ($K \ge 3$)**：若有 $K$ 個列表，用多指標尋找最小值每次需遍歷 $K$ 個元素，總時間為 $O(N \cdot K)$；而使用 Min-Heap 可以在 $O(\log K)$ 時間內取得當前最小值，將總時間降至 **$O(N \log K)$**。
    

## 最小堆（Min-Heap）解法說明

### 解法 Idea

1. **初始化 Heap**：將每個非空列表的第一個區間放入 Min-Heap 中。Heap 中儲存元組 `(interval.start, list_index, interval_index)`，依據區間起點 `start` 排序。
    
2. **彈出與推進**：
    
    - 每次從 Heap 頂端取出 `start` 最小的區間。
        
    - 將該區間傳入 `push_back` 進行重疊檢測與合併。
        
    - 如果該區間所在的列表還有下一個區間，將下一個區間放入 Heap。
        
3. **重複運作**：直到 Heap 為空為止。
    

## Python 程式碼實作（通用於 $K$ 個列表及 2 個列表）

Python

```
import heapq

class Interval:
    def __init__(self, start=0, end=0):
        self.start = start
        self.end = end

class Solution:
    def merge_k_intervals(self, lists):
        """
        :type lists: List[List[Interval]]
        :rtype: List[Interval]
        """
        min_heap = []
        
        # 1. 將每個列表的第一個區間放入 Heap
        for list_idx, lst in enumerate(lists):
            if lst:
                # 放入 tuple: (start_val, list_index, interval_index)
                heapq.heappush(min_heap, (lst[0].start, list_idx, 0))
                
        result = []
        
        # 2. 每次取出起點最小的區間
        while min_heap:
            _, list_idx, int_idx = heapq.heappop(min_heap)
            curr_interval = lists[list_idx][int_idx]
            
            # 進行合併邏輯
            self.push_back(result, curr_interval)
            
            # 如果該列表還有下一個區間，放入 Heap
            if int_idx + 1 < len(lists[list_idx]):
                next_interval = lists[list_idx][int_idx + 1]
                heapq.heappush(min_heap, (next_interval.start, list_idx, int_idx + 1))
                
        return result

    def push_back(self, intervals, interval):
        """處理區間重疊合併的輔助函式"""
        if not intervals:
            intervals.append(interval)
            return
        
        last = intervals[-1]
        if last.end < interval.start:
            intervals.append(interval)
        else:
            last.end = max(last.end, interval.end)
```

## 用範例追蹤 Min-Heap 運作流程

以 `list1 = [[1,2], [3,4]]`, `list2 = [[2,3], [5,6]]` 為例：

Plaintext

```
[初始階段]
將 list1[0](start=1) 與 list2[0](start=2) 放入 Heap。
Min-Heap 內容: [(1, list0, idx0), (2, list1, idx0)]

[Step 1] Pop 出 (1, list0, idx0) -> 區間 [1,2]
         result: [[1,2]]
         list0 還有下一項 -> Push (3, list0, idx1) 到 Heap
         Min-Heap 內容: [(2, list1, idx0), (3, list0, idx1)]

[Step 2] Pop 出 (2, list1, idx0) -> 區間 [2,3]
         與 result 末項 [1,2] 重疊 -> 合併為 [1,3]
         result: [[1,3]]
         list1 還有下一項 -> Push (5, list1, idx1) 到 Heap
         Min-Heap 內容: [(3, list0, idx1), (5, list1, idx1)]

[Step 3] Pop 出 (3, list0, idx1) -> 區間 [3,4]
         與 result 末項 [1,3] 重疊 -> 合併為 [1,4]
         result: [[1,4]]
         list0 無後續元素。
         Min-Heap 內容: [(5, list1, idx1)]

[Step 4] Pop 出 (5, list1, idx1) -> 區間 [5,6]
         無重疊 -> 追加到 result
         result: [[1,4], [5,6]]
         list1 無後續元素，Heap 變空，結束。
```