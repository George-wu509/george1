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

example: list1 = [(1,2),(3,4)] and list2 = [(2,3),(5,6)]
解釋:
step1: 在merge_two_interval用雙指針(i在list1開頭, j在list2開頭)
step2: create空的intervals = [], 用來跟list1 or list2相比, 也是最後result
step3: 開始(1,2)的start=1比(2,3)的start=2, 小(list1)的進入push_back()
step4: 在push_back(), 如果intervals是空的則list1or2直接加入intervals
step5: 如果intervals不是空的, 則list1 or 2比對intervals最後看是否重疊



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

##### 1. `merge_two_interval` 主流程
```Python
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
- `i` 與 `j` 分別追蹤 `list1` 與 `list2` 的目前索引
- 第一個 `while` 迴圈保證按 `start` 從小到大的順序取出區間。
- 後續兩個 `while` 迴圈確保某一個串列走完後，另一個串列剩餘的區間也能全數被推進結果集。
##### 2. `push_back` 合併輔助函式
```Python
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
#### 複雜度分析
- **時間複雜度**：$\mathcal{O}(N + M)$
    - 其中 $N$ 為 `list1` 的長度，$M$ 為 `list2` 的長度。雙指標只遍歷每個區間一次。
- **空間複雜度**：$\mathcal{O}(1)$（不計入回傳值所需的結果陣列空間）
    - 僅使用常數個指標變數。



```
我還是不太理解push_back function的功能, 所以這解法的idea是甚麼? 並更詳細解釋push_back function
```
這個演算法的核心 Idea 可以用一句話概括：**「邊按時間順序發放區間，邊用 `push_back` 當作『守門員』即時吃掉重疊」**。

## 整個解法的核心 Idea 到底是什麼？

如果我們把這題拆成兩件事：

1. **排序與排序合併 (Merge Sort 思想)**： `list1` 和 `list2` 本身都已經從小到大排好序了。我們用 `i` 和 `j` 兩個指標，誰的**開頭時間 (`start`) 比較早**，就把誰推進去處理。這樣能確保我們處理區間的順序永遠是從左到右、時間軸不倒退的。
    
2. **動態消消樂 (`push_back` 的職責)**： 如果我們直接把比出來的區間通通放入結果集 `intervals`，可能會產生 `[(1,2), (2,3), (3,4)]` 這種破碎且重疊的結果。 因此，我們**不直接寫 `intervals.append(interval)`**，而是交給 `push_back` 檢查：**「剛準備放進來的這個區間，跟最後一個已經放進去的區間有沒有撞在一起？」**
    

## `push_back` Function 到底在做什麼？

`push_back` 的目標只有一個：**維護 `intervals` 陣列，確保裡面的每一個區間都「絕不重疊」且「完全合併」**。

因為我們丟進 `push_back` 的區間，其 `start` 一定**大於等於**前面已經放入區間的 `start`，所以我們**只需要看最新進來的區間跟 `intervals[-1]`（最後一個區間）的關係**。

```
  情況一：完全分開 (無重疊)           情況二：重疊或接觸 (需合併)

  [ last_interval ]                 [  last_interval  ]
                      [ new ]                     [   new   ]
  ------------------------->        ------------------------->
  last.end < new.start              last.end >= new.start
  -> 直接加入                         -> 把 last.end 拉長到 max(last.end, new.end)
```

### `push_back` 逐行詳細剖析

Python

```
def push_back(self, intervals, interval):
    # -------------------------------------------------------------
    # 1. 邊界條件：如果結果集還是空的，沒有人可以比對
    # -------------------------------------------------------------
    if not intervals:
        intervals.append(interval) # 直接放進去作為第一個基準
        return
        
    # 取得目前結果集裡「最新/最後一個區間」
    last_interval = intervals[-1]
    
    # -------------------------------------------------------------
    # 2. 判斷是否「完全沒有重疊」
    # -------------------------------------------------------------
    # 如果舊區間的結束時間 < 新區間的開始時間 (例如 4 < 5)
    if last_interval.end < interval.start:
        intervals.append(interval) # 兩者撞不在一起，直接作為獨立的新區間新增進去
        return
        
    # -------------------------------------------------------------
    # 3. 處理「有重疊」或「端點相連」的情況
    # -------------------------------------------------------------
    # 走到這裡，代表 last_interval.end >= interval.start 必定成立！
    # 說明新區間已經「吃進」或「接到」舊區間的尾巴上了。
    # 我們不需要新增元素，只需要將最後一個區間的 end 延伸拉長即可！
    intervals[-1].end = max(intervals[-1].end, interval.end)
```

## 用生活情境舉例

想像你在**排定會議室預約時間**，`intervals` 是你最終劃定被佔用的時間表：

1. **第一個預約 (1:00 - 2:00)** 來了：
    
    - 目前時間表是空的，`push_back` 直接記下 `[(1:00, 2:00)]`。
        
2. **第二個預約 (2:00 - 3:00)** 來了：
    
    - `push_back` 拿它跟最後一筆 `(1:00, 2:00)` 比較。
        
    - 發現開會時間剛好接上（2:00 <= 2:00，不滿足 `end < start`）。
        
    - 代表這間會議室從 1:00 會一路被用到 3:00，所以直接把最後一筆修改成 `(1:00, 3:00)`。
        
3. **第三個預約 (3:00 - 4:00)** 來了：
    
    - `push_back` 拿它跟最後一筆 `(1:00, 3:00)` 比較。
        
    - 再次接上！修改最後一筆變成 `(1:00, 4:00)`。
        
4. **第四個預約 (5:00 - 6:00)** 來了：
    
    - `push_back` 拿它跟最後一筆 `(1:00, 4:00)` 比較。
        
    - 4:00 到 5:00 中間有空檔（`4:00 < 5:00` 成立！）。
        
    - 兩段無法合併，於是在時間表新增獨立一筆，變成 `[(1:00, 4:00), (5:00, 6:00)]`。
        

`push_back` 就是這套邏輯的實作，確保隨時維護好這張乾淨、不重疊的時間表。
