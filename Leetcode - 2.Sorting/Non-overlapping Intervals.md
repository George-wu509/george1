
LeetCode 435 / LintCode 1479 給定一個區間的集合 `intervals`，其中 `intervals[i] = [start_i, end_i]`。請回傳**需要移除區間的最小數量**，使剩餘區間互不重疊。

請注意：**只在一個點上接觸的區間是不重疊的**。例如 `[1, 2]` 和 `[2, 3]` 是互不重疊的。

**样例 1：** 输入：
```Python
intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]
```

输出：
```Python
1
```

解释： 移除 `[1, 3]` 後，剩下的區間 `[1, 2]`、`[2, 3]`、`[3, 4]` 互不重疊。

**样例 2：** 输入：
```Python
intervals = [[1, 2], [1, 2], [1, 2]]
```

输出：
```Python
2
```

解释： 你需要移除兩個 `[1, 2]`，才能讓剩下的唯一一個 `[1, 2]` 與其他區間無重疊。

**样例 3：** 输入：
```Python
intervals = [[1, 2], [2, 3]]
```

输出：
```Python
0
```

解释： 你不需要移除任何區間，因為 `[1, 2]` 和 `[2, 3]` 僅在端點 `2` 相接，題目定義為互不重疊。

```python
class Solution:
   def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
      intervals.sort(key=lambda x:x[1])
      removed = 0
      prev_end = float("-inf")

      for start, end in intervals:
         if start >= prev_end:
            prev_end = end
         else:
            removed += 1
      return removed
```
解釋版
```Python
class Solution:
    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        # 1. 依照區間的「結束時間（右端點）」由小到大排序
        intervals.sort(key=lambda x: x[1])

        removed = 0
        prev_end = float("-inf")  # 紀錄上一個被保留區間的結束時間

        # 2. 貪心掃描：優先保留結束時間最早的區間
        for start, end in intervals:
            if start >= prev_end:
                # 當前區間與上一個保留區間不重疊，安全保留並更新邊界
                prev_end = end
            else:
                # 發生重疊，因為已按結束時間排序，當前區間必定結束得更晚（或相同），直接移除
                removed += 1

        return removed
```

```Python
# 另一種寫法：依照「起始時間（左端點）」排序，重疊時貪心保留結束時間較小者
class Solution:
    def eraseOverlapIntervals(self, intervals: list[list[int]]) -> int:
        intervals.sort(key=lambda x: x[0])

        removed = 0
        prev_end = intervals[0][1]

        for start, end in intervals[1:]:
            if start >= prev_end:
                # 無重疊，直接推進右邊界
                prev_end = end
            else:
                # 發生重疊，必須二選一移除；貪心保留「結束時間較早」的那個
                removed += 1
                prev_end = min(prev_end, end)

        return removed
```

解釋: 
step1: 問題等價於「**在所有區間中，最多能選出幾個互不重疊的區間**」（最少移除數 = 總區間數 - 最多保留數） 
step2: 以 `intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]` 為例，先依結束時間排序得到 `[[1, 2], [2, 3], [1, 3], [3, 4]]` 
step3: 初始化 `removed = 0`、`prev_end = -inf` 
step4: 讀取 `[1, 2]`：`start = 1 >= -inf`（不重疊），保留它，更新 `prev_end = 2` 
step5: 讀取 `[2, 3]`：`start = 2 >= 2`（端點相接不算重疊），保留它，更新 `prev_end = 3` 
step6: 讀取 `[1, 3]`：`start = 1 < 3`（發生重疊！因為前面保留的區間佔用了到 `3` 的空間，而 `[1, 3]` 的起點在 `3` 之前），必須移除，`removed` 變為 `1`，`prev_end` 維持 `3` 
step7: 讀取 `[3, 4]`：`start = 3 >= 3`（不重疊），保留它，更新 `prev_end = 4`，所以答案 = `1`

解釋: 為什麼要按「結束時間（右端點）」排序，而不是按「起始時間」或「區間長度」排序？

因為**越早結束的區間，留給後面其他區間的可用時間軸空間就越大**。如果選了一個很早開始但很晚才結束的巨無霸區間（例如 `[1, 100]`），它會卡死後面所有短區間；反之，每次都挑選「最早結束」的合法區間，就能把右邊界 `prev_end` 壓在最小的位置，從而容納最多不重疊區間。
```Plaintext
時間軸:   1       2       3       4
          |-------|       |       |   [1, 2] -> 保留 (prev_end 更新為 2)
          |       |-------|       |   [2, 3] -> 保留 (start 2 >= prev_end 2，無重疊，更新為 3)
          |---------------|       |   [1, 3] -> 移除！(start 1 < prev_end 3，重疊，removed + 1)
          |       |       |-------|   [3, 4] -> 保留 (start 3 >= prev_end 3，無重疊，更新為 4)
```



### **LeetCode 435 / LintCode 1479 - Non-overlapping Intervals（無重疊區間）**

這道題目是 **貪心演算法（Greedy）與區間調度問題（Interval Scheduling Problem）** 的教科書級經典題（Blind 75 與 NeetCode 150 必考題）。它直接對應演算法理論中的「最大不相交區間數（Activity Selection Problem）」，也是處理會議室排程、任務衝突檢測的核心模型。

## **解法解析**

這道題可以轉化為 **逆向思維（求最大保留區間數）+ 右端點貪心排序** 來求解。

### **思路**

1. **轉化問題目標**
    
    - 直接思考「該移除哪幾個區間」容易陷入組合爆炸；換個角度想：**「移除最少區間」完全等價於「保留最多互不重疊的區間」**。
        
    - 只要我們能找出最大不相交子集的大小 `max_kept`，答案就是 `len(intervals) - max_kept`（或在掃描過程中直接統計被淘汰的數量 `removed`）。
        
2. **驗證貪心選擇性質（為什麼按結束時間排序最優？）**
    
    - 假設我們現在要挑選第一個保留的區間。在所有區間中，結束時間最早的那個區間（設為 $I_1$，結束於 $e_1$）一定可以成為某個最優解的第一個區間。
        
    - 原因：如果某個最優解的第一個區間是 $I_k$（結束於 $e_k$，且 $e_k \ge e_1$），我們把 $I_k$ 替換成 $I_1$，由於 $e_1 \le e_k$，$I_1$ 絕對不會與後面的第二個區間產生新的重疊，且留給後續區間的空間只會更大或相等。
        
    - 依此歸納，每一步都貪心選擇「不與前一個衝突，且結束時間最早」的區間，必然得到全局最優解。
        
3. **代碼執行流程**
    
    - 將 `intervals` 依照 `x[1]`（結束時間）升序排列。
        
    - 初始化 `removed = 0` 與 `prev_end = float("-inf")`。
        
    - 遍歷排序後的每個區間 `[start, end]`：
        
        - 若 `start >= prev_end`：表示無重疊，接納此區間，令 `prev_end = end`。
            
        - 若 `start < prev_end`：表示與上一個保留的區間衝突，直接淘汰此區間（`removed += 1`），`prev_end` 保持不變。
            

## **具體舉例**

假設 `intervals = [[1, 2], [2, 3], [3, 4], [1, 3]]`，依結束時間排序後為： `[[1, 2], [2, 3], [1, 3], [3, 4]]`。

遍歷過程中的變數變化如下：

|**步驟**|**當前區間 [start, end]**|**判斷條件 start >= prev_end**|**決策動作**|**更新後 prev_end**|**累計 removed**|
|---|---|---|---|---|---|
|初始|-|-|-|`-inf`|`0`|
|第 1 步|`[1, 2]`|`1 >= -inf`（成立）|**保留**第一個最早結束的區間|`2`|`0`|
|第 2 步|`[2, 3]`|`2 >= 2`（成立）|**保留**（端點 `2` 相接不算重疊）|`3`|`0`|
|第 3 步|`[1, 3]`|`1 >= 3`（不成立）|**移除**（`1 < 3`，與 `[2, 3]` 重疊）|`3`|`1`|
|第 4 步|`[3, 4]`|`3 >= 3`（成立）|**保留**（端點 `3` 相接不算重疊）|`4`|`1`|

最終返回 `removed = 1`。

## **時間與空間複雜度分析**

- **時間複雜度：$O(n \log n)$**
    
    - 排序 $n$ 個區間需要 **$O(n \log n)$** 的時間。
        
    - 線性掃描一次陣列需要 $O(n)$ 時間，因此總時間複雜度由排序主導，為 **$O(n \log n)$**。
        
- **空間複雜度：$O(\log n)$ 或 $O(n)$**
    
    - 演算法本身只使用了 `removed` 和 `prev_end` 兩個常數變數（$O(1)$ 額外空間）。
        
    - 但 Python 的內建 Timsort 排序在最壞情況下需要 **$O(n)$** 的輔助空間（平均為 $O(\log n)$）。
        

## **其他解法**

### **1. 按起始時間（左端點）排序 + 貪心取較小右端點**

- **思路**：
    
    - 依照 `x[0]`（起始時間）排序。
        
    - 當遇到重疊（`start < prev_end`）時，代表上一個保留的區間與當前區間二選一必須被移除。
        
    - 為了讓後續空間最大化，我們貪心保留「右端點較小」的那個，即執行 `prev_end = min(prev_end, end)`（等效於把右端點較大的那個區間刪掉）。
        
- **時間複雜度**：$O(n \log n)$
    
- **空間複雜度**：$O(n)$（排序所需空間）
    

### **2. 動態規劃（Dynamic Programming）**

- **思路**：
    
    - 依照起始時間排序後，定義 `dp[i]` 為「以第 `i` 個區間結尾時，最多能保留幾個互不重疊的區間」（類似最長遞增子序列 LIS）。
        
    - 狀態轉移：`dp[i] = max(dp[j] + 1)`，其中 `j < i` 且 `intervals[j][1] <= intervals[i][0]`。
        
    - 最後答案為 `n - max(dp)`。
        
- **時間複雜度**：$O(n^2)$（在 LeetCode 大測資下會超時 TLE，僅用於理解與帶權重變形題的對比）
    
- **空間複雜度**：$O(n)$
    

## **總結**

- **面試標準最佳解（右端點排序貪心法）**：時間 **$O(n \log n)$**、額外空間 **$O(1)$**，邏輯最乾淨俐落，且不需要在重疊時進行 `min()` 比較。
    
- **核心記憶口訣**：
    
    - **合併區間（Merge Intervals）** $\rightarrow$ 看**左端點**（起點）排序。
        
    - **不相交 / 移除重疊 / 射氣球（Non-overlapping）** $\rightarrow$ 看**右端點**（終點）排序。
        

## **LeetCode / LintCode 區間調度與重疊問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 435 / LintCode 1479**|Non-overlapping Intervals（無重疊區間）|**求最少移除幾個區間使剩餘互不重疊（端點相接不算重疊）**|**按右端點排序 + 貪心保留（`start >= prev_end`）**，$O(n \log n)$|
|**LeetCode 452**|Minimum Number of Arrows to Burst Balloons|**求最少幾支箭射穿所有氣球（端點相接算重疊）**|**按右端點排序 + 貪心射擊（`start > prev_end` 時加一箭）**|
|**LeetCode 56 / LintCode 156**|Merge Intervals（合併區間）|**合併所有有交集的區間**|**按左端點排序 + 動態更新當前區間右邊界 `max(end)`**|
|**LeetCode 57 / LintCode 30**|Insert Interval（插入區間）|**在已排序無重疊區間中插入新區間並合併**|**三段式線性掃描（左側無重疊、中間合併、右側無重疊）**|
|**LeetCode 252 / LintCode 920**|Meeting Rooms（會議室）|**判斷一個人能否參加所有會議（是否有任何重疊）**|**按左端點排序 + 檢查相鄰區間是否 `intervals[i][0] < intervals[i-1][1]`**|
|**LeetCode 253 / LintCode 919**|Meeting Rooms II（會議室 II）|**求同時進行會議所需的最少會議室數量（最大重疊層數）**|**按左端點排序 + 最小堆（Min-Heap）追蹤最早結束時間**，或**掃描線算法**|
|**LeetCode 1235**|Maximum Profit in Job Scheduling|**帶權重區間調度：每個區間有收益，求不重疊最大總收益**|**按右端點排序 + 動態規劃（DP）+ 二分搜尋（Binary Search）**|

### **進階變形題：如果每個區間都有一個「權重（收益 `profit[i]`）」，要求在互不重疊的前提下最大化總收益，貪心還能用嗎？**

當區間帶上權重後（對應 **LeetCode 1235 - Maximum Profit in Job Scheduling**），**貪心演算法會直接失效**（例如一個時間極長但收益極高的區間 `[1, 10, profit=100]`，可能比兩個短區間 `[1, 2, profit=10]` + `[2, 3, profit=10]` 更有價值）。此時必須改用 **按結束時間排序 + 動態規劃 + 二分搜尋**：

Python

```
import bisect

class Solution:
    def jobScheduling(self, startTime: list[int], endTime: list[int], profit: list[int]) -> int:
        # 1. 將工作打包並依照「結束時間」升序排序
        jobs = sorted(zip(startTime, endTime, profit), key=lambda x: x[1])
        n = len(jobs)

        # dp[i] 表示考慮前 i 個工作時，能獲得的最大總收益
        dp = [0] * (n + 1)
        ends = [job[1] for job in jobs]

        for i in range(1, n + 1):
            s, e, p = jobs[i - 1]
            # 2. 用二分搜尋找出「結束時間 <= s」的最後一個工作索引
            idx = bisect.bisect_right(ends, s)
            # 3. 狀態轉移：max(不選當前工作, 選當前工作 + 相容前序工作的最大收益)
            dp[i] = max(dp[i - 1], dp[idx] + p)

        return dp[n]
```