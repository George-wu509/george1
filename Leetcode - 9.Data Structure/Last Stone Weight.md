
最後一塊石頭的重量 Last Stone Weight

LeetCode 1046 (LintCode 1814) 有一堆石頭，每塊石頭的重量都是正整數。

每一回合，從中選出**最重的兩塊石頭**，然後將它們一起粉碎。假設石頭的重量分別為 `x` 和 `y`，且 `x <= y`。那麼粉碎的可能結果如下：

- 如果 `x == y`，那麼兩塊石頭都會被完全粉碎；
- 如果 `x != y`，那麼重量為 `x` 的石頭將會完全粉碎，而重量為 `y` 的石頭新重量為 `y - x`。

最後，最多只會剩下一塊石頭。回傳此石頭的重量。如果沒有石頭剩下，就回傳 `0`。

**样例 1：** 输入：
```Python
stones = [2, 7, 4, 1, 8, 1]
```

输出：
```Python
1
```

解释：

- 先選出 `7` 和 `8` 相撞，得到 `1`，陣列轉換為 `[2, 4, 1, 1, 1]`
- 再選出 `2` 和 `4` 相撞，得到 `2`，陣列轉換為 `[2, 1, 1, 1]`
- 接著選出 `2` 和 `1` 相撞，得到 `1`，陣列轉換為 `[1, 1, 1]`
- 最後選出 `1` 和 `1` 相撞，得到 `0`，陣列轉換為 `[1]`
- 最終只剩下一塊重量為 `1` 的石頭。

**样例 2：** 输入：
```Python
stones = [1]
```

输出：
```Python
1
```

解释： 只有一塊石頭，無需進行任何碰撞，直接回傳該石頭的重量 `1`。

**样例 3：** 输入：
```Python
stones = [3, 3]
```

输出：
```Python
0
```

解释： 兩塊石頭重量相同（`3 == 3`），相撞後完全粉碎，沒有任何石頭剩下，故回傳 `0`。

```Python
# 方法一：最大堆（Max-Heap）- 面試最佳解，時間 O(N log N)，空間 O(N)
import heapq
from typing import List

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        # 1. Python 的 heapq 預設為最小堆，將所有石頭重量轉為負數來模擬「最大堆」
        maxHeap = [-s for s in stones]
        heapq.heapify(maxHeap)  # 原地線性建堆，時間 O(N)

        # 2. 只要堆中還有至少兩塊石頭，就持續取出最重的兩塊進行碰撞
        while len(maxHeap) > 1:
            first = heapq.heappop(maxHeap)   # 最重的石頭（負數最小者）
            second = heapq.heappop(maxHeap)  # 第二重的石頭
            
            # 3. 因為兩者皆為負數，若 second > first 代表 |first| > |second|（重量不相等）
            if second > first:
                # 差值 (first - second) 依然為負數，推回最大堆中
                heapq.heappush(maxHeap, first - second)

        # 4. 若堆中還剩一塊石頭，轉回正數回傳；若全部粉碎則補 0 回傳
        return -maxHeap[0] if maxHeap else 0
```


```Python
# 方法二：暴力排序模擬法（Sorting Simulation）- 時間 O(N^2 log N)，空間 O(1)
from typing import List

class Solution:
    def lastStoneWeight(self, stones: List[int]) -> int:
        # 每次迴圈重新由小到大排序，取出最後面（最重）的兩塊石頭
        while len(stones) > 1:
            stones.sort()
            y = stones.pop()  # 最重
            x = stones.pop()  # 第二重
            if y != x:
                stones.append(y - x)
                
        return stones[0] if stones else 0
```

解釋: step1: 題目要求每一回合都動態取出「當前最大值」與「次大值」，並可能將差值放回集合中繼續參與下一輪比較，這是優先佇列（Priority Queue / Max-Heap）最標準的應用場景。 step2: 由於 Python 標準庫 `heapq` 只提供最小堆（Min-Heap），我們將 `stones` 中的每個元素都乘上 `-1`（例如 `8` 變成 `-8`），利用 `heapq.heapify(maxHeap)` 在 `O(N)` 時間內完成建堆。 step3: 進入 `while len(maxHeap) > 1:` 迴圈，連續呼叫兩次 `heappop` 取出最重的 `first`（例如 `-8`）與次重的 `second`（例如 `-7`）。 step4: 若 `first != second`（寫成 `second > first`），代表兩者沒有完全抵銷，將剩餘重量 `first - second`（`-8 - (-7) = -1`）透過 `heappush` 放回堆中；若相等則兩者皆不放回。 step5: 迴圈結束時 `maxHeap` 最多只會剩 `0` 或 `1` 個元素，最後別忘了加上負號 `-maxHeap[0]` 還原為正整數（若堆為空則回傳 `0`）。

`stones = [2, 7, 4, 1, 8, 1]` -> `result = 1`

1. 轉負數並建堆：`maxHeap = [-8, -7, -4, -1, -2, -1]`
    
2. 第 1 輪：pop `-8` 與 `-7`，差值 `-1` 推回 -> `maxHeap = [-4, -2, -1, -1, -1]`
    
3. 第 2 輪：pop `-4` 與 `-2`，差值 `-2` 推回 -> `maxHeap = [-2, -1, -1, -1]`
    
4. 第 3 輪：pop `-2` 與 `-1`，差值 `-1` 推回 -> `maxHeap = [-1, -1, -1]`
    
5. 第 4 輪：pop `-1` 與 `-1`，相等完全粉碎 -> `maxHeap = [-1]`
    
6. 長度為 1 結束迴圈，回傳 `-(-1) = 1`
    

解釋: 如果每回合都對整個陣列重新排序，會把大量時間浪費在已經排好序的石頭上。

透過最大堆（Max-Heap）維護石頭集合，每次取出最重兩塊與塞回新碎石都只需 **`O(log N)`** 的局部調整，大幅降低時間複雜度。

### **LeetCode 1046 - Last Stone Weight（最後一塊石頭的重量）**

這道題目是 **最大堆（Max-Heap / Priority Queue）** 的經典入門題（NeetCode 150 必考題）。問題本身是一道純粹的**貪心模擬題**（每一步的操作都是唯一確定的），考驗如何選擇合適的資料結構來高效率地動態維護「最大值」。

## **解法解析**

這道題的核心在於**動態極值查詢與更新**。

### **思路**

1. **為什麼選擇 Heap（堆積）？**
    
    - 如果使用普通陣列 `list`，每次要找出最重的兩塊石頭，要麼花 `O(N)` 線性掃描兩次，要麼花 `O(N log N)` 重新排序，總共進行約 `N` 輪碰撞後，總時間複雜度會退化為 `O(N^2)` 或 `O(N^2 log N)`。
        
    - 使用 **最大堆（Max-Heap）**，建立初始堆積只需 `O(N)`，之後每一輪碰撞最多執行 **2 次 `heappop` + 1 次 `heappush`**，每次操作僅需 `O(log N)`。
        
2. **Python 模擬最大堆的技巧**
    
    - Python 的 `heapq` 模組僅支援最小堆（Min-Heap）。
        
    - 對於數值型資料，最標準的做法是**存入相反數（`-x`）**：原本最大的數字 `8` 變成最小的 `-8`，自然會浮到堆頂 `maxHeap[0]`。
        
    - 計算碰撞後的剩餘重量時，直接用 `first - second`（例如 `-8 - (-7) = -1`），結果本身就已經是負數，可以直接 `heappush` 回堆中，連正負轉換都省下來。
        
3. **迴圈終止條件**
    
    - 題目規定「最後最多只會剩下一塊石頭」，因此迴圈條件必須寫 `while len(maxHeap) > 1:`。
        
    - 當 `len(maxHeap) == 1` 或 `0` 時跳出迴圈，用三元運算子 `-maxHeap[0] if maxHeap else 0` 處理全部粉碎（堆為空）的邊界情況。
        

## **具體舉例**

假設 `stones = [2, 7, 4, 1, 8, 1]`，我們追蹤每一步最大堆（還原為正數表示方便閱讀）的變化過程：

|**步驟**|**當前最大堆狀態（由大到小概念示意）**|**取出最重 y**|**取出次重 x**|**碰撞結果 (y - x)**|**動作**|
|---|---|---|---|---|---|
|初始化|`[8, 7, 4, 2, 1, 1]`|-|-|-|`heapify` 建堆|
|第 1 輪|`[8, 7, 4, 2, 1, 1]`|`8`|`7`|`1`|推回 `1`|
|第 2 輪|`[4, 2, 1, 1, 1]`|`4`|`2`|`2`|推回 `2`|
|第 3 輪|`[2, 1, 1, 1]`|`2`|`1`|`1`|推回 `1`|
|第 4 輪|`[1, 1, 1]`|`1`|`1`|`0`|完全粉碎，不推回|
|結束|`[1]`|-|-|-|堆長度為 `1`，回傳 `1`|

## **時間與空間複雜度分析**

設 `N` 為初始石頭的數量。

- **時間複雜度：O(N log N)**
    
    - 將陣列轉為負數需要 `O(N)`，`heapq.heapify` 原地建堆需要 `O(N)`。
        
    - 每次碰撞至少會讓石頭總數減少 `1` 塊（若兩塊相等則減少 `2` 塊），因此 `while` 迴圈最多執行 `N - 1` 次。
        
    - 每次迴圈內部最多進行 `3` 次堆積操作（2 次 `pop` + 1 次 `push`），每次操作為 `O(log N)`，故總時間複雜度為 **O(N log N)**。
        
- **空間複雜度：O(N)**
    
    - 需要建立一個長度為 `N` 的陣列 `maxHeap` 來存放負數重量（若在原地直接修改輸入陣列 `stones` 並呼叫 `heapq.heapify(stones)`，則額外空間複雜度可降為 **O(1)**）。
        

## **其他解法**

### **1. 桶排序 / 計數排序法（Bucket Sort）- 當重量範圍較小時為 O(N + W)**

- **思路**：
    
    - 觀察題目限制：每塊石頭的重量範圍為 `1 <= stones[i] <= 1000`（設最大重量為 `W`）。
        
    - 建立一個大小為 `W + 1` 的計數陣列 `buckets`，統計每種重量的石頭各有多少顆。
        
    - 使用指標 `curr_weight` 從最大重量 `W` 往下掃描：
        
        - 若當前重量有偶數顆，兩兩相撞全部抵銷，直接歸零；若有奇數顆，抵銷後會剩下 `1` 顆。
            
        - 剩下的這 `1` 顆必須往下找下一顆次重的石頭 `next_weight` 相撞，產生重量為 `curr_weight - next_weight` 的新石頭並加進對應桶子中。
            
- **時間複雜度**：`O(N + W)`（當 `N` 很大但石頭最大重量 `W` 很小時，效率優於 `O(N log N)`）
    
- **空間複雜度**：`O(W)`
    

### **2. 陣列持續排序法（Sorting Simulation）**

- **思路**：
    
    - 不使用堆積，每次直接呼叫 `stones.sort()` 並 `pop()` 出最後兩個元素相減再放回。
        
- **時間複雜度**：`O(N^2 log N)`（適合資料規模極小 `N <= 30` 時快速撰寫，但面試中不建議作為最終答案）。
    
- **空間複雜度**：`O(1)`
    

## **總結**

- **面試最佳解**：**最大堆（Max-Heap）**，透過「取負數 + `heapq`」實作，時間 `O(N log N)`，邏輯簡潔且不易出錯。
    
- **核心記憶口訣**：**「動態找最大，負數進最小堆」**。
    
- **常見陷阱**：
    
    1. 忘記檢查 `if maxHeap:` 直接回傳 `-maxHeap[0]`，當輸入為 `[2, 2]` 全部粉碎導致堆為空時，會觸發 `IndexError`。
        
    2. 迴圈結束後忘記把負數還原為正數（回傳了 `-1` 而非 `1`）。
        

## **堆積模擬與碰撞消除（Heap / Simulation）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 1046**|Last Stone Weight（最後一塊石頭的重量）|每次取**最重兩塊石頭**相撞抵銷，求最終剩餘重量|**Max-Heap 貪心模擬**，時間 `O(N log N)` 空間 `O(N)`|
|**LeetCode 1049**|Last Stone Weight II（最後一塊石頭的重量 II）|**任意順序**兩兩相撞，求**可能達到的最小剩餘重量**|**0-1 背包動態規劃（DP）**，轉化為將石頭分成總和最接近的兩堆|
|**LeetCode 1167**|Minimum Cost to Connect Sticks（連接棒材的最低費用）|每次合併**最短兩根木棒**（花費為兩者之和），求總最低花費|**Min-Heap 哈夫曼編碼（Huffman Coding）貪心模型**，時間 `O(N log N)`|
|**LeetCode 1962**|Remove Stones to Minimize the Total|執行 `k` 次操作，每次選最多石頭的一堆**移除一半**，求剩餘最小總數|**Max-Heap 模擬**，每次彈出最大值並推回 `x - x // 2`，時間 `O(N + K log N)`|
|**LeetCode 2208**|Minimum Operations to Halve Array Sum|每次選一個數字**減半**，求使陣列總和至少減少一半的最少操作次數|**Max-Heap 模擬**，每次貪心將當前最大元素減半，時間 `O(N log N)`|
|**LeetCode 703**|Kth Largest Element in a Stream|動態資料串流中持續查詢第 `K` 大元素|**固定大小為 `K` 的 Min-Heap**，單次更新 `O(log K)`|
|**LeetCode 973**|K Closest Points to Origin（最接近原點的 K 個點）|找出距離原點最近的 `K` 個座標|**大小為 `K` 的 Max-Heap**（淘汰距離較遠的點），時間 `O(N log K)`|
|**LeetCode 735**|Asteroid Collision（行星碰撞）|一維陣列中正負方向行星相撞消除，求最終剩餘狀態|**單調堆疊（Stack）模擬**，僅當「棧頂向右、當前向左」時發生碰撞|

### **進階延伸題：最後一塊石頭的重量 II（LeetCode 1049）**

面試官考完 `LC 1046` 後，最常順勢追問這道姊妹題：**「如果不再規定每次只能拿最重的兩塊，而是允許你『自由選擇』任意兩塊石頭相撞，最後剩下石頭的『最小可能重量』是多少？」** 這題不能再用貪心（Max-Heap）了！本質上是**把石頭分成兩堆，使兩堆重量差最小**（經典 **0-1 背包問題**）：

Python

```
from typing import List

class Solution:
    def lastStoneWeightII(self, stones: List[int]) -> int:
        total_sum = sum(stones)
        target = total_sum // 2  # 目標是湊出盡可能接近總和一半的重量
        
        # dp[j] 表示容量為 j 的背包最多能裝多少重量的石頭
        dp = [0] * (target + 1)
        
        for stone in stones:
            # 0-1 背包一維優化：必須從大到小倒序遍歷，避免同一塊石頭被重複選取
            for j in range(target, stone - 1, -1):
                dp[j] = max(dp[j], dp[j - stone] + stone)
                
        # 一堆重量為 dp[target]，另一堆為 total_sum - dp[target]，兩者相減即為最小剩餘重量
        return total_sum - 2 * dp[target]
```