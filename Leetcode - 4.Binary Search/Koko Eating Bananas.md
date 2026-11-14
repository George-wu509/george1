
LeetCode 875 / LintCode 1492 珂珂喜歡吃香蕉。這裡有 `n` 堆香蕉，第 `i` 堆中有 `piles[i]` 根香蕉。警衛已經離開了，將在 `h` 小時後回來。

珂珂可以決定她吃香蕉的速度 `k` （單位：根/小時）。每個小時，她將會選擇一堆香蕉，從中吃掉 `k` 根。如果這堆香蕉少於 `k` 根，她將吃掉這堆的所有香蕉，然後這一小時內不會再吃更多的香蕉。

珂珂喜歡慢慢吃，但仍然想在警衛回來前吃掉所有的香蕉。請返回她可以在 `h` 小時內吃掉所有香蕉的最小速度 `k`（`k` 為整數）。

**範例 1：** 輸入：

```Python
piles = [3, 6, 7, 11]
h = 8
```

輸出：
```Python
4
```

解釋： 當速度 `k = 4` 時： 第 1 堆（3 根）需 `1` 小時 第 2 堆（6 根）需 `2` 小時 第 3 堆（7 根）需 `2` 小時 第 4 堆（11 根）需 `3` 小時 總共 `1 + 2 + 2 + 3 = 8` 小時，剛好在 `h = 8` 小時內吃完

**範例 2：** 輸入：
```Python
piles = [30, 11, 23, 4, 20]
h = 5
```

輸出：
```Python
30
```

解釋： 一共有 5 堆香蕉，且只有 `h = 5` 小時，每小時最多只能吃一堆，因此速度必須至少等於最大堆的數量 `30`

**範例 3：** 輸入：
```Python

piles = [30, 11, 23, 4, 20]
h = 6
```

輸出：
```Python
23
```

解釋： 當速度 `k = 23` 時，各堆花費時間為 `2 + 1 + 1 + 1 + 1 = 6` 小時，滿足 `h = 6` 小時內吃完

### 解法一：二分搜尋答案空間（Optimal - $O(n \log(\max(\text{piles})))$）

```Python
from typing import List
import math

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 最小速度為 1，最大速度為香蕉最多那一堆的數量
        left, right = 1, max(piles)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2  # 猜測的吃香蕉速度 k
            
            # 計算以速度 mid 吃完所有香蕉需要的總小時數（無條件進位）
            hours = 0
            for p in piles:
                hours += (p + mid - 1) // mid  # 等同於 math.ceil(p / mid)
            
            if hours <= h:
                # 在規定時間內吃得完，記錄當前可行解，並嘗試更慢的速度
                ans = mid
                right = mid - 1
            else:
                # 時間超過 h，代表吃太慢了，必須提高速度
                left = mid + 1
                
        return ans
```


### 解法二：暴力線性枚舉（Brute Force - $O(n \cdot \max(\text{piles}))$）


```Python
from typing import List

class Solution:
    def minEatingSpeed(self, piles: List[int], h: int) -> int:
        # 從速度 1 開始逐一測試，第一個滿足總時間 <= h 的即為最小速度
        for k in range(1, max(piles) + 1):
            hours = sum((p + k - 1) // k for p in piles)
            if hours <= h:
                return k
        return max(piles)
```

解釋:

- `step1`: 確定速度 `k` 的搜尋範圍：每小時至少吃 `1` 根（`left = 1`），最多吃 `max(piles)` 根（`right = max(piles)`，因為超過最大堆數量也不會加快一小時吃多堆）。
    
- `step2`: 計算單堆花費時間的公式：堆中有 `p` 根香蕉、速度為 `k` 時，所需小時數為 $\lceil p / k \rceil$，整數寫法為 `(p + k - 1) // k`（避免浮點數誤差）。
    
- `step3`: 以 `piles = [3, 6, 7, 11]`, `h = 8` 為例，初始 `left = 1`, `right = 11`。
    
- `step4`: 第 1 輪二分：`mid = (1 + 11) // 2 = 6`。計算總時間：$\lceil 3/6 \rceil + \lceil 6/6 \rceil + \lceil 7/6 \rceil + \lceil 11/6 \rceil = 1 + 1 + 2 + 2 = 6 \le 8$。吃得完！記錄 `ans = 6`，並嘗試找更小的速度 `right = 5`。
    
- `step5`: 第 2 輪二分：`mid = (1 + 5) // 2 = 3`。計算總時間：$\lceil 3/3 \rceil + \lceil 6/3 \rceil + \lceil 7/3 \rceil + \lceil 11/3 \rceil = 1 + 2 + 3 + 4 = 10 > 8$。吃不完！速度太慢，更新 `left = 4`。
    
- `step6`: 第 3 輪二分：`mid = (4 + 5) // 2 = 4`。計算總時間：$1 + 2 + 2 + 3 = 8 \le 8$。吃得完！記錄 `ans = 4`，`right = 3`。此時 `left > right` 迴圈結束，回傳 `4`。
    

解釋: 這道題是典型的 **「對答案進行二分搜尋（Binary Search on Answer）」**。

吃香蕉的速度 `k` 與總耗時 `hours(k)` 之間存在**單調性**：速度 `k` 越大，總耗時越短。因此只要在 `[1, max(piles)]` 的答案區間內二分猜測速度 `k`，透過 `hours <= h` 判斷可行性，就能快速收斂出最小合法速度。

### **LeetCode 875 / LintCode 1492 - Koko Eating Bananas（愛吃香蕉的珂珂）**

這道題目是 **二分搜尋答案空間（Binary Search on Answer）** 的代表題，也是 NeetCode 150 的必考題。與傳統在「已排序陣列中找目標值」不同，本題的輸入陣列 `piles` **並未排序**，我們二分搜尋的對象是「**可能的答案範圍**」。

## **解法解析**

這道題可以轉化為 **尋找滿足條件的左邊界（Lower Bound）** 來求解。

### **思路**

1. **界定答案的上下界（Search Space）**
    
    - **下界 `left = 1`**：題目要求每小時至少吃 1 根香蕉。
        
    - **上界 `right = max(piles)`**：因為題目規定「如果這堆香蕉少於 `k` 根，吃完這堆後這一小時內不會再吃其他堆」，所以當 `k = max(piles)` 時，每一堆都只需要 1 小時，總共花費 `len(piles)` 小時（題目保證 `h >= len(piles)`），再把 `k` 調大也不會減少總時間。
        
2. **單調性與判定函數 `canFinish(k)`**
    
    - 對於任意速度 `k`，吃完所有香蕉的總小時數為：
        
        $$\text{hours}(k) = \sum_{p \in \text{piles}} \left\lceil \frac{p}{k} \right\rceil$$
        
    - 當 `k` 增大時，`hours(k)` 單調遞減（不會變大）。
        
    - 我們要找的是**最小的 `k`**，使得 `hours(k) <= h`。
        
3. **二分收斂方向**
    
    - 若 `hours(mid) <= h`：代表 `mid` 是一個**可行解**，但我們想找更小的 `k`，因此先記錄 `ans = mid`，然後向左收縮搜尋空間（`right = mid - 1`）。
        
    - 若 `hours(mid) > h`：代表 `mid` 太慢、無法在 `h` 小時內吃完，必須向右提高速度（`left = mid + 1`）。
        

## **具體舉例**

假設 `piles = [3, 6, 7, 11]`，`h = 8`，初始 `left = 1, right = 11`，變數變化如下：

|**輪次**|**left, right**|**mid (速度 k)**|**各堆耗時 ⌈p/k⌉**|**總耗時 hours**|**與 h = 8 比較**|**動作**|
|---|---|---|---|---|---|---|
|第 1 輪|`1, 11`|`6`|`1 + 1 + 2 + 2`|`6`|`6 <= 8` (可行)|`ans = 6`，`right = 5`|
|第 2 輪|`1, 5`|`3`|`1 + 2 + 3 + 4`|`10`|`10 > 8` (超時)|`left = 4`|
|第 3 輪|`4, 5`|`4`|`1 + 2 + 2 + 3`|`8`|`8 <= 8` (可行)|`ans = 4`，`right = 3`|
|結束|`4, 3`|—|—|—|`left > right`|終止迴圈，回傳 `4`|

## **時間與空間複雜度分析**

- **時間複雜度：$O(n \log M)$**
    
    - 設 `n = len(piles)` 為香蕉堆數，`M = max(piles)` 為最大堆的香蕉數量。
        
    - 二分搜尋的區間長度為 $M$，共需執行 $O(\log M)$ 次二分迭代；每次迭代需要遍歷整個陣列計算總小時數，耗時 $O(n)$，因此總時間複雜度為 $O(n \log M)$。
        
- **空間複雜度：$O(1)$**
    
    - 僅使用常數個變數（`left`, `right`, `mid`, `ans`, `hours`），不需要額外的輔助陣列。
        

## **其他解法**

### **1. 左閉右開區間二分寫法（不需額外 `ans` 變數）**

- **思路**：
    
    - 設定 `left = 1, right = max(piles)`，迴圈條件使用 `while left < right:`。
        
    - 當 `hours <= h` 時，令 `right = mid`（保留 `mid` 作為候選答案）；當 `hours > h` 時，令 `left = mid + 1`。迴圈結束時 `left == right` 即為最小速度。
        
- **時間複雜度**：$O(n \log M)$
    
- **空間複雜度**：$O(1)$
    

### **2. 數學下界優化（縮小二分起點）**

- **思路**：
    
    - 因為總共至少有 `sum(piles)` 根香蕉要在 `h` 小時內吃完，平均每小時至少要吃 $\lceil \text{sum(piles)} / h \rceil$ 根，所以 `left` 可以直接從 `max(1, sum(piles) // h)` 開始，減少二分迭代次數。
        
- **時間複雜度**：$O(n \log M)$（常數更優）
    
- **空間複雜度**：$O(1)$
    

## **總結**

- **面試最優解**：**二分搜尋答案空間**，時間 $O(n \log(\max(\text{piles})))$、空間 $O(1)$。
    
- **核心模板口訣**：當題目出現「**求滿足某條件的最小值（或最大值）**」，且答案具備「單調性」時，直接套用**二分答案 + `check(mid)` 驗證函數**。
    
- **易錯細節**：
    
    1. **向上取整寫法**：在 Python 中寫 `(p + mid - 1) // mid` 比 `math.ceil(p / mid)` 更快且完全避免大數浮點數精度問題。
        
    2. **上界設定**：`right` 必須設為 `max(piles)`，不能設為 `len(piles)`。
        

## **Blind 75 + NeetCode 150 二分答案與陣列二分問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 875**|Koko Eating Bananas（愛吃香蕉的珂珂）|**答案空間二分（求最小速度）**|**二分猜測速度 `k`，計算向上取整總時數 `<= h`，時間 $O(n \log M)$**|
|**LeetCode 1011**|Capacity To Ship Packages Within D Days（在 D 天內送達包裹的能力）|**答案空間二分（求最小載重）**|**二分船隻載重 `[max(weights), sum(weights)]`，貪心模擬裝船天數**|
|**LeetCode 410**|Split Array Largest Sum（分割陣列的最大值）|**答案空間二分（最小化最大值）**|**二分猜測子陣列最大和，貪心統計所需分段數 `<= k`**|
|**LeetCode 704**|Binary Search（二分查找）|**基礎 1D 有序陣列二分**|**標準左右閉區間 `[left, right]` 二分搜尋，時間 $O(\log n)$**|
|**LeetCode 74**|Search a 2D Matrix（搜索二維矩陣）|**2D 轉 1D 有序陣列二分**|**將二維座標映射為一維索引 `mid // cols, mid % cols`，時間 $O(\log(m \cdot n))$**|
|**LeetCode 153**|Find Minimum in Rotated Sorted Array（尋找旋轉排序陣列中的最小值）|**旋轉有序陣列二分**|**比較 `nums[mid]` 與 `nums[right]` 判斷最小值所在半區，時間 $O(\log n)$**|
|**LeetCode 33**|Search in Rotated Sorted Array（搜索旋轉排序陣列）|**旋轉有序陣列二分**|**每次切分必有一半完全有序，判斷 `target` 是否在有序區間內，時間 $O(\log n)$**|
|**LeetCode 981**|Time Based Key-Value Store（基於時間的鍵值存儲）|**設計題 + 二分搜尋上界**|**HashMap 存時間戳列表，查詢時二分找 `<= timestamp` 的最大值，時間 $O(\log n)$**|

### 延伸變形題：在 D 天內送達包裹的能力（LeetCode 1011 - 姐妹題）

傳送帶上的包裹必須在 `days` 天內從一個港口運送到另一個港口。第 `i` 個包裹的重量為 `weights[i]`。每一天，我們都會按給出重量（`weights`）的順序往船上裝載包裹（**不能打亂順序，且不能把單個包裹拆開**）。我們裝載的重量不會超過船的最大運載重量。返回能在 `days` 天內將傳送帶上的所有包裹送達的船的最低運載能力：

Python

```
from typing import List

class Solution:
    def shipWithinDays(self, weights: List[int], days: int) -> int:
        # 最小載重必須能裝下最重的單個包裹；最大載重為一天裝完所有包裹
        left, right = max(weights), sum(weights)
        ans = right
        
        while left <= right:
            mid = left + (right - left) // 2  # 猜測的船隻載重上限
            
            # 模擬按順序裝船需要的天數
            needed_days = 1
            current_load = 0
            for w in weights:
                if current_load + w > mid:
                    needed_days += 1
                    current_load = 0
                current_load += w
                
            if needed_days <= days:
                ans = mid
                right = mid - 1  # 載重夠用，嘗試更小的載重
            else:
                left = mid + 1   # 天數超過限制，必須增加載重
                
        return ans
```