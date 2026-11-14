
LeetCode 4 / LintCode 65 給定兩個大小分別為 `m` 和 `n` 的正序（從小到大）陣列 `nums1` 和 `nums2`。請你找出並返回這兩個正序陣列的中位數。

演算法的時間複雜度應該為 $O(\log(m + n))$。

**範例 1：** 輸入：
```Python
nums1 = [1, 3]
nums2 = [2]
```

輸出：
```Python
2.00000
```

解釋： 合併陣列 = `[1, 2, 3]`，中位數 `2`

**範例 2：** 輸入：
```Python
nums1 = [1, 2]
nums2 = [3, 4]
```

輸出：
```Python
2.50000
```

解釋： 合併陣列 = `[1, 2, 3, 4]`，中位數 `(2 + 3) / 2 = 2.5`

**範例 3：** 輸入：
```Python
nums1 = [1, 3, 8]
nums2 = [2, 4, 5, 6]
```

輸出：
```Python
4.00000
```

解釋： 合併陣列 = `[1, 2, 3, 4, 5, 6, 8]`，中位數 `4`

### 解法一：二分搜尋切割法（Optimal - $O(\log(\min(m, n)))$）

```Python
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        # 確保 nums1 是較短的陣列，將二分搜尋範圍縮小到 min(m, n)，同時避免 j 越界
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1
        
        m, n = len(nums1), len(nums2)
        left, right = 0, m
        half_len = (m + n + 1) // 2  # 左半部分應該包含的元素總數
        
        while left <= right:
            i = (left + right) // 2  # nums1 的切割點（左半部有 i 個元素）
            j = half_len - i         # nums2 的切割點（左半部有 j 個元素）
            
            # 處理邊界情況：切割點在最左邊用 -inf，在最右邊用 +inf
            nums1_left_max = float('-inf') if i == 0 else nums1[i - 1]
            nums1_right_min = float('inf') if i == m else nums1[i]
            nums2_left_max = float('-inf') if j == 0 else nums2[j - 1]
            nums2_right_min = float('inf') if j == n else nums2[j]
            
            # 檢查是否滿足「左半部所有元素 <= 右半部所有元素」
            if nums1_left_max <= nums2_right_min and nums2_left_max <= nums1_right_min:
                # 總長度為奇數：左半部最大值即為中位數
                if (m + n) % 2 == 1:
                    return float(max(nums1_left_max, nums2_left_max))
                # 總長度為偶數：左半部最大值與右半部最小值的平均
                else:
                    return (max(nums1_left_max, nums2_left_max) + min(nums1_right_min, nums2_right_min)) / 2.0
            elif nums1_left_max > nums2_right_min:
                # nums1 左邊太大，切割點 i 需要往左移
                right = i - 1
            else:
                # nums2 左邊太大（即 nums1 右邊太小），切割點 i 需要往右移
                left = i + 1
```


### 解法二：雙指針合併法（直觀解法 - $O(m + n)$）

```Python
from typing import List

class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m, n = len(nums1), len(nums2)
        total = m + n
        p1, p2 = 0, 0
        prev, curr = 0, 0
        
        # 只需要走到中間位置 (total // 2) 即可停止
        for _ in range(total // 2 + 1):
            prev = curr
            if p1 < m and (p2 >= n or nums1[p1] <= nums2[p2]):
                curr = nums1[p1]
                p1 += 1
            else:
                curr = nums2[p2]
                p2 += 1
                
        if total % 2 == 1:
            return float(curr)
        return (prev + curr) / 2.0
```

解釋:

- `step1`: 核心目標不是「合併陣列」，而是在兩個陣列中各切一刀（切割點 `i` 和 `j`），把所有元素分成**數量相等的左半部與右半部**。
    
- `step2`: 確保 `nums1` 是較短的陣列（長度 `m <= n`），這樣對 `nums1` 做二分搜尋時，搜尋區間為 `[0, m]`，且 `j = (m + n + 1) // 2 - i` 永遠不會小於 0 或大於 `n`。
    
- `step3`: 以 `nums1 = [1, 3, 8]`（`m=3`）, `nums2 = [2, 4, 5, 6]`（`n=4`）為例，`half_len = (3 + 4 + 1) // 2 = 4`，初始 `left = 0, right = 3`。
    
- `step4`: 第 1 輪二分：`i = (0 + 3) // 2 = 1`，`j = 4 - 1 = 3`。切割為 `[1 | 3, 8]` 與 `[2, 4, 5 | 6]`。
    
    - 檢查交叉大小：`nums1_left_max (1) <= nums2_right_min (6)` 成立，但 `nums2_left_max (5) <= nums1_right_min (3)` **不成立**（5 > 3），代表 `nums1` 切太少了，更新 `left = i + 1 = 2`。
        
- `step5`: 第 2 輪二分：`i = (2 + 3) // 2 = 2`，`j = 4 - 2 = 2`。切割為 `[1, 3 | 8]` 與 `[2, 4 | 5, 6]`。
    
    - 檢查交叉大小：`nums1_left_max (3) <= nums2_right_min (5)` 且 `nums2_left_max (4) <= nums1_right_min (8)` **全部成立**！
        
- `step6`: 總長度 `3 + 4 = 7` 為奇數，中位數直接取左半部最大值 `max(3, 4) = 4.0`。
    

解釋: 中位數的本質是「將一個有序集合劃分為兩個長度相等的子集，且左子集的最大值小於等於右子集的最小值」。

只要在較短陣列上確定了切割位置 `i`，較長陣列的切割位置 `j` 就會被唯一確定（`i + j = (m + n + 1) // 2`），因此問題就轉化為在 `[0, m]` 區間內尋找合法 `i` 的**二分搜尋**問題。

### **LeetCode 4 / LintCode 65 - Median of Two Sorted Arrays（尋找兩個正序陣列的中位數）**

這道題目是 **二分搜尋（Binary Search）** 的經典進階題（Hard 難度），也是 Blind 75 與 NeetCode 150 的必考題。題目要求在 $O(\log(m + n))$ 時間內找出兩個有序陣列的中位數，這直接暗示了**不能合併陣列**（合併至少需要 $O(m + n)$），必須透過二分搜尋每次排除一半的搜尋空間。

## **解法解析**

這道題可以轉化為 **尋找正確的陣列分割線（Partition）** 來求解。

### **思路**

1. **中位數的分割定義**
    
    - 假設兩個陣列合併後的總長度為 `m + n`。
        
    - 我們希望找到一條分割線，將 `nums1` 分成左右兩半（左邊 `i` 個元素），同時將 `nums2` 分成左右兩半（左邊 `j` 個元素）。
        
    - 為了讓左半部與右半部元素數量相等（奇數時讓左半部多 1 個），`i` 和 `j` 必須滿足：
        
        $$i + j = \lfloor \frac{m + n + 1}{2} \rfloor$$
        
    - 只要決定了 `i`，就能直接算出 `j = half_len - i`。
        
2. **合法分割的條件**
    
    - 因為兩個陣列本身已經是正序的，同一陣列內部自然滿足「左 <= 右」（即 `nums1[i-1] <= nums1[i]` 且 `nums2[j-1] <= nums2[j]`）。
        
    - 因此只需要驗證**跨陣列的交叉邊界**：
        
        - `nums1[i - 1] <= nums2[j]`（`nums1` 左側最大值 $\le$ `nums2` 右側最小值）
            
        - `nums2[j - 1] <= nums1[i]`（`nums2` 左側最大值 $\le$ `nums1` 右側最小值）
            
3. **二分調整方向與邊界處理**
    
    - 若 `nums1[i - 1] > nums2[j]`：表示 `nums1` 左邊拿太多、數值太大，需要縮小 `i`（`right = i - 1`）。
        
    - 若 `nums2[j - 1] > nums1[i]`：表示 `nums1` 左邊拿太少、`nums1[i]` 太小，需要增大 `i`（`left = i + 1`）。
        
    - **邊界極值技巧**：當 `i == 0` 或 `j == 0` 時，代表該陣列左側沒有元素，將左側最大值設為 `-inf`；當 `i == m` 或 `j == n` 時，代表右側沒有元素，將右側最小值設為 `+inf`，即可統一判斷邏輯，無需寫繁瑣的 `if-else`。
        

## **具體舉例**

假設 `nums1 = [1, 3, 8]`（`m = 3`），`nums2 = [2, 4, 5, 6]`（`n = 4`），`half_len = (3 + 4 + 1) // 2 = 4`，變數變化如下：

|**輪次**|**left, right**|**i**|**j**|**nums1 分割 (L1 \| R1)**|**nums2 分割 (L2 \| R2)**|**條件檢查 (L1<=R2 且 L2<=R1)**|**動作**|
|---|---|---|---|---|---|---|---|
|第 1 輪|`0, 3`|`1`|`3`|`[1 \| 3, 8]`|`[2, 4, 5 \| 6]`|`1 <= 6` (True), `5 <= 3` (False)|`L2 > R1`，`left = 2`|
|第 2 輪|`2, 3`|`2`|`2`|`[1, 3 \| 8]`|`[2, 4 \| 5, 6]`|`3 <= 5` (True), `4 <= 8` (True)|**滿足條件**，回傳 `max(3, 4) = 4.0`|

## **時間與空間複雜度分析**

- **時間複雜度：$O(\log(\min(m, n)))$**
    
    - 我們先保證 `nums1` 為較短的陣列（長度為 $\min(m, n)$），並僅在 `[0, min(m, n)]` 區間進行二分搜尋，每次迭代將搜尋範圍減半，因此時間複雜度為 $O(\log(\min(m, n)))$，優於題目要求的 $O(\log(m + n))$。
        
- **空間複雜度：$O(1)$**
    
    - 僅使用常數個指標與邊界變數（`left`, `right`, `i`, `j` 等），沒有建立任何額外陣列。
        

## **其他解法**

### **1. 尋找第 K 小元素法（分治 / 二分排除法）**

- **思路**：
    
    - 將「求中位數」轉化為更通用的問題：「在兩個有序陣列中尋找第 `k` 小的數字」。
        
    - 每次比較 `nums1[k//2 - 1]` 與 `nums2[k//2 - 1]`，較小的那一段前 `k//2` 個元素絕不可能是第 `k` 小，直接捨棄並將 `k` 減去排除的數量。
        
- **時間複雜度**：$O(\log(m + n))$
    
- **空間複雜度**：$O(1)$（迭代寫法）或 $O(\log(m + n))$（遞迴堆疊）
    

### **2. 雙指針線性掃描（Two Pointers）**

- **思路**：
    
    - 不需要真正開新陣列合併，只需用兩個指針 `p1`、`p2` 模擬合併過程，走到第 `(m + n) // 2` 步時記錄下中間的兩個數值即可。
        
- **時間複雜度**：$O(m + n)$
    
- **空間複雜度**：$O(1)$
    

### **3. 暴力合併後排序**

- **思路**：
    
    - 直接將兩個陣列拼接 `sorted(nums1 + nums2)`，再根據總長度的奇偶性取中間索引。
        
- **時間複雜度**：$O((m + n)\log(m + n))$
    
- **空間複雜度**：$O(m + n)$
    

## **總結**

- **面試最優解**：**二分搜尋切割法**，時間 $O(\log(\min(m, n)))$、空間 $O(1)$，完全符合題目對數級別時間複雜度的嚴格限制。
    
- **面試保底思路**：若一時想不起切割邊界條件，可先提出 **雙指針掃描法**（時間 $O(m + n)$、空間 $O(1)$）確保正確性，再進一步優化至二分切割法。
    
- **易錯細節**：
    
    1. 忘記先交換 `nums1` 與 `nums2` 確保 `m <= n`，導致 `j = half_len - i` 出現負數或越界。
        
    2. `half_len` 寫成 `(m + n) // 2` 而非 `(m + n + 1) // 2`，導致奇數長度時中位數落在右半部而非左半部。
        

## **Blind 75 + NeetCode 150 二分搜尋與中位數類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 4**|Median of Two Sorted Arrays（尋找兩個正序陣列的中位數）|**雙有序陣列二分搜尋**|**在較短陣列上二分尋找分割線 `i`，時間 $O(\log(\min(m, n)))$**|
|**LeetCode 704**|Binary Search（二分查找）|**基礎 1D 有序陣列二分**|**標準左右閉區間 `[left, right]` 二分搜尋，時間 $O(\log n)$**|
|**LeetCode 74**|Search a 2D Matrix（搜索二維矩陣）|**2D 轉 1D 有序陣列二分**|**將二維座標映射為一維索引 `mid // cols, mid % cols`，時間 $O(\log(m \cdot n))$**|
|**LeetCode 875**|Koko Eating Bananas（愛吃香蕉的珂珂）|**答案空間二分（Binary Search on Answer）**|**二分猜測吃香蕉速度 `k`，用 `check(k)` 驗證總時間，時間 $O(n \log(\max(piles)))$**|
|**LeetCode 153**|Find Minimum in Rotated Sorted Array（尋找旋轉排序陣列中的最小值）|**旋轉有序陣列二分**|**比較 `nums[mid]` 與 `nums[right]` 判斷最小值所在半區，時間 $O(\log n)$**|
|**LeetCode 33**|Search in Rotated Sorted Array（搜索旋轉排序陣列）|**旋轉有序陣列二分**|**每次切分必有一半完全有序，判斷 `target` 是否在有序區間內，時間 $O(\log n)$**|
|**LeetCode 981**|Time Based Key-Value Store（基於時間的鍵值存儲）|**設計題 + 二分搜尋上界**|**HashMap 存時間戳列表，查詢時二分找 `<= timestamp` 的最大值，時間 $O(\log n)$**|
|**LeetCode 295**|Find Median from Data Stream（數據流的中位數）|**動態數據流求中位數**|**雙堆（最大堆存較小一半 + 最小堆存較大一半），插入 $O(\log n)$、查詢 $O(1)$**|

### 延伸變形題：尋找兩個正序陣列中的第 K 小元素（LintCode 65 通用解法）

若題目改為「求兩個正序陣列合併後的第 `k` 小元素（`1 <= k <= m + n`）」，可以使用每次排除 `k // 2` 個元素的二分寫法：

Python

```
from typing import List

class Solution:
    def findKthSortedArrays(self, nums1: List[int], nums2: List[int], k: int) -> int:
        m, n = len(nums1), len(nums2)
        idx1, idx2 = 0, 0  # 記錄兩個陣列目前已被排除的起始偏移量
        
        while True:
            # 邊界情況 1：nums1 已經全部排除，直接從 nums2 取第 k 個
            if idx1 == m:
                return nums2[idx2 + k - 1]
            # 邊界情況 2：nums2 已經全部排除，直接從 nums1 取第 k 個
            if idx2 == n:
                return nums1[idx1 + k - 1]
            # 邊界情況 3：k 縮小到 1，直接比較兩陣列當前最小元素的較小者
            if k == 1:
                return min(nums1[idx1], nums2[idx2])
            
            # 正常情況：各自往前探測 k // 2 步（注意不能超過陣列總長度）
            step = k // 2
            new_idx1 = min(idx1 + step, m) - 1
            new_idx2 = min(idx2 + step, n) - 1
            
            if nums1[new_idx1] <= nums2[new_idx2]:
                # 排除 nums1 中 [idx1 .. new_idx1] 這段元素
                k -= (new_idx1 - idx1 + 1)
                idx1 = new_idx1 + 1
            else:
                # 排除 nums2 中 [idx2 .. new_idx2] 這段元素
                k -= (new_idx2 - idx2 + 1)
                idx2 = new_idx2 + 1
```