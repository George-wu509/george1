
|                                           |     |
| ----------------------------------------- | --- |
| [[#### 乘積最大子陣列 Maximum Product Subarray]] |     |
| [[#### 一維 DP 陣列法解法解釋]]                    |     |
|                                           |     |


#### 乘積最大子陣列 Maximum Product Subarray

LeetCode 152 / LintCode 191 给你一个整数数组 `nums` ，请你找出数组中乘积最大的非空连续子数组（该子数组中至少包含一个数字），并返回该子数组所对应的乘积。

**样例 1：** 输入：
```Python
nums = [2, 3, -2, 4]
```

输出：
```Python
6
```

解释： 子数组 [2, 3] 有最大乘积 6。

**样例 2：** 输入：
```Python
nums = [-2, 0, -1]
```

输出：
```Python
0
```

解释： 结果不能为 2，因为 [-2, -1] 不是连续子数组。

**样例 3：** 输入：
```Python
nums = [-2, 3, -4]
```

输出：
```Python
24
```

解释： 整个数组 [-2, 3, -4] 的乘积为 (-2) * 3 * (-4) = 24。
```Python
# 解法一：一維 DP 陣列（同時記錄最大值與最小值）
def maxProduct(self, nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = nums[0]
    
    for i in range(1, n):
        max_dp[i] = max(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        min_dp[i] = min(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        
    return max(max_dp)
```


```Python
# 解法二：空間優化 O(1) 滾動變數
def maxProduct(self, nums: list[int]) -> int:
    if not nums:
        return 0
    res = cur_max = cur_min = nums[0]
    
    for num in nums[1:]:
        temp_max = cur_max
        cur_max = max(num, temp_max * num, cur_min * num)
        cur_min = min(num, temp_max * num, cur_min * num)
        res = max(res, cur_max)
        
    return res
```

解釋: step1: 因為陣列中包含負數，「負負得正」，當前的**最小值（負數）**乘以下一個負數可能會變成**最大值**，因此需要同時維護 `cur_max` 和 `cur_min`。 step2: 初始化 `res = cur_max = cur_min = nums[0]`。以 `nums = [-2, 3, -4]` 為例：初始 `res = -2, cur_max = -2, cur_min = -2`。 step3: 遍歷 `num = 3` 時：

- `cur_max = max(3, -2*3, -2*3) = 3`
    
- `cur_min = min(3, -2*3, -2*3) = -6`
    
- `res = max(-2, 3) = 3` step4: 遍歷 `num = -4` 時（必須先用 `temp_max` 保存上一輪的 `cur_max`）：
    
- `cur_max = max(-4, 3*(-4), -6*(-4)) = max(-4, -12, 24) = 24`
    
- `cur_min = min(-4, 3*(-4), -6*(-4)) = -12`
    
- `res = max(3, 24) = 24` step5: 迴圈結束，所以答案 = `24`。
    

解釋: 考慮以第 `i` 個元素結尾的最大乘積子陣列，有三種來源：

1. 從 `nums[i]` 重新開始一個新的子陣列（例如前面乘積為 0 或負數）。
    
2. 接在前面「最大乘積」子陣列後面：`cur_max * nums[i]`（當 `nums[i]` 為正數時有利）。
    
3. 接在前面「最小乘積」子陣列後面：`cur_min * nums[i]`（當 `nums[i]` 為負數時，負負得正）。
    

### **LeetCode 152 / LintCode 191 - Maximum Product Subarray（乘積最大子陣列）**

這道題目是 **Kadane 演算法（最大子陣列和）** 的進階變形，屬於經典的 **動態規劃（Dynamic Programming, DP）** 問題。與加法不同，乘法具有「負負得正」以及「乘 0 歸零」的特性，因此不能只記錄局部最大值，必須**同時追蹤局部最大值與局部最小值**。

## **解法解析**

這道題的核心在於處理 **正負號反轉** 與 **0 的截斷效應**。

### **思路**

1. **觀察規律**
    - 如果陣列全是正數，最大乘積就是整個陣列的乘積。
    - 如果遇到 `0`，任何包含 `0` 的子陣列乘積都會變成 `0`，相當於把陣列切割成前後兩段重新計算（狀態轉移中的 `max(num, ...)` 自動處理了重新開始的情況）。
    - 如果遇到 **負數**，原本的最大正數乘積會變成最小負數；反之，原本累積的最小負數乘積，再乘上一個負數就會翻轉成最大正數。
        
2. **發現狀態轉移公式**
    - 定義 `cur_max` 為以當前位置結尾的子陣列最大乘積，`cur_min` 為以當前位置結尾的子陣列最小乘積：
        - `cur_max = max(num, prev_max * num, prev_min * num)`
        - `cur_min = min(num, prev_max * num, prev_min * num)`
        - `res = max(res, cur_max)`
            
3. **代碼執行流程**
    
    - 初始化 `res`、`cur_max`、`cur_min` 皆為 `nums[0]`。
        
    - 從索引 `1` 開始遍歷陣列，每次先將 `cur_max` 存入暫存變數 `temp_max`，避免更新 `cur_max` 後影響 `cur_min` 的計算。
        
    - 同步更新 `cur_max` 與 `cur_min`，並用 `cur_max` 刷新全域最大值 `res`。
        
    - 遍歷完成後返回 `res`。
        

## **具體舉例**

假設 `nums = [2, 3, -2, 4]`，我們的變數變化如下：

|**i**|**num**|**cur_max 計算過程**|**cur_max**|**cur_min**|**res**|
|---|---|---|---|---|---|
|初始化 (`i = 0`)|`2`|初始值|`2`|`2`|`2`|
|`i = 1`|`3`|`max(3, 2*3, 2*3)`|`6`|`3`|`6`|
|`i = 2`|`-2`|`max(-2, 6*(-2), 3*(-2))`|`-2`|`-12`|`6`|
|`i = 3`|`4`|`max(4, (-2)*4, (-12)*4)`|`4`|`-48`|`6`|

最後返回 `6`，表示最大連續子陣列乘積為 **6**（對應子陣列 `[2, 3]`）。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 只需遍歷一次長度為 `n` 的陣列，每次計算 `max` 與 `min` 為 `O(1)`，因此總時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**
    
    - 採用滾動變數優化後，僅需常數個變數（`res`、`cur_max`、`cur_min`、`temp_max`），空間複雜度為 **O(1)**（若使用 `max_dp` 和 `min_dp` 陣列則為 **O(n)**）。
        

## **其他解法**

### **1. 前後綴乘積法（Prefix & Suffix Product）**

- **思路**：
    
    - 在沒有 `0` 的區間內，若負數個數為偶數，整個區間乘積最大；若負數個數為奇數，最大乘積一定出現在「去掉最左邊負數的後綴」或「去掉最右邊負數的前綴」中。
        
    - 從左到右計算前綴積，再從右到左計算後綴積，遇到 `0` 就重置為 `1`，取所有過程中的最大值即可。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(1)
    

### **2. 一維 DP 陣列法**

- **思路**：
    
    - 建立兩個長度為 `n` 的陣列 `max_dp` 和 `min_dp` 記錄每個位置的最值，邏輯更直觀，適合初學者理解狀態轉移。
        
- **時間複雜度**：O(n)
- **空間複雜度**：O(n)
    

### **3. 暴力雙重迴圈法（Brute Force）**

- **思路**：
    
    - 枚舉所有起點 `i` 與終點 `j`，累乘區間 `[i..j]` 並更新最大值（在 LeetCode 上會超時 TLE）。
        
- **時間複雜度**：O(n²)
- **空間複雜度**：O(1)
    

## **總結**

- **面試首選最佳解法**：**O(1) 空間的雙變數動態規劃（同時維護 `cur_max` 與 `cur_min`）**，時間 `O(n)`，空間 `O(1)`。
    
- **常見易錯點**：
    
    1. 忘記記錄 `cur_min`，導致遇到兩個負數時無法「負負得正」。
        
    2. 更新 `cur_max` 後直接用新的 `cur_max` 去算 `cur_min`，沒有先用 `temp_max` 保存舊值。
        
    3. 全陣列只有一個負數（如 `[-2]`）時，誤將 `res` 初始化為 `0` 而非 `nums[0]`。
        

## **LeetCode / LintCode 子陣列與動態規劃類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 152**|Maximum Product Subarray（乘積最大子陣列）|**1D 連續子陣列，含負數與 0，求最大乘積**|**動態規劃：同時維護 `cur_max` 與 `cur_min`（O(n) 時間, O(1) 空間）**|
|**LeetCode 53**|Maximum Subarray（最大子陣列和）|**1D 連續子陣列，求最大總和**|**Kadane 演算法：`cur_max = max(num, cur_max + num)`**|
|**LeetCode 918**|Maximum Sum Circular Subarray（環形子陣列的最大和）|**1D 環形連續子陣列，求最大總和**|**Kadane 變形：`max(一般最大子陣列和, 總和 - 最小子陣列和)`**|
|**LeetCode 713**|Subarray Product Less Than K（乘積小於 K 的子陣列）|**1D 正整數陣列，求乘積小於 K 的連續子陣列個數**|**滑動視窗（Sliding Window / 雙指針）O(n)**|
|**LeetCode 238**|Product of Array Except Self（除自身以外陣列的乘積）|**1D 陣列乘積，不能使用除法**|**前綴積 × 後綴積（Prefix & Suffix Product）O(n)**|
|**LeetCode 560**|Subarray Sum Equals K（和為 K 的子陣列）|**1D 陣列（含負數），求總和等於 K 的子陣列個數**|**前綴和（Prefix Sum）+ 雜湊表（Hash Map）O(n)**|
|**LeetCode 1567**|Maximum Length of Subarray With Positive Product（乘積為正數的最長子陣列長度）|**1D 陣列，求乘積大於 0 的最長連續子陣列長度**|**動態規劃：維護正乘積長度 `pos` 與負乘積長度 `neg`**|
|**LeetCode 300**|Longest Increasing Subsequence（最長遞增子序列）|**1D 非連續子序列，求最長嚴格遞增長度**|**動態規劃 O(n²) / 貪心 + 二分搜尋 O(n log n)**|
|**LeetCode 121**|Best Time to Buy and Sell Stock（買賣股票的最佳時機）|**1D 陣列，單次買賣求最大利潤**|**貪心 / 1D 動態規劃（追蹤歷史最低價）O(n)**|
|**LeetCode 198**|House Robber（打家劫舍）|**1D 線性陣列，不能選相鄰元素，求最大總和**|**動態規劃 `DP[i] = max(DP[i-1], DP[i-2] + nums[i])`**|

### **延伸變形題：乘積為正數的最長子陣列長度（LeetCode 1567）**

給你一個整數陣列 `nums`，請你求出乘積為正數的最長連續子陣列的長度（思路與本題完全一致，只是從追蹤「最大/最小乘積值」變成追蹤「正乘積/負乘積的長度」）：

```Python
class Solution:
    def getMaxLen(self, nums: list[int]) -> int:
        pos_len = 0  # 以當前位置結尾，乘積為正的最長子陣列長度
        neg_len = 0  # 以當前位置結尾，乘積為負的最長子陣列長度
        max_len = 0

        for num in nums:
            if num == 0:
                # 遇到 0，乘積歸零，正負長度全部重置
                pos_len = 0
                neg_len = 0
            elif num > 0:
                # 遇到正數：正長度 +1；若原本有負長度則也 +1，否則維持 0
                pos_len += 1
                neg_len = neg_len + 1 if neg_len > 0 else 0
            else:
                # 遇到負數：正負長度互換並 +1（負負得正、正負得負）
                prev_pos = pos_len
                pos_len = neg_len + 1 if neg_len > 0 else 0
                neg_len = prev_pos + 1

            max_len = max(max_len, pos_len)

        return max_len
```





#### 一維 DP 陣列法解法解釋
```
leetcode的Maximum Product Subarray找出整數陣列中乘積最大的非空連續子陣列，返回該最大乘積. 請用簡易的方式解釋以下的解法如有需要用圖輔助:
```
這段程式碼的核心精神只有四個字：**「負負得正」**。

在一般的「最大子陣列和」（Maximum Subarray）問題中，我們只需要記錄「目前最大值」；但在**乘積**問題裡，一個極小的負數（例如 `-100`）只要再乘上一個負數（例如 `-2`），就會瞬間翻身變成極大的正數（`+200`）。

因此，我們必須同時養兩條線：**`max_dp`（目前最強正數）** 與 **`min_dp`（潛力股負數）**。
```python
def maxProduct(self, nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = nums[0]
    
    for i in range(1, n):
        max_dp[i] = max(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        min_dp[i] = min(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        
    return max(max_dp)
```

```python
def maxProduct(self, nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    
    # 1. 準備兩個筆記本，記錄「以第 i 個數字結尾」時的最大與最小乘積
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = nums[0]
    
    # 2. 從第 2 個數字開始往後推算
    for i in range(1, n):
        # 每次都從「自立門戶」、「乘上前最大」、「乘上前最小」三者中挑最大與最小
        max_dp[i] = max(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        min_dp[i] = min(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        
    # 3. 因為最大乘積可能出現在陣列中間（例如後面遇到 0 歸零），所以要看整張表的最大值
    return max(max_dp)
```

### 一、 視覺化圖解：以 `nums = [2, 3, -2, -4]` 為例

每走到一個新數字 `nums[i]`，子陣列因為必須「連續」，所以只有 **3 種選擇**：

1. **自立門戶**：放棄前面，從自己重新開始（`nums[i]`）
2. **順風車**：接在上一輪的最大值後面（`max_dp[i-1] * nums[i]`）
3. **逆轉勝**：接在上一輪的最小值後面（`min_dp[i-1] * nums[i]`）
```
陣列 nums:    [   2   ,      3      ,      -2      ,      -4      ]
                 │           │             │              │
max_dp (最大):    2  ──×3──► 6  ──自立──►  -2  ──逆轉勝──► 48  (由 min_dp 的 -12 × -4 翻身！)
                 │           │             │              │
min_dp (最小):    2  ──自立──► 3  ──×(-2)─► -12 ──×(-4)──► -4  (暫時墊底，等待下個負數)
```

#### 逐步拆解表

| **索引 i** | **當前數字 nums[i]** | **3 個候選算式 (自己, 前最大×自己, 前最小×自己)**     | **max_dp[i] (取最大)** | **min_dp[i] (取最小)** | **白話文發生了什麼事？**                                        |
| -------- | ---------------- | ------------------------------------ | ------------------- | ------------------- | ----------------------------------------------------- |
| **0**    | `2`              | 初始值                                  | **2**               | **2**               | 只有一個數，最大最小都是自己                                        |
| **1**    | `3`              | `3`, `2×3=6`, `2×3=6`                | **6**               | **3**               | 乘起來變大，最大變成 `6`                                        |
| **2**    | `-2`             | `-2`, `6×(-2)=-12`, `3×(-2)=-6`      | **-2**              | **-12**             | 遇到負數，`max_dp` 縮水成 `-2`；但 `min_dp` 存下了 **`-12`** 這顆未爆彈 |
| **3**    | `-4`             | `-4`, `(-2)×(-4)=8`, `(-12)×(-4)=48` | **48**              | **-4**              | **關鍵時刻！** `-4` 乘上上一輪的 `min_dp (-12)`，負負得正變成 **`48`**  |

最後一步 `return max(max_dp)`，從 `[2, 6, -2, 48]` 中挑出最大的 **`48`** 作為答案。

### 二、 程式碼逐行對應

```python
def maxProduct(self, nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    
    # 1. 準備兩個筆記本，記錄「以第 i 個數字結尾」時的最大與最小乘積
    max_dp = [0] * n
    min_dp = [0] * n
    max_dp[0] = min_dp[0] = nums[0]
    
    # 2. 從第 2 個數字開始往後推算
    for i in range(1, n):
        # 每次都從「自立門戶」、「乘上前最大」、「乘上前最小」三者中挑最大與最小
        max_dp[i] = max(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        min_dp[i] = min(nums[i], max_dp[i - 1] * nums[i], min_dp[i - 1] * nums[i])
        
    # 3. 因為最大乘積可能出現在陣列中間（例如後面遇到 0 歸零），所以要看整張表的最大值
    return max(max_dp)
```

### 三、 複雜度分析

- **時間複雜度：$O(n)$** — 只需要從頭到尾掃描陣列一次。
    
- **空間複雜度：$O(n)$** — 額外開了 `max_dp` 與 `min_dp` 兩個長度為 `n` 的陣列。