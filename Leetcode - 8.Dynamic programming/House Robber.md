
打家劫舍 House Robber

LeetCode 198 / LintCode 392 你是一个专业的小偷，计划偷窃沿街的房屋。每间房内都藏有一定的现金，影响你偷窃的唯一制约因素就是相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警**。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **不触动警报装置的情况下** ，一夜之内能够偷窃到的最高金额。

**样例 1：** 输入：
```Python
nums = [1, 2, 3, 1]
```

输出：

```Python
4
```

解释： 偷窃 1 号房屋 (金额 = 1) ，然后偷窃 3 号房屋 (金额 = 3)。 偷窃到的最高金额 = 1 + 3 = 4 。

**样例 2：** 输入：
```Python
nums = [2, 7, 9, 3, 1]
```

输出：
```Python
12
```

解释： 偷窃 1 号房屋 (金额 = 2), 偷窃 3 号房屋 (金额 = 9)，接着偷窃 5 号房屋 (金额 = 1)。 偷窃到的最高金额 = 2 + 9 + 1 = 12 。

**样例 3：** 输入：
```Python
nums = [2, 1, 1, 2]
```

输出：
```Python
4
```

解释： 偷窃 1 号房屋 (金额 = 2) 和 4 号房屋 (金额 = 2)，中間跳過兩間房屋，最高金額 = 2 + 2 = 4。

```Python
# 解法一：一維 DP 陣列
def rob(self, nums: list[int]) -> int:
    if not nums:
        return 0
    n = len(nums)
    if n == 1:
        return nums[0]
        
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    
    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
        
    return dp[-1]
```


```Python
# 解法二：空間優化 O(1) 滾動變數
def rob(self, nums: list[int]) -> int:
    prev_two = 0  # 代表 dp[i-2]
    prev_one = 0  # 代表 dp[i-1]
    
    for num in nums:
        temp = max(prev_one, prev_two + num)
        prev_two = prev_one
        prev_one = temp
        
    return prev_one
```

解釋: step1: 狀態轉移方程為 `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`，因為每次計算只依賴前兩個狀態，所以只需用 `prev_two` 和 `prev_one` 兩個變數滾動更新。 step2: 初始化 `prev_two = 0, prev_one = 0`。以 `nums = [2, 7, 9, 3, 1]` 為例： step3: `num = 2`：`temp = max(0, 0 + 2) = 2` → `prev_two = 0, prev_one = 2` step4: `num = 7`：`temp = max(2, 0 + 7) = 7` → `prev_two = 2, prev_one = 7` step5: `num = 9`：`temp = max(7, 2 + 9) = 11` → `prev_two = 7, prev_one = 11` step6: `num = 3`：`temp = max(11, 7 + 3) = 11` → `prev_two = 11, prev_one = 11` step7: `num = 1`：`temp = max(11, 11 + 1) = 12` → `prev_two = 11, prev_one = 12` step8: 迴圈結束，返回 `prev_one = 12`。

解釋: 考慮走到第 `i` 間房屋時，只有兩種決策（選或不選）：

1. **不偷第 `i` 間房屋**：最大金額就等於前 `i-1` 間房屋能偷到的最大金額，即 `dp[i-1]`。
    
2. **偷第 `i` 間房屋**：因為不能偷相鄰的第 `i-1` 間，所以最大金額等於前 `i-2` 間的最大金額加上第 `i` 間的金額，即 `dp[i-2] + nums[i]`。 兩者取最大值：`dp[i] = max(dp[i-1], dp[i-2] + nums[i])`。
    

### **LeetCode 198 / LintCode 392 - House Robber（打家劫舍）**

這道題目是 **1D 動態規劃（Dynamic Programming, DP）** 中「選與不選（決策類）」的最經典入門題。核心限制是**不能連續選擇相鄰的元素**，需要在這個條件下求總和的最大值。

## **解法解析**

這道題可以透過定義子問題「前 `i` 間房屋能獲得的最大金額」來建立遞推關係。

### **思路**

1. **觀察規律**
    
    - 只有 1 間房 `[2]`：只能偷這間，最大金額為 `2`。
        
    - 有 2 間房 `[2, 7]`：不能兩間都偷，挑金額大的那間，`max(2, 7) = 7`。
        
    - 有 3 間房 `[2, 7, 9]`：要嘛偷第 1 間 + 第 3 間（`2 + 9 = 11`），要嘛只偷第 2 間（`7`），取較大者 `11`。
        
    - 注意：並不是「隔一間偷一間（全偷奇數或全偷偶數）」就一定最大，例如 `[2, 1, 1, 2]` 應該偷第 1 間和第 4 間（中間跳過兩間），總和為 `4`。
        
2. **發現遞推公式**
    
    - 定義 `dp[i]` 為考慮下標 `0` 到 `i` 的房屋時，能偷竊到的最高金額：
        
        - `dp[0] = nums[0]`
            
        - `dp[1] = max(nums[0], nums[1])`
            
        - `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`（當 `i >= 2`）
            
3. **代碼執行流程**
    
    - 使用滾動變數 `prev_two`（代表 `dp[i-2]`）和 `prev_one`（代表 `dp[i-1]`），初始皆設為 `0`。
        
    - 依序遍歷每間房屋的金額 `num`，計算當前最優解 `temp = max(prev_one, prev_two + num)`。
        
    - 將變數往後平移：`prev_two = prev_one`，`prev_one = temp`。
        
    - 遍歷結束後回傳 `prev_one`。
        

## **具體舉例**

假設 `nums = [2, 7, 9, 3, 1]`，我們的變數變化如下：

|**i**|**nums[i]**|**dp[i] 計算過程**|**dp 陣列狀態**|
|---|---|---|---|
|`i = 0`|`2`|初始值 `nums[0]`|`[2, -, -, -, -]`|
|`i = 1`|`7`|`max(2, 7)`|`[2, 7, -, -, -]`|
|`i = 2`|`9`|`max(dp[1], dp[0] + 9) = max(7, 2 + 9)`|`[2, 7, 11, -, -]`|
|`i = 3`|`3`|`max(dp[2], dp[1] + 3) = max(11, 7 + 3)`|`[2, 7, 11, 11, -]`|
|`i = 4`|`1`|`max(dp[3], dp[2] + 1) = max(11, 11 + 1)`|`[2, 7, 11, 11, 12]`|

最後返回 `12`，表示最高能偷竊 **12** 元（偷第 1、3、5 間房屋：`2 + 9 + 1 = 12`）。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 只需要遍歷一次長度為 `n` 的 `nums` 陣列，每次狀態轉移為 `O(1)` 計算，因此時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**
    
    - 使用滾動變數優化後，只需要 `prev_two`、`prev_one`、`temp` 三個常數變數，空間複雜度為 **O(1)**（若使用長度為 `n` 的 `dp` 陣列則為 **O(n)**）。
        

## **其他解法**

### **1. 記憶化搜索（Top-Down DFS + Memoization）**

- **思路**：
    
    - 從最後一間房屋 `n - 1` 開始遞迴往下拆解：`dfs(i) = max(dfs(i - 1), dfs(i - 2) + nums[i])`，並用雜湊表或陣列 `memo` 記錄已經算過的結果，避免重複計算。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(n)（遞迴呼叫堆疊 + `memo` 陣列）
    

### **2. 二維狀態機 DP（明確區分偷與不偷）**

- **思路**：
    
    - 定義 `dp[i][0]` 為第 `i` 間**不偷**的最大金額，`dp[i][1]` 為第 `i` 間**偷**的最大金額：
        
        - `dp[i][0] = max(dp[i-1][0], dp[i-1][1])`
            
        - `dp[i][1] = dp[i-1][0] + nums[i]`
            
    - 這種寫法在遇到更複雜限制（例如「最多只能偷 k 次」或「有冷凍期」）時非常容易擴展。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(n)（可優化至 O(1)）
    

## **總結**

- **面試首選最佳解法**：**O(1) 空間的滾動變數動態規劃**，時間 `O(n)`，空間 `O(1)`，代碼極簡且不易出錯。
    
- **常見易錯點**：
    
    1. 誤以為只要比較「所有奇數位置總和」與「所有偶數位置總和」即可（反例：`[2, 1, 1, 2]`）。
        
    2. 陣列長度為 `0` 或 `1` 時，若使用 `dp` 陣列寫法沒有做邊界判斷，存取 `dp[1]` 會發生 `IndexError`（滾動變數寫法則天然相容空陣列與單元素陣列）。
        

## **LeetCode / LintCode 打家劫舍與線性 DP 類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 198**|House Robber（打家劫舍）|**1D 線性陣列，不能偷相鄰房屋，求最大金額**|**動態規劃 `DP[i] = max(DP[i-1], DP[i-2] + nums[i])`**|
|**LeetCode 213**|House Robber II（打家劫舍 II）|**1D 環形陣列（頭尾相連），不能偷相鄰房屋**|**拆分成兩個線性問題：`max(rob(nums[:-1]), rob(nums[1:]))`**|
|**LeetCode 337**|House Robber III（打家劫舍 III）|**二元樹結構，父子節點不能同時偷**|**樹形 DP（Post-order DFS 回傳 `[不偷當前, 偷當前]`）**|
|**LeetCode 740**|Delete and Earn（刪除並獲得點數）|**選取數字 `x` 後必須刪除所有 `x-1` 與 `x+1`**|**統計每個數字總分後，轉化為標準 House Robber 問題**|
|**LeetCode 2560**|House Robber IV（打家劫舍 IV）|**至少偷 `k` 間不相鄰房屋，求單間最大金額的最小值**|**二分搜尋答案（Binary Search）+ 貪心驗證**|
|**LeetCode 746**|Min Cost Climbing Stairs（使用最小花費爬樓梯）|**1D 陣列，每次走 1 或 2 步，求到達頂部最小花費**|**動態規劃 `DP[i] = min(DP[i-1] + cost[i-1], DP[i-2] + cost[i-2])`**|
|**LeetCode 309**|Best Time to Buy and Sell Stock with Cooldown|**買賣股票含冷凍期（賣出後隔天不能買）**|**狀態機動態規劃（持有、不持有且冷凍、不持有可買入）**|
|**LeetCode 91**|Decode Ways（解碼方法）|**1D 字串解碼，每次解 1 或 2 個字元，求方案數**|**動態規劃 `DP[i] = DP[i-1] + DP[i-2]`（需判斷字元合法性）**|

### **延伸變形題：打家劫舍 II（LeetCode 213 - 環形房屋）**

如果所有房屋圍成一圈（第一間和最後一間相鄰，不能同時偷），只需要將環形問題拆解為兩種線性情況取最大值：

1. **不偷最後一間**：計算範圍 `nums[0 : n-1]`
    
2. **不偷第一間**：計算範圍 `nums[1 : n]`
    

Python

```
class Solution:
    def rob(self, nums: list[int]) -> int:
        if len(nums) == 1:
            return nums[0]

        def rob_linear(houses: list[int]) -> int:
            prev_two = 0
            prev_one = 0
            for money in houses:
                prev_two, prev_one = prev_one, max(prev_one, prev_two + money)
            return prev_one

        # 比較「去掉最後一間」與「去掉第一間」兩種情況的最大值
        return max(rob_linear(nums[:-1]), rob_linear(nums[1:]
```