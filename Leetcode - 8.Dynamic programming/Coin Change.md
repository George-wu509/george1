
零錢兌換 Coin Change

LeetCode 322 / LintCode 669 给你一个整数数组 `coins` ，表示不同面额的硬币；以及一个整数 `amount` ，表示总金额。 计算并返回可以凑成总金额所需的 **最少的硬币个数** 。如果没有任何一种硬币组合能组成总金额，返回 `-1` 。 你可以认为每种硬币的数量是无限的。

**样例 1：** 输入：

```Python
coins = [1, 2, 5], amount = 11
```

输出：
```Python
3
```

解释： 11 = 5 + 5 + 1 共 3 枚硬币

**样例 2：** 输入：
```Python
coins = [2], amount = 3
```

输出：
```Python
-1
```

解释： 无法用面额为 2 的硬币凑出总金额 3

**样例 3：** 输入：
```Python
coins = [1], amount = 0
```

输出：
```Python
0
```

解释： 总金额为 0 时不需要任何硬币
```Python
# 解法一：記憶化搜索 (Top-Down DP)
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        memo = {}

        def dfs(rem: int) -> int:
            if rem == 0:
                return 0
            if rem < 0:
                return -1
            if rem in memo:
                return memo[rem]

            min_cost = float('inf')
            for coin in coins:
                res = dfs(rem - coin)
                if res != -1:
                    min_cost = min(min_cost, res + 1)

            memo[rem] = min_cost if min_cost != float('inf') else -1
            return memo[rem]

        return dfs(amount)
```


```Python
# 解法二：動態規劃一維陣列 (Bottom-Up DP - 最優解)
class Solution:
    def coinChange(self, coins: list[int], amount: int) -> int:
        # 初始化 dp 陣列，預設值設為 amount + 1（代表無窮大，因為最多只會用到 amount 枚 1 元硬幣）
        dp = [amount + 1] * (amount + 1)
        dp[0] = 0

        for i in range(1, amount + 1):
            for coin in coins:
                if i - coin >= 0:
                    dp[i] = min(dp[i], dp[i - coin] + 1)

        return dp[amount] if dp[amount] != amount + 1 else -1
```

解釋: step1: 狀態定義 `dp[i]` 為湊出金額 `i` 所需的最少硬幣數，陣列長度為 `amount + 1`，初始值設為 `amount + 1`（相當於正無窮）。 step2: Base Case 為 `dp[0] = 0`（湊出金額 0 需要 0 枚硬幣）。 step3: 假設 `coins = [1, 2, 5]`, `amount = 6`，初始 `dp = [0, 7, 7, 7, 7, 7, 7]`。 step4: `i = 1` 時，只能選 `coin = 1` -> `dp[1] = min(7, dp[0] + 1) = 1` -> `dp = [0, 1, 7, 7, 7, 7, 7]`。 step5: `i = 2` 時，選 `coin = 1` 得 `dp[1]+1=2`，選 `coin = 2` 得 `dp[0]+1=1` -> `dp[2] = 1`。 step6: `i = 3` 時，選 `coin = 1` 得 `dp[2]+1=2`，選 `coin = 2` 得 `dp[1]+1=2` -> `dp[3] = 2`。 step7: `i = 4` 時，選 `coin = 1` 得 `dp[3]+1=3`，選 `coin = 2` 得 `dp[2]+1=2` -> `dp[4] = 2`。 step8: `i = 5` 時，選 `coin = 5` 得 `dp[0]+1=1` -> `dp[5] = 1`。 step9: `i = 6` 時，選 `coin = 1` 得 `dp[5]+1=2`，選 `coin = 2` 得 `dp[4]+1=3`，選 `coin = 5` 得 `dp[1]+1=2` -> `dp[6] = 2`。 step10: 最終回傳 `dp[6] = 2`（即 `5 + 1`）。

解釋: 考慮最後一枚選擇了哪種面額的硬幣 `coin`。 最少硬幣數 `dp[i] = min(dp[i - coin] + 1)`，其中 `coin` 遍歷所有滿足 `i - coin ≥ 0` 的硬幣面額。 本題本質上是 **完全背包問題（Unbounded Knapsack）求最小值**，每次轉移就像爬樓梯一樣，只不過每次可以爬的步數由固定的 `1, 2` 變成了 `coins` 陣列中的任意面額。

### **LeetCode 322 / LintCode 669 - Coin Change（零錢兌換）**

這道題目是經典的 **動態規劃（Dynamic Programming, DP）** 與 **完全背包（Unbounded Knapsack）** 問題。與「爬樓梯」不同的是，爬樓梯求的是「方案總數」，而零錢兌換求的是到達目標金額的「最少步數（硬幣數）」，且每次可走的步長由 `coins` 陣列決定。

## **解法解析**

這道題可以轉化為 **自底向上（Bottom-Up）的遞推關係式** 來求解。

### **思路**

1. **觀察規律**
    
    - 假設 `coins = [1, 2, 5]`：
        
    - 若 `amount = 0`，不需要任何硬幣：`dp[0] = 0`
        
    - 若 `amount = 1`，只能拿 1 元：`dp[1] = 1`
        
    - 若 `amount = 2`，可以直接拿 2 元：`dp[2] = 1`
        
    - 若 `amount = 3`，可以從 `amount = 2` 加 1 元，或從 `amount = 1` 加 2 元：
        
        - `min(dp[2] + 1, dp[1] + 1) = 2` → 共 **2 枚硬幣**
            
    - 若 `amount = 6`，最後一步可以是加 1 元（從 5 轉移）、加 2 元（從 4 轉移）或加 5 元（從 1 轉移）：
        
        - `min(dp[5] + 1, dp[4] + 1, dp[1] + 1) = min(1+1, 2+1, 1+1) = 2` → 共 **2 枚硬幣**
            
2. **發現遞推公式**
    
    - `dp[i] = min(dp[i], dp[i - coin] + 1)`（對於所有 `coin` 滿足 `i - coin ≥ 0`）
        
    - 初始條件：
        
        - `dp[0] = 0`
            
        - 其餘 `dp[1 ... amount] = amount + 1`（作為無法湊出的哨兵最大值）
            
3. **代碼執行流程**
    
    - 建立長度為 `amount + 1` 的一維陣列 `dp`，全部填入 `amount + 1`，並設定 `dp[0] = 0`。
        
    - 外層迴圈從 `i = 1` 遍歷到 `amount`，依序計算每個金額的最少硬幣數。
        
    - 內層迴圈遍歷 `coins` 中的每一種硬幣面額，若 `i - coin ≥ 0`，則更新 `dp[i] = min(dp[i], dp[i - coin] + 1)`。
        
    - 迴圈結束後，檢查 `dp[amount]` 是否仍等於 `amount + 1`：若是則代表無法湊出，回傳 `-1`；否則回傳 `dp[amount]`。
        

## **具體舉例**

假設 `coins = [1, 2, 5]`，`amount = 11`，`dp` 陣列的逐步變化如下：

|**i（當前金額）**|**候選轉移來源 dp[i - coin] + 1**|**dp[i]（最少硬幣數）**|**對應組合**|
|---|---|---|---|
|`0`|Base Case|`0`|`[]`|
|`1`|`dp[0]+1 = 1`|`1`|`[1]`|
|`2`|`dp[1]+1 = 2`, `dp[0]+1 = 1`|`1`|`[2]`|
|`3`|`dp[2]+1 = 2`, `dp[1]+1 = 2`|`2`|`[2, 1]`|
|`4`|`dp[3]+1 = 3`, `dp[2]+1 = 2`|`2`|`[2, 2]`|
|`5`|`dp[4]+1 = 3`, `dp[3]+1 = 3`, `dp[0]+1 = 1`|`1`|`[5]`|
|`6`|`dp[5]+1 = 2`, `dp[4]+1 = 3`, `dp[1]+1 = 2`|`2`|`[5, 1]`|
|`7`|`dp[6]+1 = 3`, `dp[5]+1 = 2`, `dp[2]+1 = 2`|`2`|`[5, 2]`|
|`8`|`dp[7]+1 = 3`, `dp[6]+1 = 3`, `dp[3]+1 = 3`|`3`|`[5, 2, 1]`|
|`9`|`dp[8]+1 = 4`, `dp[7]+1 = 3`, `dp[4]+1 = 3`|`3`|`[5, 2, 2]`|
|`10`|`dp[9]+1 = 4`, `dp[8]+1 = 4`, `dp[5]+1 = 2`|`2`|`[5, 5]`|
|`11`|`dp[10]+1 = 3`, `dp[9]+1 = 4`, `dp[6]+1 = 3`|`3`|`[5, 5, 1]`|

最後返回 `dp[11] = 3`，表示湊出 `11` 元最少需要 **3 枚硬幣**。

## **時間與空間複雜度分析**

- **時間複雜度：O(amount × n)**（其中 `n` 為硬幣種類數 `len(coins)`）
    
    - 外層迴圈執行 `amount` 次，內層迴圈遍歷 `n` 種硬幣，每次狀態轉移為 `O(1)`，因此總時間複雜度為 **O(amount × n)**。
        
- **空間複雜度：O(amount)**
    
    - 需要一個長度為 `amount + 1` 的一維陣列 `dp` 來儲存每個金額的最少硬幣數，因此空間複雜度為 **O(amount)**。
        

## **其他解法**

### **1. 記憶化搜索（Top-Down DP / DFS + Memo）**

- **思路**：
    
    - 從目標 `amount` 往下遞迴扣除硬幣面額，用雜湊表 `memo` 記錄已經算過的剩餘金額結果，避免重複計算子問題。
        
- **時間複雜度**：O(amount × n)
    
- **空間複雜度**：O(amount)（遞迴呼叫棧與 `memo` 雜湊表）
    

### **2. 廣度優先搜索（BFS 最短路徑）**

- **思路**：
    
    - 將 `amount` 視為起點，每次減去一枚硬幣視為走到下一個節點，目標是走到 `0`。因為每步權重都是 1 枚硬幣，**第一次到達 `0` 時的層數**即為最少硬幣數。配合 `visited` 集合去重。
        
- **時間複雜度**：O(amount × n)
    
- **空間複雜度**：O(amount)
    

### **3. 貪心 + 剪枝 DFS（DFS + Pruning）**

- **思路**：
    
    - 先將 `coins` 由大到小排序，優先嘗試用最大面額盡可能多拿，一旦當前硬幣數已經大於等於已知的 `ans`，立刻剪枝返回。在隨機數據下極快，但最壞情況仍為指數級。
        
- **時間複雜度**：最壞 O(amount^n)，實際剪枝後通常極快
    
- **空間複雜度**：O(amount / min_coin)
    

## **總結**

- **最直觀的解法**：**記憶化搜索（Top-Down DP）**，直接按照題意往下拆解子問題，時間 `O(amount × n)`，空間 `O(amount)`。
    
- **最佳解法（面試首選）**：**Bottom-Up 一維陣列動態規劃**，無遞迴棧開銷，代碼簡潔且不易超時，時間 `O(amount × n)`，空間 `O(amount)`。
    
- **常見易錯點**：
    
    - 不可直接用純貪心（每次拿最大硬幣）求解！例如 `coins = [1, 3, 4], amount = 6`，貪心會選 `4 + 1 + 1`（3 枚），但最佳解是 `3 + 3`（2 枚）。
        
    - 陣列初始化切勿設為 `float('inf')` 後直接 `+ 1` 卻忘記轉型，使用 `amount + 1` 作為上限值最安全且乾淨。
        

## **LeetCode / LintCode 背包與零錢兌換類問題整理**

|**题号**|**题目名称**|**问题类型**|**解法**|
|---|---|---|---|
|**LeetCode 322 / LintCode 669**|Coin Change（零钱兑换）|**完全背包求最小值**，硬币无限，求凑出 `amount` 的最少硬币数|**1D 动态规划 `dp[i] = min(dp[i], dp[i - coin] + 1)`**|
|**LeetCode 518 / LintCode 740**|Coin Change II（零钱兑换 II）|**完全背包求组合数**（不计顺序），硬币无限，求凑出 `amount` 的方案数|**1D 动态规划 `dp[i] += dp[i - coin]`（外层遍历 `coins`，内层正序遍历 `amount`）**|
|**LeetCode 377 / LintCode 564**|Combination Sum IV（组合总和 Ⅳ）|**完全背包求排列数**（计顺序，类似爬楼梯），求凑出 `target` 的排列数|**1D 动态规划 `dp[i] += dp[i - num]`（外层遍历 `target`，内层遍历 `nums`）**|
|**LeetCode 279 / LintCode 513**|Perfect Squares（完全平方数）|**完全背包求最小值**，物品为 `1, 4, 9, 16...`，求组成 `n` 的最少平方数个数|**1D 动态规划 `dp[i] = min(dp[i], dp[i - j*j] + 1)`** 或 BFS|
|**LeetCode 416 / LintCode 588**|Partition Equal Subset Sum（分割等和子集）|**0-1 背包存在性问题**，每个数字只能用一次，判断是否能凑出 `sum / 2`|**1D 动态规划 `dp[j] = dp[j] or dp[j - num]`（内层需倒序遍历）**|
|**LeetCode 494 / LintCode 1208**|Target Sum（目标和）|**0-1 背包求方案数**，给每个数加 `+` 或 `-` 号，求结果等于 `target` 的方法数|**转化为子集和問題 `dp[j] += dp[j - num]`（内层倒序遍历）**|
|**LeetCode 983**|Minimum Cost For Tickets（最低票价）|**1D 路径/时间轴最小值 DP**，选择 1 天、7 天或 30 天通行证覆盖所有旅行日|**1D 动态规划 `dp[i] = min(dp[i-1]+c1, dp[i-7]+c7, dp[i-30]+c30)`**|
|**LeetCode 139 / LintCode 107**|Word Break（单词拆分）|**1D 字符串完全背包存在性**，判断字符串 `s` 能否由字典中的单词拼接而成|**1D 动态规划 `dp[i] = dp[j] and (s[j:i] in wordSet)`**|

### **延伸變形題：零錢兌換 II（LeetCode 518）- 求「組合總數」而非「最少硬幣數」**

给你一个整数数组 `coins` 表示不同面额的硬币，另给一个整数 `amount` 表示总金额。请你计算并返回可以凑成总金额的 **硬币组合数**（不考慮順序，即 `1+2` 和 `2+1` 算同一種）。

Python

```
class Solution:
    def change(self, amount: int, coins: list[int]) -> int:
        # dp[i] 表示凑出总金额 i 的组合方案数
        dp = [0] * (amount + 1)
        dp[0] = 1  # 基本情况：凑出金额 0 有 1 种方式（什么硬币都不选）

        # 关键点：外层遍历硬币，内层遍历金额，确保计算的是「组合数」而非「排列数」
        for coin in coins:
            for i in range(coin, amount + 1):
                dp[i] += dp[i - coin]

        return dp[amount]
```