
買賣股票的最佳時機（含冷凍期） Best Time to Buy and Sell Stock with Cooldown

LeetCode 309 / LintCode 995 给定一个整数数组 `prices`，其中第 `i` 个元素代表了第 `i` 天的股票价格。

设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）：

- **卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。**
    

**注意**：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。

**样例 1：** 输入：
```Python
prices = [1, 2, 3, 0, 2]
```

输出：
```Python
3
```

解释： 对应的交易状态为: [买入, 卖出, 冷冻期, 买入, 卖出] 第一天买入 (1)，第二天卖出 (2)，第三天冷冻期，第四天买入 (0)，第五天卖出 (2)。 總利潤 = (2 - 1) + (2 - 0) = 3 。

**样例 2：** 输入：
```Python
prices = [1]
```

输出：
```Python
0
```

解释： 只有一天，無法完成買入再賣出，最大利潤為 0 。

**样例 3：** 输入：
```Python
prices = [2, 1, 4]
```

输出：
```Python
3
```

解释： 第二天買入 (1)，第三天賣出 (4)，最大利潤 = 4 - 1 = 3 。

```Python
# 解法一：二維狀態機 DP 陣列
def maxProfit(self, prices: list[int]) -> int:
    if not prices:
        return 0
    n = len(prices)
    # dp[i][0]: 第 i 天結束後「持有股票」的最大利潤
    # dp[i][1]: 第 i 天結束後「不持有股票，且當天賣出（明天進入冷凍期）」的最大利潤
    # dp[i][2]: 第 i 天結束後「不持有股票，且當天未賣出（可隨時買入）」的最大利潤
    dp = [[0] * 3 for _ in range(n)]
    dp[0][0] = -prices[0]
    dp[0][1] = float('-inf')
    dp[0][2] = 0

    for i in range(1, n):
        dp[i][0] = max(dp[i - 1][0], dp[i - 1][2] - prices[i])
        dp[i][1] = dp[i - 1][0] + prices[i]
        dp[i][2] = max(dp[i - 1][2], dp[i - 1][1])

    return max(dp[-1][1], dp[-1][2])
```


```Python
# 解法二：空間優化 O(1) 狀態機滾動變數
def maxProfit(self, prices: list[int]) -> int:
    if not prices:
        return 0
    hold = -prices[0]      # 持有股票
    sold = float('-inf')   # 當天剛賣出股票（處於冷凍期）
    rest = 0               # 未持有股票且非冷凍期（自由狀態）

    for price in prices[1:]:
        prev_hold, prev_sold, prev_rest = hold, sold, rest
        hold = max(prev_hold, prev_rest - price)
        sold = prev_hold + price
        rest = max(prev_rest, prev_sold)

    return max(sold, rest)
```

解釋: step1: 每天結束時只可能處於三種狀態之一：`hold`（持有股票）、`sold`（當天剛賣出股票，導致隔天冷凍）、`rest`（不持有股票且不在冷凍期，可以買入）。 step2: 以 `prices = [1, 2, 3, 0, 2]` 為例，初始化第 0 天 (`price = 1`)：`hold = -1`, `sold = -inf`, `rest = 0`。 step3: 第 1 天 (`price = 2`)：

- `hold = max(-1, 0 - 2) = -1`（維持第 0 天買入的股票）
    
- `sold = -1 + 2 = 1`（賣出第 0 天買入的股票，獲利 1）
    
- `rest = max(0, -inf) = 0` step4: 第 2 天 (`price = 3`)：
    
- `hold = max(-1, 0 - 3) = -1`
    
- `sold = -1 + 3 = 2`（若今天才賣出，獲利 2）
    
- `rest = max(0, 1) = 1`（若昨天賣出，今天度過冷凍期，保留利潤 1） step5: 第 3 天 (`price = 0`)：
    
- `hold = max(-1, 1 - 0) = 1`（用 `rest` 狀態的利潤 1 買入價格 0 的股票，手中剩餘利潤 1）
    
- `sold = -1 + 0 = -1`
    
- `rest = max(1, 2) = 2` step6: 第 4 天 (`price = 2`)：
    
- `hold = max(1, 2 - 2) = 1`
    
- `sold = 1 + 2 = 3`（賣出第 3 天買入的股票，總利潤 `1 + 2 = 3`）
    
- `rest = max(2, -1) = 2` step7: 遍歷結束，最後一天手上有股票一定不是最優解，返回 `max(sold, rest) = max(3, 2) = 3`。
    

解釋: 這題的核心是**狀態機動態規劃（State Machine DP）**，把「冷凍期 1 天」這個限制拆解進狀態定義裡：

1. **狀態 `hold`（持有股票）**：可以是昨天就持有（`prev_hold`），或是昨天處於可買入的自由狀態且今天買入（`prev_rest - price`）。**注意：不能從 `prev_sold` 直接買入，因為剛賣出的隔天是冷凍期！**
    
2. **狀態 `sold`（當天賣出）**：只能從昨天持有股票的狀態賣出轉移而來（`prev_hold + price`）。
    
3. **狀態 `rest`（空手且可買入）**：可以是昨天本來就空手（`prev_rest`），或是昨天剛賣出、今天剛好度過冷凍期解除限制（`prev_sold`）。
    

### **LeetCode 309 / LintCode 995 - Best Time to Buy and Sell Stock with Cooldown（買賣股票的最佳時機含冷凍期）**

這道題目是 **狀態機動態規劃（State Machine DP）** 的經典代表作。與一般的股票買賣問題不同，本題多了一個「賣出後隔天為冷凍期（Cooldown），無法買入」的限制。透過將每一天的狀態細分為**三種互斥狀態**，就能把複雜的時間限制轉化為清晰的狀態轉移圖。

## **解法解析**

這道題可以透過定義三個狀態變數並同步更新來求解。

### **思路**

1. **觀察規律**
    
    - 如果沒有冷凍期（LeetCode 122），只要當天價格比前一天高就可以賺取差價；但有了冷凍期後，今天賣出會犧牲明天的買入機會，因此不能只看單日漲跌，必須記錄歷史最佳決策。
        
    - 傳統只用「持有 / 不持有」兩種狀態無法區分「今天剛賣出（明天不能買）」與「以前就賣出（明天可以買）」，因此必須把「不持有」拆成 `sold` 和 `rest` 兩個狀態。
        
2. **發現狀態轉移公式**
    
    - `hold = max(prev_hold, prev_rest - price)`
        
    - `sold = prev_hold + price`
        
    - `rest = max(prev_rest, prev_sold)`
        
3. **代碼執行流程**
    
    - 初始化第 0 天狀態：`hold = -prices[0]`（第 0 天買入），`sold = -inf`（第 0 天不可能賣出），`rest = 0`（第 0 天觀望）。
        
    - 從第 1 天開始遍歷 `prices`，每次先用 `prev_hold, prev_sold, prev_rest` 保存前一天的三個狀態，避免同一天內狀態互相覆蓋。
        
    - 遍歷結束後，比較兩種不持有股票的狀態 `max(sold, rest)` 即為最大利潤。
        

## **具體舉例**

假設 `prices = [1, 2, 3, 0, 2]`，每天結束後的狀態變化如下：

|**天數 (i)**|**股價 (price)**|**hold（持有）**|**sold（剛賣出）**|**rest（空手可買）**|**最佳操作含義說明**|
|---|---|---|---|---|---|
|`i = 0`|`1`|`-1`|`-inf`|`0`|第 0 天花 1 元買入|
|`i = 1`|`2`|`-1`|`1`|`0`|若今天賣出可賺 1 元|
|`i = 2`|`3`|`-1`|`2`|`1`|若今天賣出可賺 2 元；或昨天賣出今天冷凍（利潤 1）|
|`i = 3`|`0`|`1`|`-1`|`2`|用 `rest=1` 在股價 0 元買入，持有且淨利仍為 `1`|
|`i = 4`|`2`|`1`|`3`|`2`|將昨天 0 元買入的股票以 2 元賣出，總利潤達 `3`|

最後返回 `max(sold, rest) = max(3, 2) = 3`，表示最大利潤為 **3**。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 只需遍歷一次長度為 `n` 的 `prices` 陣列，每天更新三個狀態的操作均為 `O(1)`，因此時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**
    
    - 使用三個滾動變數 `hold`、`sold`、`rest` 代替二維 `dp[n][3]` 陣列，僅需常數額外空間，空間複雜度為 **O(1)**（若使用 `dp` 陣列則為 **O(n)**）。
        

## **其他解法**

### **1. 傳統二維 DP 陣列（`dp[i][0/1/2]`）**

- **思路**：
    
    - 明確開闢一個 `n x 3` 的二維表格記錄每一天三種狀態的最大收益，不需要使用 `prev_*` 暫存變數，讀寫分離更直觀，適合初學者推導狀態轉移方程。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(n)
    

### **2. 記憶化搜索（Top-Down DFS + Memoization）**

- **思路**：
    
    - 定義 `dfs(i, can_buy)` 表示第 `i` 天能否買入時的最大收益：若今天買入，則遞迴到 `dfs(i + 1, False) - prices[i]`；若今天賣出，因為有冷凍期，直接跳過 `i + 1` 天遞迴到 `dfs(i + 2, True) + prices[i]`。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(n)
    

## **總結**

- **面試首選最佳解法**：**O(1) 空間的三狀態機滾動變數 DP（`hold` / `sold` / `rest`）**，時間 `O(n)`，空間 `O(1)`，邏輯清晰且容易擴展到其他股票變形題。
    
- **常見易錯點**：
    
    1. **同一天更新狀態時沒有先保存前一天的值**：例如先更新了 `hold`，接著計算 `sold = hold + price` 時誤用了當天剛買入的 `hold`（相當於當天買又當天賣）。
        
    2. **錯誤地讓 `hold` 從 `prev_sold` 轉移**：`sold` 代表當天剛賣出，隔天必須經歷至少一天的 `rest` 狀態才能再次買入。
        

## **LeetCode / LintCode 買賣股票系列與狀態機 DP 問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 309**|Best Time to Buy and Sell Stock with Cooldown（含冷凍期）|**不限交易次數，賣出後冷凍 1 天**|**三狀態機 DP（`hold`、`sold`、`rest`）O(n) 時間, O(1) 空間**|
|**LeetCode 121**|Best Time to Buy and Sell Stock（買賣股票的最佳時機）|**最多交易 1 次，求最大利潤**|**貪心 / 1D DP（維護歷史最低買入價 `min_price`）**|
|**LeetCode 122**|Best Time to Buy and Sell Stock II（買賣股票的最佳時機 II）|**不限交易次數，無冷凍期與手續費**|**貪心（累加所有上漲波段）或雙狀態機 DP（`hold`、`cash`）**|
|**LeetCode 123**|Best Time to Buy and Sell Stock III（買賣股票的最佳時機 III）|**最多完成 2 筆交易，求最大利潤**|**四狀態機 DP（`buy1`、`sell1`、`buy2`、`sell2`）O(n)**|
|**LeetCode 188**|Best Time to Buy and Sell Stock IV（買賣股票的最佳時機 IV）|**最多完成 `k` 筆交易，求最大利潤**|**二維狀態機 DP：`dp[j][0/1]`（`j` 為交易次數上限）O(nk)**|
|**LeetCode 714**|Best Time to Buy and Sell Stock with Transaction Fee（含手續費）|**不限交易次數，每筆交易需支付手續費 `fee`**|**雙狀態機 DP：`cash = max(cash, hold + price - fee)`**|
|**LeetCode 198**|House Robber（打家劫舍）|**1D 線性陣列，不能連續選擇相鄰元素**|**動態規劃 `DP[i] = max(DP[i-1], DP[i-2] + nums[i])`**|

### **延伸變形題：買賣股票的最佳時機（含手續費）（LeetCode 714）**

如果不設冷凍期，但每次完成一筆交易（買入並賣出）需要支付一筆固定手續費 `fee`，只需將三狀態簡化回**雙狀態機 DP**（`hold` 持有股票、`cash` 不持有股票），並在賣出（或買入）時扣除 `fee` 即可：

Python

```
class Solution:
    def maxProfit(self, prices: list[int], fee: int) -> int:
        hold = -prices[0]  # 持有股票的最大利潤
        cash = 0           # 不持有股票的最大利潤

        for price in prices[1:]:
            prev_hold, prev_cash = hold, cash
            hold = max(prev_hold, prev_cash - price)
            cash = max(prev_cash, prev_hold + price - fee)

        return cash
```