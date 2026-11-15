目標和 Target Sum

LeetCode 494 / LintCode 1208 给你一个非负整数数组 `nums` 和一个整数 `target` 。 向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** ：

- 例如，`nums = [2, 1]` ，可以在 `2` 之前添加 `'+'` ，在 `1` 之前添加 `'-'` ，然后串联起来得到表达式 `"+2-1"` 。
    

返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

**样例 1：** 输入：
```Python
nums = [1, 1, 1, 1, 1], target = 3
```

输出：
```Python
5
```

解释： 一共有 5 种方法让最终目标和为 3 ： -1 + 1 + 1 + 1 + 1 = 3 +1 - 1 + 1 + 1 + 1 = 3 +1 + 1 - 1 + 1 + 1 = 3 +1 + 1 + 1 - 1 + 1 = 3 +1 + 1 + 1 + 1 - 1 = 3

**样例 2：** 输入：
```Python
nums = [1], target = 1
```

输出：
```Python
1
```

解释： 只有一种方案：+1 = 1

**样例 3：** 输入：
```Python
nums = [0, 0, 1], target = 1
```

输出：
```Python
4
```

解释： 两个 0 前面可以任意放置 '+' 或 '-'（各有 2 种选择），加上 +1 共 2 * 2 * 1 = 4 种方案

```Python
# 解法一：記憶化搜索 (Top-Down DP / DFS + Memoization)
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        memo = {}

        def dfs(index: int, curr_sum: int) -> int:
            if index == len(nums):
                return 1 if curr_sum == target else 0
            if (index, curr_sum) in memo:
                return memo[(index, curr_sum)]

            add = dfs(index + 1, curr_sum + nums[index])
            sub = dfs(index + 1, curr_sum - nums[index])

            memo[(index, curr_sum)] = add + sub
            return memo[(index, curr_sum)]

        return dfs(0, 0)
```


```Python
# 解法二：數學轉化 + 動態規劃一維陣列 (0-1 背包 Bottom-Up DP - 最優解)
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        # 設添加 '+' 的數字總和為 P，添加 '-' 的數字總和為 N
        # P - N = target 且 P + N = total  =>  P = (total + target) // 2
        if abs(target) > total or (total + target) % 2 != 0:
            return 0

        bag_size = (total + target) // 2
        # dp[j] 表示從 nums 中選出若干數字，使其總和恰好為 j 的方案數
        dp = [0] * (bag_size + 1)
        dp[0] = 1

        for num in nums:
            # 關鍵：內層迴圈必須「倒序」遍歷，確保每個數字只被使用一次（0-1 背包）
            for j in range(bag_size, num - 1, -1):
                dp[j] += dp[j - num]

        return dp[bag_size]
```

解釋: step1: 設所有前面加 `+` 號的元素總和為 `P`，所有前面加 `-` 號的元素總和為 `N`。由題意可得 `P - N = target`，且 `P + N = total`（陣列總和）。 step2: 兩式相加消去 `N`，得到 `2P = total + target`，即目標正數子集和 `bag_size = (total + target) // 2`。若 `abs(target) > total` 或 `total + target` 為奇數，則無整數解，直接回傳 `0`。 step3: 狀態定義 `dp[j]` 為「從已遍歷的數字中選出若干個，使它們的和恰好為 `j` 的方案總數」，陣列長度為 `bag_size + 1`，初始 `dp[0] = 1`（不選任何數字湊出 0 有 1 種方法）。 step4: 以样例 1 `nums = [1, 1, 1, 1, 1]`, `target = 3` 為例，`total = 5`，`bag_size = (5 + 3) // 2 = 4`。初始 `dp = [1, 0, 0, 0, 0]`。 step5: 第 1 個 `1`：倒序更新 `j = 4..1` -> `dp[1] += dp[0]` -> `dp = [1, 1, 0, 0, 0]`。 step6: 第 2 個 `1`：倒序更新 `j = 4..1` -> `dp[2] += dp[1]`, `dp[1] += dp[0]` -> `dp = [1, 2, 1, 0, 0]`。 step7: 第 3 個 `1`：倒序更新 `j = 4..1` -> `dp = [1, 3, 3, 1, 0]`。 step8: 第 4 個 `1`：倒序更新 `j = 4..1` -> `dp = [1, 4, 6, 4, 1]`。 step9: 第 5 個 `1`：倒序更新 `j = 4..1` -> `dp = [1, 5, 10, 10, 5]`。 step10: 最終回傳 `dp[4] = 5`（代表從 5 個 `1` 中選出 4 個加 `+` 號、1 個加 `-` 號，共有 `C(5, 4) = 5` 種方法）。

解釋: 透過代數變換，將「給每個數字分配 `+` 或 `-` 號」轉化為「從陣列中挑選一個子集，使其元素總和恰好為 `(total + target) / 2` 的方案數」。 對於當前數字 `num`，湊出總和 `j` 的方案數等於「不選 `num` 的方案數」加上「選 `num` 的方案數」： 狀態轉移方程 `dp[j] = dp[j] + dp[j - num]`，其中 `j` 從 `bag_size` 倒序遍歷至 `num`。 本題本質上是 **0-1 背包問題（0-1 Knapsack）求組合方案數**，因此內層迴圈同樣必須 **倒序遍歷**，避免同一個數字被重複選取。

### **LeetCode 494 / LintCode 1208 - Target Sum（目標和）**

這道題目是經典的 **0-1 背包求方案數** 問題。如果直接用暴力回溯（DFS），每個數字有 `+` 和 `-` 兩種選擇，時間複雜度為 `O(2^n)`；透過代數拆分轉化為子集和問題後，可以用一維動態規劃在多項式時間內高效求解。

## **解法解析**

這道題可以轉化為 **代數等價變換 + 一維 0-1 背包動態規劃** 來求解。

### **思路**

1. **觀察規律（數學推導）**
    
    - 假設陣列 `nums` 被分成兩組：前面放 `+` 號的數字集合 `P`，以及前面放 `-` 號的數字集合 `N`（這裡 `N` 代表這些數字本身的正數總和）。
        
    - 根據題目條件：
        
        - `sum(P) - sum(N) = target`
            
        - `sum(P) + sum(N) = total`
            
    - 兩式相加可得：`2 * sum(P) = total + target`，因此 `sum(P) = (total + target) / 2`。
        
    - 邊界剪枝條件：
        
        - 若 `abs(target) > total`（就算全加 `+` 或全加 `-` 也達不到 `target`），回傳 `0`。
            
        - 若 `(total + target) % 2 != 0`（`sum(P)` 不是整數，因為陣列全為整數，不可能湊出小數），回傳 `0`。
            
2. **發現遞推公式**
    
    - 設目標背包容量 `bag_size = (total + target) // 2`。
        
    - `dp[j]` 表示裝滿容量為 `j` 的背包共有多少種選法：
        
        - `dp[j] += dp[j - num]`
            
    - 初始條件：
        
        - `dp[0] = 1`（容量為 0 時，什麼都不選就是 1 種合法方案）
            
        - 其餘 `dp[1 ... bag_size] = 0`
            
3. **代碼執行流程**
    
    - 計算 `total = sum(nums)`，並檢查 `abs(target) > total` 或 `(total + target) % 2 != 0`，不符合則直接返回 `0`。
        
    - 計算 `bag_size = (total + target) // 2`，初始化長度為 `bag_size + 1` 的陣列 `dp`，設 `dp[0] = 1`。
        
    - 外層迴圈遍歷 `nums` 中的每個數字 `num`。
        
    - 內層迴圈讓 `j` 從 `bag_size` **倒序** 遍歷到 `num`（注意：當 `num == 0` 時，倒序到 `0` 會執行 `dp[0] += dp[0]`，自動讓方案數翻倍，完美處理 `0` 的正負兩種選法！）。
        
    - 迴圈結束後回傳 `dp[bag_size]`。
        

## **具體舉例**

假設 `nums = [1, 1, 1, 1, 1]`，`target = 3`，則 `total = 5`，`bag_size = (5 + 3) // 2 = 4`。每輪加入新數字後 `dp` 陣列的變化如下：

|**當前數字 num**|**dp[0]**|**dp[1]**|**dp[2]**|**dp[3]**|**dp[4]**|**含義說明**|
|---|---|---|---|---|---|---|
|初始化|`1`|`0`|`0`|`0`|`0`|不選任何數，湊出 `0` 有 `1` 種方法|
|第 1 個 `1`|`1`|`1`|`0`|`0`|`0`|前 1 個數中選 0 個或 1 個|
|第 2 個 `1`|`1`|`2`|`1`|`0`|`0`|前 2 個數中選 0, 1, 2 個的組合數|
|第 3 個 `1`|`1`|`3`|`3`|`1`|`0`|對應二項式係數（楊輝三角）|
|第 4 個 `1`|`1`|`4`|`6`|`4`|`1`|前 4 個數中湊出 `0~4` 的方案數|
|第 5 個 `1`|`1`|`5`|`10`|`10`|**`5`**|前 5 個數中湊出 `4` 共有 **`5`** 種方案|

最終返回 `dp[4] = 5`，表示共有 **5 種** 不同的表達式。

## **時間與空間複雜度分析**

- **時間複雜度：O(n × bag_size)**（其中 `n` 為陣列長度，`bag_size = (sum(nums) + target) // 2`）
    
    - 外層迴圈遍歷 `n` 個元素，內層迴圈最多執行 `bag_size` 次加法運算，因此總時間複雜度為 **O(n × bag_size)**。
        
- **空間複雜度：O(bag_size)**
    
    - 使用一維滾動陣列優化後，只需維護長度為 `bag_size + 1` 的整數陣列 `dp`，因此空間複雜度為 **O(bag_size)**。
        

## **其他解法**

### **1. 記憶化搜索（Top-Down DP / DFS + Memo）**

- **思路**：
    
    - 不需要做數學推導，直接從 `index = 0, curr_sum = 0` 出發，每一步遞迴嘗試 `+ nums[index]` 與 `- nums[index]` 兩條路徑，並以 `(index, curr_sum)` 作為鍵值存入雜湊表 `memo` 快取結果。
        
- **時間複雜度**：O(n × total)（狀態總數最多為 `n` 個索引乘上 `[-total, total]` 共 `2 * total + 1` 種和）
    
- **空間複雜度**：O(n × total)（雜湊表與遞迴棧開銷）
    

### **2. 二維 0-1 背包動態規劃（2D Bottom-Up DP）**

- **思路**：
    
    - 定義 `dp[i][j]` 為前 `i` 個數字湊出和為 `j` 的方案數：
        
        - 當 `j < nums[i-1]` 時：`dp[i][j] = dp[i-1][j]`
            
        - 當 `j >= nums[i-1]` 時：`dp[i][j] = dp[i-1][j] + dp[i-1][j - nums[i-1]]`
            
- **時間複雜度**：O(n × bag_size)
    
- **空間複雜度**：O(n × bag_size)
    

### **3. 折半搜索（Meet in the Middle）**

- **思路**：
    
    - 若題目修改為 `nums` 包含負數或 `total` 非常大（例如 `10^9`）導致無法開 `dp` 陣列，但陣列長度 `n <= 40` 時，可將陣列對半拆成左右兩部分（各 `n/2` 個數），分別暴力枚舉所有 `2^(n/2)` 種正負號結果存入雜湊表，再配對相加等於 `target` 的組合。
        
- **時間複雜度**：O(2^(n/2))
    
- **空間複雜度**：O(2^(n/2))
    

## **總結**

- **最直觀的解法**：**記憶化搜索（DFS + Memo）**，無需數學轉換，直接模擬加號與減號分支，適合在面試中快速寫出保底解，時間 `O(n × total)`，空間 `O(n × total)`。
    
- **最佳解法（面試最優解）**：**數學推導 + 一維陣列 0-1 背包 DP**，將雙向加減轉化為單向正數子集和，時間 `O(n × bag_size)`，空間僅需 `O(bag_size)`。
    
- **常見易錯點**：
    
    - **陣列中包含 `0` 的情況**：本題允許 `nums[i] == 0`（例如 `nums = [0, 0, 1], target = 1`）。一維倒序寫法 `for j in range(bag_size, num - 1, -1)` 當 `num = 0` 時會執行 `dp[j] += dp[j]`（即 `dp[j] *= 2`），天然正確處理了 `+0` 和 `-0` 算兩種不同表達式的要求；若自己寫「遇到 0 跳過」反而會算錯！
        
    - **忘記檢查 `abs(target) > total`**：當 `target` 為負數且絕對值大於 `total` 時（例如 `nums = [1], target = -3`），`(total + target)` 會變成負數，必須先用 `abs(target) > total` 擋掉，否則初始化陣列長度會報錯。
        

## **LeetCode / LintCode 0-1 背包與符號組合類問題整理**

|**题号**|**题目名称**|**问题类型**|**解法**|
|---|---|---|---|
|**LeetCode 494 / LintCode 1208**|Target Sum（目标和）|**0-1 背包求方案数**，给每个数添 `+` 或 `-` 号，求凑出 `target` 的方案数|**转化为子集和 `P = (sum + target) / 2`，`dp[j] += dp[j - num]`（内层倒序）**|
|**LeetCode 416 / LintCode 588**|Partition Equal Subset Sum（分割等和子集）|**0-1 背包存在性问题**，判断能否选出子集和等于 `sum / 2`|**1D 动态规划 `dp[j] = dp[j] or dp[j - num]`（内层倒序）**|
|**LeetCode 1049**|Last Stone Weight II（最后一块石头的重量 II）|**0-1 背包求最接近 `sum / 2` 的最大值**，等价于给石头分配 `+` 和 `-` 号使总和最小|**1D 动态规划 `dp[j] = max(dp[j], dp[j - w] + w)`（内层倒序）**|
|**LeetCode 474 / LintCode 668**|Ones and Zeroes（一和零）|**二维费用 0-1 背包求最大物品数**，同时受限于 `m` 个 `0` 和 `n` 个 `1`|**2D 动态规划 `dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)`（双层倒序）**|
|**LeetCode 518 / LintCode 740**|Coin Change II（零钱兑换 II）|**完全背包求组合数**，物品无限次使用，求凑出 `amount` 的方案数|**1D 动态规划 `dp[i] += dp[i - coin]`（外层遍历物品，内层正序）**|
|**LeetCode 377 / LintCode 564**|Combination Sum IV（组合总和 Ⅳ）|**完全背包求排列数**，物品无限次使用，不同顺序算不同方案|**1D 动态规划 `dp[i] += dp[i - num]`（外层遍历背包，内层遍历物品）**|
|**LeetCode 1755**|Closest Subsequence Sum（最接近目标值的子序列和）|**子集和最接近 `goal`**，数组包含负数且 `goal` 极大（`10^9`）|**折半搜索（Meet in the Middle）+ 排序雙指針 / 二分查找**|

### **延伸變形題：最後一塊石頭的重量 II（LeetCode 1049）- 同樣將元素分成正負兩組，但求「最小差值」而非「方案數」**

有一堆石头，用整数数组 `stones` 表示。其中 `stones[i]` 表示第 `i` 块石头的重量。每一回合，从中选出 **任意两块石头**，然后将它们一起粉碎。假设石头的重量分别为 `x` 和 `y`，且 `x <= y`。那么粉碎的可能结果如下：

- 如果 `x == y`，那么两块石头都会被完全粉碎；
    
- 如果 `x != y`，那么重量为 `x` 的石头将会完全粉碎，而重量为 `y` 的石头新重量为 `y - x`。
    

最后，**最多只会剩下一块** 石头。返回此石头 **最小的可能重量** 。如果没有石头剩下，就返回 `0`。

Python

```
class Solution:
    def lastStoneWeightII(self, stones: list[int]) -> int:
        # 核心思想与 Target Sum 完全一致：将石头分成两堆 P 和 N，使 |P - N| 最小
        # 即让其中一堆的重量尽可能接近 total // 2
        total = sum(stones)
        target = total // 2

        # dp[j] 表示容量为 j 的背包最多能装下的石头总重量
        dp = [0] * (target + 1)

        for stone in stones:
            for j in range(target, stone - 1, -1):
                dp[j] = max(dp[j], dp[j - stone] + stone)

        # 一堆重量为 dp[target]，另一堆重量为 total - dp[target]，差值即为最小剩余重量
        return total - 2 * dp[target]
```