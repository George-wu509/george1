
分割等和子集 Partition Equal Subset Sum

LeetCode 416 / LintCode 588 给你一个 **只包含正整数** 的 **非空** 数组 `nums` 。请你判断是否可以将这个数组分割成两个子集，使得两个子集的元素和相等。

**样例 1：** 输入：
```Python
nums = [1, 5, 11, 5]
```

输出：
```Python
True
```

解释： 数组可以分割成 [1, 5, 5] 和 [11] 两个子集的元素和均为 11

**样例 2：** 输入：
```Python
nums = [1, 2, 3, 5]
```

输出：
```Python
False
```

解释： 数组总和为 11（奇数），无法分割成两个元素和相等的子集

**样例 3：** 输入：
```Python
nums = [2, 2, 3, 5]
```

输出：
```Python
False
```

解释： 数组总和为 12，目标和为 6，但无法从 [2, 2, 3, 5] 中选出和为 6 的子集
```Python
# 解法一：記憶化搜索 / 集合遞推 (Set DP)
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        if total % 2 != 0:
            return False

        target = total // 2
        # 記錄目前為止所有能湊出的子集和
        possible_sums = {0}

        for num in nums:
            next_sums = set()
            for s in possible_sums:
                if s + num == target:
                    return True
                if s + num < target:
                    next_sums.add(s + num)
            possible_sums.update(next_sums)

        return target in possible_sums
```


```Python
# 解法二：動態規劃一維陣列 (0-1 背包 Bottom-Up DP - 最優解)
class Solution:
    def canPartition(self, nums: list[int]) -> bool:
        total = sum(nums)
        # 若總和為奇數，不可能平分成兩個整數和
        if total % 2 != 0:
            return False

        target = total // 2
        # dp[j] 表示是否能從陣列中選出若干個數字，使其總和恰好為 j
        dp = [False] * (target + 1)
        dp[0] = True

        for num in nums:
            # 關鍵：內層迴圈必須「倒序」遍歷，確保每個數字只被使用一次（0-1 背包）
            for j in range(target, num - 1, -1):
                dp[j] = dp[j] or dp[j - num]
            # 提早剪枝：一旦湊出 target 即可直接回傳 True
            if dp[target]:
                return True

        return dp[target]
```

解釋: step1: 先計算整個陣列的總和 `total`。若 `total` 為奇數，直接回傳 `False`；否則目標子集和為 `target = total // 2`。 step2: 狀態定義 `dp[j]` 為「能否從已遍歷的數字中選出若干個，使它們的和恰好為 `j`」，陣列長度為 `target + 1`，初始 `dp[0] = True`（不選任何數字即可湊出 0），其餘為 `False`。 step3: 以样例 1 `nums = [1, 5, 11, 5]` 為例，`total = 22`，`target = 11`。初始 `dp` 只有 `dp[0] = True`。 step4: 遍歷第一個數字 `num = 1`，倒序檢查 `j` 從 `11` 到 `1`：只有 `j = 1` 時 `dp[1 - 1] = dp[0]` 為 `True` -> `dp[1] = True`（目前可湊出 `{0, 1}`）。 step5: 遍歷第二個數字 `num = 5`，倒序檢查 `j` 從 `11` 到 `5`： - `j = 6` 時 `dp[6 - 5] = dp[1]` 為 `True` -> `dp[6] = True` - `j = 5` 時 `dp[5 - 5] = dp[0]` 為 `True` -> `dp[5] = True`（目前可湊出 `{0, 1, 5, 6}`）。 step6: 遍歷第三個數字 `num = 11`，倒序檢查 `j = 11` 時，`dp[11 - 11] = dp[0]` 為 `True` -> `dp[11] = True`。 step7: 觸發提早剪枝 `if dp[target]: return True`，最終回傳 `True`。

解釋: 問題等價於：能否從陣列 `nums` 中挑出一部分數字，使它們的總和恰好等於 `sum(nums) // 2`（剩下的數字總和自然也等於 `sum(nums) // 2`）。 考慮對於當前數字 `num`，我們選擇「不放入子集」或「放入子集」： 狀態轉移方程 `dp[j] = dp[j] or dp[j - num]`，其中 `j` 從 `target` 倒序遍歷至 `num`。 本題本質上是 **0-1 背包問題（0-1 Knapsack）求存在性**，每個數字最多只能選一次，因此內層容量迴圈必須 **從大到小倒序遍歷**，防止同一個數字在同一輪被重複累加。

### **LeetCode 416 / LintCode 588 - Partition Equal Subset Sum（分割等和子集）**

這道題目是經典的 **0-1 背包（0-1 Knapsack）** 動態規劃問題。題目表面上要求將陣列分成兩半，但透過數學等價轉換，只需判斷能否從陣列中選出若干元素湊出「總和的一半 `target`」，即可把雙子集分割問題簡化為單一目標的子集和問題。

## **解法解析**

這道題可以轉化為 **一維滾動陣列的 0-1 背包動態規劃** 來求解。

### **思路**

1. **觀察規律**
    
    - 設整個陣列元素總和為 `total`。若要分成兩個和相等的子集，每個子集的和必須是 `target = total / 2`。
        
    - 若 `total` 為奇數（`total % 2 != 0`），直接無解，回傳 `False`。
        
    - 若陣列中最大的單一元素 `max(nums) > target`，也必定無解，可以直接剪枝回傳 `False`。
        
    - 只要能挑出一個子集總和等於 `target`，剩下的元素總和必定也是 `target`。
        
2. **發現遞推公式**
    
    - 二維定義：`dp[i][j]` 表示從前 `i` 個數字中選取，能否湊出總和 `j`：
        
        - 不選第 `i` 個數：`dp[i][j] = dp[i-1][j]`
            
        - 選第 `i` 個數（前提 `j >= nums[i]`）：`dp[i][j] = dp[i-1][j - nums[i]]`
            
    - 壓縮成一維陣列後：`dp[j] = dp[j] or dp[j - num]`
        
    - 初始條件：
        
        - `dp[0] = True`
            
        - 其餘 `dp[1 ... target] = False`
            
3. **代碼執行流程**
    
    - 先計算 `total = sum(nums)`，若為奇數則直接返回 `False`。
        
    - 計算 `target = total // 2`，建立長度為 `target + 1` 的布林陣列 `dp`，設 `dp[0] = True`。
        
    - 外層迴圈依序取出每個數字 `num`。
        
    - 內層迴圈讓容量 `j` 從 `target` **倒序** 遞減到 `num`，更新 `dp[j] = dp[j] or dp[j - num]`。
        
    - 每輪更新後若 `dp[target]` 已經變為 `True`，即可提前結束並返回 `True`。
        

## **具體舉例**

假設 `nums = [1, 5, 11, 5]`，`total = 22`，`target = 11`，每輪加入新數字後 `dp[j] == True` 的位置變化如下：

|**當前數字 num**|**內層倒序遍歷範圍 j**|**本輪新增可湊出的和**|**所有 dp[j] == True 的集合**|
|---|---|---|---|
|初始化|-|`0`|`{0}`|
|`1`|`11 -> 1`|`0 + 1 = 1`|`{0, 1}`|
|`5`|`11 -> 5`|`1 + 5 = 6`, `0 + 5 = 5`|`{0, 1, 5, 6}`|
|`11`|`11 -> 11`|`0 + 11 = 11`|`{0, 1, 5, 6, 11}`（已命中 `target = 11`！）|

此時 `dp[11]` 已變為 `True`，無需再遍歷最後一個 `5`，直接提前返回 `True`。

## **時間與空間複雜度分析**

- **時間複雜度：O(n × target)**（其中 `n` 為陣列長度，`target = sum(nums) // 2`）
    
    - 外層迴圈遍歷 `n` 個數字，內層迴圈最多執行 `target` 次布林運算，因此總時間複雜度為 **O(n × target)**。
        
- **空間複雜度：O(target)**
    
    - 透過一維滾動陣列優化，只需維護長度為 `target + 1` 的布林陣列 `dp`，因此空間複雜度為 **O(target)**。
        

## **其他解法**

### **1. 位運算優化（Bitset / Python 大整數位移 - 極致加速）**

- **思路**：
    
    - 在 Python 中，整數支援任意精度位運算。我們可以用一個二進位整數 `bits = 1` 來代替整個 `dp` 陣列：第 `j` 個 bit 為 `1` 就代表總和 `j` 可以被湊出。
        
    - 每讀入一個數字 `num`，將 `bits` 左移 `num` 位再與原本的 `bits` 做按位或運算：`bits |= bits << num`，一行代碼就完成了內層整個迴圈的狀態轉移！最後檢查 `(bits >> target) & 1` 是否為 `1`。
        
- **時間複雜度**：O(n × target / W)（`W` 為機器字長 32 或 64，實測速度快數十倍）
    
- **空間複雜度**：O(target / W)
    

### **2. 記憶化搜索（Top-Down DP / DFS + Memo）**

- **思路**：
    
    - 定義 `dfs(index, remaining)` 表示從第 `index` 個元素開始選，能否湊出剩餘目標 `remaining`。
        
    - 每次遞迴有兩個分支：`dfs(index + 1, remaining - nums[index])`（選當前數）或 `dfs(index + 1, remaining)`（不選當前數），搭配 `@cache` 或二維陣列記錄算過的狀態。
        
- **時間複雜度**：O(n × target)
    
- **空間複雜度**：O(n × target)（遞迴呼叫棧與記憶化狀態表）
    

## **總結**

- **最直觀的解法**：**記憶化搜索（DFS + Memo）** 或 **集合遞推（Set DP）**，直接列舉「選或不選」的所有可能總和，時間 `O(n × target)`。
    
- **最佳解法（面試首選）**：**一維陣列 0-1 背包 DP（內層倒序遍歷）**，空間僅需 `O(target)`，配合 `if dp[target]: return True` 提前剪枝效率極高。
    
- **常見易錯點**：
    
    - **內層迴圈方向寫反**：如果把內層寫成正序 `for j in range(num, target + 1)`，會變成「完全背包」，導致同一個數字被重複選取多次（例如 `nums = [1, 5]`, `target = 6` 會誤把 `1` 用兩次湊出 `2`）。**0-1 背包一維優化務必倒序遍歷！**
        
    - **忘記先判斷奇偶**：若 `total % 2 == 1`，整除 `total // 2` 會向下取整（例如 `11 // 2 = 5`），若沒先擋掉奇數會算出錯誤答案。
        

## **LeetCode / LintCode 0-1 背包與子集和類問題整理**

|**题号**|**题目名称**|**问题类型**|**解法**|
|---|---|---|---|
|**LeetCode 416 / LintCode 588**|Partition Equal Subset Sum（分割等和子集）|**0-1 背包存在性问题**，判断能否选出子集和等于 `sum / 2`|**1D 动态规划 `dp[j] = dp[j] or dp[j - num]`（内层倒序）**|
|**LeetCode 494 / LintCode 1208**|Target Sum（目标和）|**0-1 背包求方案数**，给每个元素添 `+` 或 `-` 号，求凑出 `target` 的方法数|**转化为子集和 `P = (sum + target) / 2`，`dp[j] += dp[j - num]`（内层倒序）**|
|**LeetCode 1049**|Last Stone Weight II（最后一块石头的重量 II）|**0-1 背包求最接近 `sum / 2` 的最大值**，两两粉碎求最小剩余重量|**1D 动态规划 `dp[j] = max(dp[j], dp[j - w] + w)`（内层倒序）**|
|**LeetCode 474 / LintCode 668**|Ones and Zeroes（一和零）|**二维费用 0-1 背包求最大物品数**，同时受限于 `m` 个 `0` 和 `n` 个 `1`|**2D 动态规划 `dp[i][j] = max(dp[i][j], dp[i - zeros][j - ones] + 1)`（双层倒序）**|
|**LeetCode 698 / LintCode 90**|Partition to K Equal Sum Subsets（划分为k个相等的子集）|**分割为 `k` 个等和子集**（`k` 可大于 2）|**回溯 + 剪枝** 或 **状态压缩 DP（Bitmask DP）**|
|**LeetCode 322 / LintCode 669**|Coin Change（零钱兑换）|**完全背包求最小值**，硬币无限次使用，求最少硬币数|**1D 动态规划 `dp[i] = min(dp[i], dp[i - coin] + 1)`（内层正序）**|
|**LeetCode 518 / LintCode 740**|Coin Change II（零钱兑换 II）|**完全背包求组合数**，硬币无限次使用，求凑出 `amount` 的方案数|**1D 动态规划 `dp[i] += dp[i - coin]`（外层遍历物品，内层正序）**|

### **延伸變形題：目標和（LeetCode 494）- 從「判斷能否湊出」變成「求湊出的方案總數」**

给你一个非负整数数组 `nums` 和一个整数 `target` 。向数组中的每个整数前添加 `'+'` 或 `'-'` ，然后串联起所有整数，可以构造一个 **表达式** 。返回可以通过上述方法构造的、运算结果等于 `target` 的不同 **表达式** 的数目。

Python

```
class Solution:
    def findTargetSumWays(self, nums: list[int], target: int) -> int:
        total = sum(nums)
        # 设添加 '+' 的子集和为 P，添加 '-' 的子集和为 N
        # 有 P - N = target 且 P + N = total  =>  P = (total + target) // 2
        if abs(target) > total or (total + target) % 2 != 0:
            return 0

        bag_size = (total + target) // 2
        # dp[j] 表示从 nums 中选出若干元素凑出和为 j 的方案总数
        dp = [0] * (bag_size + 1)
        dp[0] = 1  # 凑出和为 0 有 1 种方案（什么都不选）

        for num in nums:
            # 同样是 0-1 背包，内层必须倒序遍历；求方案数时将 or 改为 +=
            for j in range(bag_size, num - 1, -1):
                dp[j] += dp[j - num]

        return dp[bag_size]
```