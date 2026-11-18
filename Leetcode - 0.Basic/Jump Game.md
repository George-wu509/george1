
LeetCode 55 / LintCode 116 给定一个非负整数数组 `nums` ，你最初位于数组的 **第一个下标** 。数组中的每个元素代表你在该位置可以跳跃的最大长度。判断你是否能够到达最后一个下标，如果可以，返回 `true` ；否则，返回 `false` 。

**样例 1：** 输入：
```Python
nums = [2, 3, 1, 1, 4]
```

输出：
```Python
True
```

解释： 可以先跳 1 步，从下标 0 到达下标 1, 然后再从下标 1 跳 3 步到达最后一个下标。

**样例 2：** 输入：
```Python
nums = [3, 2, 1, 0, 4]
```

输出：
```Python
False
```

解释： 无论怎样，总会到达下标为 3 的位置。但该下标的最大跳跃长度是 0 ， 所以永远不可能到达最后一个下标。

**样例 3：** 输入：
```Python
nums = [0]
```

输出：
```Python
True
```

解释： 起点即终点，不需要跳跃即可到达。
```Python
# 解法一：貪心法（從前往後維護最遠可達距離）
def canJump(self, nums: list[int]) -> bool:
    max_reach = 0
    for i, jump in enumerate(nums):
        if i > max_reach:
            return False
        max_reach = max(max_reach, i + jump)
        if max_reach >= len(nums) - 1:
            return True
    return True
```


```Python
# 解法二：貪心法（NeetCode 經典從後往前推終點 Goal）
def canJump(self, nums: list[int]) -> bool:
    goal = len(nums) - 1
    for i in range(len(nums) - 2, -1, -1):
        if i + nums[i] >= goal:
            goal = i
    return goal == 0
```
解釋(my):
max_reach代表可以到達的最遠id, 所以只要max_reach可以大於nums的最後id代表會到達, 為True


解釋（以解法一為例）:
step1: 維護一個變數 `max_reach`，記錄目前為止**最遠能到達的索引位置**，初始為 `0` 
step2: 遍歷陣列 `nums = [2, 3, 1, 1, 4]`，終點索引為 `4` 
step3: `i = 0, jump = 2` -> `0 <= max_reach(0)`，更新 `max_reach = max(0, 0 + 2) = 2`
step4: `i = 1, jump = 3` -> `1 <= max_reach(2)`，更新 `max_reach = max(2, 1 + 3) = 4` 
step5: 此時 `max_reach(4) >= 終點(4)`，代表已經可以覆蓋到終點，直接提前返回 `True`

解釋（以解法二為例）: 考慮從終點倒推回來，只要某個位置 `i` 能跳到當前的目標 `goal`（即 `i + nums[i] >= goal`），就把 `goal` 往前移動到 `i`。最後只要檢查 `goal` 能否一路退回到起點 `0` 即可。

### **LeetCode 55 / LintCode 116 - Jump Game（跳躍遊戲）**

這道題目是 Blind 75 與 NeetCode 150 中最經典的 **貪心演算法（Greedy）** 代表題。題目不需要我們求出「具體怎麼跳」或「最少跳幾步」，只需要判斷**能否到達終點**，因此我們不需要窮舉所有路徑，只需關注「覆蓋範圍的極限」。

## **解法解析**

這道題的核心在於將「每一步的選擇」轉化為 **「可達區間的擴張」** 來求解。

### **思路**

1. **觀察規律**
    
    - 只要你能到達位置 `k`，那麼 `0` 到 `k` 之間的**每一個位置**你都能到達（因為題目說的是「最大」跳躍長度，代表你可以選擇跳 `1, 2, ..., nums[i]` 步）。
        
    - 真正會阻擋我們前進的只有 **`0`（陷阱）**，而且必須是「前面所有位置都跳不過這個 `0`」的情況。
        
    - 例如 `[3, 2, 1, 0, 4]`：
        
        - 在 `i = 0` 最遠到 `3`
            
        - 在 `i = 1` 最遠到 `1 + 2 = 3`
            
        - 在 `i = 2` 最遠到 `2 + 1 = 3`
            
        - 在 `i = 3` 最遠到 `3 + 0 = 3`
            
        - 當迴圈走到 `i = 4` 時，發現 `4 > max_reach (3)`，代表索引 `4` 永遠無法被觸及，直接失敗。
            
2. **發現核心判斷公式**
    
    - **可達條件**：在遍歷到索引 `i` 時，必須滿足 `i <= max_reach`。若 `i > max_reach`，代表中間出現斷層，無法繼續前進，直接回傳 `False`。
        
    - **更新最遠距離**：`max_reach = max(max_reach, i + nums[i])`
        
    - **提早結束條件**：一旦 `max_reach >= len(nums) - 1`，即可直接回傳 `True`。
        
3. **代碼執行流程**
    
    - 初始化 `max_reach = 0`。
        
    - 依序走訪每個索引 `i` 與對應步數 `jump`。
        
    - 先檢查當前位置 `i` 是否在可達範圍內（`i > max_reach` 則返回 `False`）。
        
    - 接著更新最遠可達索引，並判斷是否已經超過或抵達最後一個索引。
        

## **具體舉例**

假設 `nums = [3, 2, 1, 0, 4]`（終點索引為 `4`），變數變化如下：

|**i**|**nums[i]**|**檢查 i > max_reach**|**i + nums[i]**|**更新後 max_reach**|**狀態說明**|
|---|---|---|---|---|---|
|初始化|-|-|-|`0`|起點在索引 0|
|`i = 0`|`3`|`0 > 0` (False)|`0 + 3 = 3`|`3`|最遠可跳到索引 3|
|`i = 1`|`2`|`1 > 3` (False)|`1 + 2 = 3`|`3`|最遠仍停在索引 3|
|`i = 2`|`1`|`2 > 3` (False)|`2 + 1 = 3`|`3`|最遠仍停在索引 3|
|`i = 3`|`0`|`3 > 3` (False)|`3 + 0 = 3`|`3`|卡在 0，無法突破索引 3|
|`i = 4`|`4`|`4 > 3` (**True**)|-|`3`|索引 4 不可達，回傳 `False`|

最後返回 `False`，表示無法到達最後一個下標。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 最多只需遍歷陣列 `nums` 一次，每次更新與比較皆為 `O(1)` 操作，因此時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**
    
    - 只需要常數個變數（`max_reach` 或 `goal`）來記錄狀態，不需要額外陣列，因此空間複雜度為 **O(1)**。
        

## **其他解法**

### **1. 動態規劃（Dynamic Programming）**

- **思路**：
    
    - 定義 `dp[i]` 為布林值，表示「能否到達索引 `i`」，初始 `dp[0] = True`。
        
    - 對於每個 `i`，往前檢查 `j`（`0 <= j < i`），若 `dp[j] == True` 且 `j + nums[j] >= i`，則 `dp[i] = True`。
        
- **時間複雜度**：O(n²)（在 LeetCode 大測資下容易超時 TLE）
    
- **空間複雜度**：O(n)
    

### **2. 記憶化搜索（DFS + Memoization）**

- **思路**：
    
    - 從索引 `0` 開始嘗試跳 `1` 到 `nums[i]` 步，用 `memo` 陣列記錄已經確定「走不到終點」的索引，避免重複遞迴。
        
- **時間複雜度**：O(n²)
    
- **空間複雜度**：O(n)
    

## **總結**

- **最直觀但較慢的解法**：動態規劃 `DP` 或 `DFS + Memoization`，時間 `O(n²)`，空間 `O(n)`。
    
- **最佳解法（面試首選）**：**貪心演算法（Greedy）**，時間 `O(n)`，空間 `O(1)`。
    
- **解題關鍵**：不需要糾結「每一步具體要跳幾格」，只要關注「目前能覆蓋到的最遠邊界」是否能一路推到終點即可。
    

## **Jump Game / 貪心與跳躍類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 55 / LintCode 116**|Jump Game（跳躍遊戲）|**1D 陣列，判斷能否到達終點**|貪心（Greedy）維護最遠可達距離 `max_reach`，時間 `O(n)`|
|**LeetCode 45 / LintCode 117**|Jump Game II（跳躍遊戲 II）|**1D 陣列，保證能到終點，求最少跳躍次數**|貪心（隱式 BFS 分層）維護當前邊界 `end` 與下一步最遠邊界 `farthest`|
|**LeetCode 1306**|Jump Game III（跳躍遊戲 III）|**1D 陣列，可往左或往右跳 `arr[i]` 步，判斷能否到達值為 `0` 的位置**|**BFS / DFS + `visited` 集合**（避免無限循環）|
|**LeetCode 1345**|Jump Game IV（跳躍遊戲 IV）|**1D 陣列，可跳相鄰格或相同數值的任意格，求最少步數**|**BFS + 雜湊表分組**（走過相同值後需清空該組避免超時）|
|**LeetCode 1871**|Jump Game VII（跳躍遊戲 VII）|**01 字串，每次跳 `[minJump, maxJump]` 步且只能落在 `'0'` 上**|**滑動視窗（Sliding Window）+ 動態規劃 / BFS**|
|**LeetCode 134 / LintCode 187**|Gas Station（加油站）|**環形陣列，判斷能否繞一圈並找出起點**|**貪心（Greedy）**，總油量夠就一定有解，油箱負數則重置起點|
|**LeetCode 1024**|Video Stitching（影片拼接）|**區間覆蓋問題，用最少片段覆蓋 `[0, time]`**|**貪心（Greedy）**，結構與 Jump Game II 完全相同|

### **進階變形：Jump Game II（求到達終點的最少跳躍次數）**

如果題目改為：假設你**一定能到達最後一個下標**，請計算到達終點的**最少跳躍次數**？

Python

```
class Solution:
    def jump(self, nums: list[int]) -> int:
        jumps = 0          # 記錄跳躍次數
        current_end = 0    # 當前這一步能到達的最遠邊界
        farthest = 0       # 在當前邊界內，下一步能到達的最遠位置
        
        # 注意：遍歷到 len(nums) - 2 即可，因為到達終點前最後一跳已經算進去了
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            # 走到當前這一跳的邊界了，必須再跳一次並更新邊界
            if i == current_end:
                jumps += 1
                current_end = farthest
                
                # 如果新邊界已經覆蓋終點，可提早結束
                if current_end >= len(nums) - 1:
                    break
                    
        return jumps
```