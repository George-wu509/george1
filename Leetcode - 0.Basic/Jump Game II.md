
LeetCode 45 / LintCode 117
给定一个长度为 `n` 的 **0 索引**整数数组 `nums`。初始位置为 `nums[0]`。
每个元素 `nums[i]` 表示从索引 `i` 向前跳转的最大长度。换句话说，如果你在 `nums[i]` 处，你可以跳转到任意 `nums[i + j]` 处:
- `0 <= j <= nums[i]`
- `i + j < n`

返回到达 `nums[n - 1]` 的最小跳跃次数。生成的测试用例可以到达 `nums[n - 1]`。

**样例 1：**
输入：
```python
nums = [2, 3, 1, 1, 4]
````

输出：
```Python
2
```

解释： 跳到最后一个位置的最小跳跃数是 `2`。 从下标为 0 跳到下标为 1 的位置，跳 `1` 步，然后跳 `3` 步到达数组的最后一个位置。

**样例 2：** 输入：
```Python
nums = [2, 3, 0, 1, 4]
```

输出：
```Python
2
```

解释： 先從索引 `0` 跳 `1` 步到索引 `1`（數值為 3），再從索引 `1` 直接跳 `3` 步到達終點索引 `4`，共 `2` 步。

**样例 3：** 输入：
```Python
nums = [1, 2, 1, 1, 1]
```

输出：
```Python
3
```

解释： 路徑為 `0 -> 1 -> 3 -> 4`，最少需要跳 `3` 次。
```Python
# 解法一：貪心演算法 Greedy（單次遍歷，最佳解）
class Solution:
    def jump(self, nums: list[int]) -> int:
        jumps = 0
        current_end = 0  # 當前這一步能到達的最遠邊界
        farthest = 0     # 下一跳能到達的最遠位置

        # 注意：只需遍歷到 len(nums) - 2，因為到達或超過終點就不需要再跳了
        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            # 抵達當前跳躍的邊界，必須進行下一次跳躍
            if i == current_end:
                jumps += 1
                current_end = farthest
                
        return jumps
```

```Python
# 解法二：1D BFS 層級區間雙指標寫法（NeetCode 經典寫法）
class Solution:
    def jump(self, nums: list[int]) -> int:
        res = 0
        l = r = 0  # [l, r] 代表當前跳躍步數所能覆蓋的索引區間

        while r < len(nums) - 1:
            farthest = 0
            for i in range(l, r + 1):
                farthest = max(farthest, i + nums[i])
            l = r + 1
            r = farthest
            res += 1
            
        return res
```

解釋: step1: 初始化三個變數：`jumps = 0`（跳躍次數）、`current_end = 0`（當前跳躍能觸及的最遠右邊界）、`farthest = 0`（掃描過程中發現的下一跳最遠可達位置）。 step2: 使用迴圈遍歷 `i` 從 `0` 到 `len(nums) - 2`（**注意不到最後一格**，避免剛好在終點時多算一次跳躍）。 step3: 每次在索引 `i` 時，更新下一跳能達到的最遠位置：`farthest = max(farthest, i + nums[i])`。 step4: 當指標 `i` 走到當前邊界 `i == current_end` 時，代表這一步的可選範圍已經全部探索完畢，必須正式跳出下一步：將 `jumps += 1`，並把邊界推進到 `current_end = farthest`。 step5: 若更新後的 `current_end` 已經大於等於 `len(nums) - 1`，後續迴圈不會再觸發邊界（或可提前 `break`），最終回傳 `jumps`。

解釋: 這題的核心思想是 **貪心（Greedy）＋ 1D BFS 分層擴展**。

每跳一步，都相當於在陣列上劃出一個新的「可達區間 `[l, r]`」；在當前區間內選擇能讓**下一跳邊界 `farthest` 推得最遠**的落腳點，就能保證總跳躍次數最少。

### **LeetCode 45 / LintCode 117 - Jump Game II（跳躍遊戲 II）**

這道題目是經典的 **貪心演算法（Greedy）** 問題，也可以看作是在一維陣列上進行 **廣度優先搜尋（1D BFS）** 的最短路徑問題。與 `Jump Game I`（只問能不能到達）不同，這題保證一定能到達終點，並要求**最少跳躍次數**。

## **解法解析**

這道題如果使用動態規劃（DP）需要 `O(n²)` 時間，但透過 **維護跳躍邊界的貪心策略**，可以在 `O(n)` 時間內一次解決。

### **思路**

1. **將跳躍過程視為「區間擴張（1D BFS）」**
    
    - 第 `0` 跳：位於索引 `0`，可達區間為 `[0, 0]`。
        
    - 第 `1` 跳：從 `[0, 0]` 出發最遠可到 `0 + nums[0]`，可達區間擴張為 `[1, farthest_1]`。
        
    - 第 `2` 跳：遍歷 `[1, farthest_1]` 內的所有格子，找出能跳最遠的落腳點，將區間擴張為 `[farthest_1 + 1, farthest_2]`。
        
    - 只要右邊界覆蓋到 `n - 1`，當前的層數就是最小跳躍次數。
        
2. **為什麼迴圈只跑到 `len(nums) - 2`？**
    
    - 假設陣列長度為 5（終點索引為 `4`）。
        
    - 如果 `current_end` 剛好在 `i = 3` 時更新成 `4`，我們已經到達終點了。
        
    - 若迴圈還跑到 `i = 4`，就會再次觸發 `i == current_end`，導致 `jumps` 多加 1 次不必要的跳躍。
        
3. **代碼執行流程**
    
    - 維護 `current_end` 作為當前這一跳的「決策死線」。
        
    - 在到達死線前，持續用 `i + nums[i]` 刷新 `farthest`（貪心挑選潛力最大的跳板）。
        
    - 一旦 `i == current_end`，立刻結算這一步（`jumps += 1`），並把新的死線設為 `farthest`。
        

## **具體舉例**

假設 `nums = [2, 3, 1, 1, 4]`（長度為 5，迴圈 `i` 執行 `0 ~ 3`），變數變化如下：

|**i**|**nums[i]**|**i + nums[i]**|**farthest**|**觸發 i == current_end?**|**jumps**|**current_end**|**說明**|
|---|---|---|---|---|---|---|---|
|初始化|-|-|`0`|-|`0`|`0`|起點在索引 0|
|`0`|`2`|`2`|`2`|**是 (`0 == 0`)**|**`1`**|**`2`**|第 1 跳最遠可到索引 2|
|`1`|`3`|`4`|`4`|否 (`1 != 2`)|`1`|`2`|發現索引 1 可直接跳到 4|
|`2`|`1`|`3`|`4`|**是 (`2 == 2`)**|**`2`**|**`4`**|到達第 1 跳邊界，結算第 2 跳|
|`3`|`1`|`4`|`4`|否 (`3 != 4`)|`2`|`4`|迴圈結束（不訪問索引 4）|

最後返回 `2`，表示最少需要 **2 次跳躍**（`0 -> 1 -> 4`）。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 單層 `for` 迴圈遍歷陣列 `n - 1` 次，每次操作皆為常數時間 `O(1)`，因此總時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**
    
    - 僅使用 `jumps`、`current_end`、`farthest` 三個整數變數，不需要額外陣列或佇列，空間複雜度為 **O(1)**。
        

## **其他解法**

### **1. 1D BFS 雙指標區間寫法（`[l, r]` 視窗推進）**

- **思路**：
    
    - 顯式維護當前跳躍步數的左右邊界 `l` 和 `r`，每次掃描 `l ~ r` 算出下一層的右邊界 `farthest`，再令 `l = r + 1, r = farthest`。
        
- **時間複雜度**：O(n)（每個元素最多被內層迴圈訪問一次）
    
- **空間複雜度**：O(1)
    

### **2. 動態規劃（Dynamic Programming）**

- **思路**：
    
    - 定義 `dp[i]` 為到達索引 `i` 的最少跳躍次數。對於每個 `i`，枚舉前面所有能跳到 `i` 的位置 `j`（`j + nums[j] >= i`），轉移方程為 `dp[i] = min(dp[i], dp[j] + 1)`。
        
- **時間複雜度**：O(n²)（在 LeetCode 大測資下較慢，不如貪心高效）
    
- **空間複雜度**：O(n)
    

## **總結**

- **最佳解法（面試首選）**：**單次遍歷貪心法（`current_end` + `farthest`）**，程式碼僅 10 行，時間 `O(n)`、空間 `O(1)`。
    
- **最易理解記憶法**：**1D BFS 雙指標區間 `[l, r]`**，能直觀看出「每一跳就是一層 BFS 擴張」，不容易寫錯邊界條件 `len(nums) - 1`。
    
- **關鍵細節**：遍歷終點必須停在 **`len(nums) - 2`**，避免在最後一格重複計算跳躍。
    

## **LeetCode 跳躍遊戲與貪心類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 55**|Jump Game（跳躍遊戲）|**1D 陣列可達性判斷（能否到達終點）**|**貪心維護最遠可達索引 `max_reach`**，`O(n)` 時間|
|**LeetCode 45**|Jump Game II（跳躍遊戲 II）|**1D 陣列最短路徑（求最少跳躍次數）**|**貪心邊界推進 / 1D BFS 分層**，`O(n)` 時間 `O(1)` 空間|
|**LeetCode 1306**|Jump Game III（跳躍遊戲 III）|**1D 陣列雙向跳躍（可往左或往右跳 `arr[i]`）**|**BFS / DFS + `visited` 集合防死循環**，`O(n)`|
|**LeetCode 1345**|Jump Game IV（跳躍遊戲 IV）|**1D 陣列等值傳送 + 左右移動求最少步數**|**BFS + Hash Map 記錄同值索引（用完即刪防超時）**|
|**LeetCode 1340**|Jump Game V（跳躍遊戲 V）|**1D 陣列有向無環圖（只能往矮處跳且不越過高牆）**|**記憶化搜索（Memoization DFS）或排序後 DP**|
|**LeetCode 1696**|Jump Game VI（跳躍遊戲 VI）|**1D 陣列最大路徑和（每次最多跳 `k` 步）**|**動態規劃 + 單調佇列（Monotonic Deque）優化至 `O(n)`**|
|**LeetCode 1024**|Video Stitching（視頻拼接）|**區間覆蓋問題（選最少片段覆蓋 `[0, T]`）**|**區間貪心（與 Jump Game II 邏輯完全等價）**|

### **進階變形：如果題目「不保證」一定能到達終點，無法到達時回傳 `-1`，該如何修改？**

Python

```
class Solution:
    def jumpWithUnreachableCheck(self, nums: list[int]) -> int:
        if len(nums) <= 1:
            return 0
            
        jumps = 0
        current_end = 0
        farthest = 0

        for i in range(len(nums) - 1):
            farthest = max(farthest, i + nums[i])
            
            if i == current_end:
                # 關鍵檢查：如果當前邊界已經是極限，無法再往後推進，代表卡死無法到達終點
                if farthest <= current_end:
                    return -1
                jumps += 1
                current_end = farthest
                if current_end >= len(nums) - 1:
                    return jumps
                    
        return jumps
```

```

<ElicitationsGroup message="繼續整理下一題 Blind 75 / NeetCode 150 筆記：">
  <Elicitation label="LeetCode 55. Jump Game（跳躍遊戲 I）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 55. Jump Game（跳躍遊戲）的完整筆記。"/>
  <Elicitation label="LeetCode 53. Maximum Subarray（最大子數組和）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 53. Maximum Subarray（最大子數組和）的完整筆記。"/>
  <Elicitation label="LeetCode 134. Gas Station（加油站）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 134. Gas Station（加油站）的完整筆記。"/>
</ElicitationsGroup>
```