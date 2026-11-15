
打家劫舍 II House Robber II

LeetCode 213 / LintCode 534 你是一个专业的小偷，计划偷窃沿街的房屋，每间房内都藏有一定的现金。这个地方所有的房屋都 **围成一圈** ，这意味着第一个房屋和最后一个房屋是紧挨着的。同时，相邻的房屋装有相互连通的防盗系统，**如果两间相邻的房屋在同一晚上被小偷闯入，系统会自动报警** 。

给定一个代表每个房屋存放金额的非负整数数组，计算你 **在不触动警报装置的情况下** ，今晚能够偷窃到的最高金额。

**样例 1：** 输入：
```Python
nums = [2, 3, 2]
```

输出：
```Python
3
```

解释： 你不能先偷窃 1 号房屋（金额 = 2），然后偷窃 3 号房屋（金额 = 2）, 因为他们是相邻的。

**样例 2：** 输入：
```Python
nums = [1, 2, 3, 1]
```

输出：
```Python
4
```

解释： 你可以先偷窃 1 号房屋（金额 = 1），然后偷窃 3 号房屋（金额 = 3）。 偷窃到的最高金额 = 1 + 3 = 4 。

**样例 3：** 输入：
```Python
nums = [1, 2, 3]
```

输出：
```Python
3
```

解释： 三間房屋圍成一圈，任意兩間都相鄰，只能偷金額最大的 3 號房屋（金額 = 3）。

```Python
# 解法一：拆解為兩個一維 DP 陣列問題
def rob(self, nums: list[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    def rob_dp(houses: list[int]) -> int:
        n = len(houses)
        if n == 1:
            return houses[0]
        dp = [0] * n
        dp[0] = houses[0]
        dp[1] = max(houses[0], houses[1])
        for i in range(2, n):
            dp[i] = max(dp[i - 1], dp[i - 2] + houses[i])
        return dp[-1]

    return max(rob_dp(nums[:-1]), rob_dp(nums[1:]))
```


```Python
# 解法二：空間優化 O(1) 滾動變數
def rob(self, nums: list[int]) -> int:
    if not nums:
        return 0
    if len(nums) == 1:
        return nums[0]

    def rob_linear(houses: list[int]) -> int:
        prev_two = 0
        prev_one = 0
        for money in houses:
            temp = max(prev_one, prev_two + money)
            prev_two = prev_one
            prev_one = temp
        return prev_one

    return max(rob_linear(nums[:-1]), rob_linear(nums[1:]))
```

解釋: step1: 因為房屋圍成一圈，**第一間 (`nums[0]`) 與最後一間 (`nums[-1]`) 不能同時偷**，因此可以將環形問題拆解為兩個獨立的線性「打家劫舍 I」問題：

- 情況 A：不偷最後一間，考慮範圍 `nums[:-1]`
    
- 情況 B：不偷第一間，考慮範圍 `nums[1:]` step2: 以 `nums = [1, 2, 3, 1]` 為例，首先特判陣列長度為 1 的情況（本例長度為 4，繼續執行）。 step3: 計算情況 A `rob_linear([1, 2, 3])`：
    
- `money = 1`：`temp = max(0, 0+1) = 1` → `prev_two = 0, prev_one = 1`
    
- `money = 2`：`temp = max(1, 0+2) = 2` → `prev_two = 1, prev_one = 2`
    
- `money = 3`：`temp = max(2, 1+3) = 4` → `prev_two = 2, prev_one = 4`
    
- 情況 A 最大值 = `4` step4: 計算情況 B `rob_linear([2, 3, 1])`：
    
- `money = 2`：`temp = max(0, 0+2) = 2` → `prev_two = 0, prev_one = 2`
    
- `money = 3`：`temp = max(2, 0+3) = 3` → `prev_two = 2, prev_one = 3`
    
- `money = 1`：`temp = max(3, 2+1) = 3` → `prev_two = 3, prev_one = 3`
    
- 情況 B 最大值 = `3` step5: 取兩者最大值 `max(4, 3) = 4`，所以答案 = `4`。
    

解釋: 環形排列唯一的額外限制是「首尾相鄰」，這代表第一間與最後一間最多只能選一間（也可以兩間都不選）：

1. **排除最後一間 (`nums[0 : n-1]`)**：保證不會同時觸發首尾警報，在此範圍內「可以選第一間，也可以不選第一間」。
    
2. **排除第一間 (`nums[1 : n]`)**：同樣保證不會同時觸發首尾警報，在此範圍內「可以選最後一間，也可以不選最後一間」。 這兩種情況已經完整覆蓋了所有合法的偷竊組合（包含兩頭都不偷的情況），對兩者分別執行線性 DP `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` 再取最大值即可。
    

### **LeetCode 213 / LintCode 534 - House Robber II（打家劫舍 II）**

這道題目是 **House Robber（打家劫舍 I）** 的環形進階版，屬於典型的 **環形陣列動態規劃（Circular DP）** 問題。處理環形 DP 的通用核心技巧是「破環成鏈」：透過分類討論首尾元素的取捨，將環形問題轉化為兩個不帶環的線性子問題。

## **解法解析**

這道題可以透過「破環成鏈」轉化為兩次標準的線性動態規劃來求解。

### **思路**

1. **觀察規律**
    
    - 若只有 `1` 間房屋（如 `[5]`）：沒有其他相鄰房屋，直接偷這間，回傳 `nums[0]`。
        
    - 若有 `2` 間房屋（如 `[2, 3]`）：首尾相鄰只能選一間，回傳 `max(2, 3) = 3`（套用拆解公式 `max(rob([2]), rob([3]))` 同樣成立）。
        
    - 若有 `n` 間房屋（`n` 大於等於 `3`）：首尾相連導致 `nums[0]` 和 `nums[n-1]` 互斥，只要把它們分別從候選名單中剔除一次，剩下的區間就是完全正常的直線排列。
        
2. **發現轉化與遞推公式**
    
    - 定義線性輔助函數 `rob_linear(houses)`，狀態轉移方程與打家劫舍 I 相同：
        
        - `dp[i] = max(dp[i-1], dp[i-2] + houses[i])`
            
    - 最終答案為兩種合法區間的最優解取較大值：
        
        - `ans = max(rob_linear(nums[0 : n-1]), rob_linear(nums[1 : n]))`
            
3. **代碼執行流程**
    
    - 先處理邊界條件：若 `nums` 為空回傳 `0`；若長度為 `1` 直接回傳 `nums[0]`（若不特判，切片 `nums[:-1]` 和 `nums[1:]` 都會變成空陣列而誤回傳 `0`）。
        
    - 定義 `rob_linear` 函數，利用兩個滾動變數 `prev_two` 與 `prev_one` 計算線性陣列的最大收益。
        
    - 分別傳入 `nums[:-1]` 與 `nums[1:]` 呼叫 `rob_linear`，並返回兩者的最大值。
        

## **具體舉例**

假設 `nums = [2, 3, 2]`，拆分成兩個線性子陣列後的計算過程如下：

|**拆解情況**|**考慮區間**|**遍歷過程 (prev_two, prev_one)**|**該區間最大值**|
|---|---|---|---|
|**情況 A（不偷尾）**|`[2, 3]`|`money=2` → `(0, 2)`<br><br>  <br><br>`money=3` → `(2, 3)`|`3`|
|**情況 B（不偷頭）**|`[3, 2]`|`money=3` → `(0, 3)`<br><br>  <br><br>`money=2` → `(3, 3)`|`3`|

最後比較 `max(3, 3)` 返回 `3`，表示最高能偷竊 **3** 元（只偷中間的 2 號房屋）。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 需要對長度為 `n - 1` 的子陣列進行兩次線性遍歷，總操作次數約為 `2n`，因此時間複雜度為 **O(n)**。
        
- **空間複雜度：O(1)**（嚴格指標傳遞下）
    
    - `rob_linear` 內部僅使用常數個滾動變數，空間複雜度為 **O(1)**。（註：在 Python 中直接寫 `nums[:-1]` 切片會產生 `O(n)` 的新列表副本；若改為傳入起訖索引 `rob_linear(start, end)`，則空間複雜度為嚴格的 **O(1)**）。
        

## **其他解法**

### **1. 傳遞索引取代陣列切片（嚴格 O(1) 空間優化）**

- **思路**：
    
    - 避免 Python 列表切片 `nums[:-1]` 複製陣列帶來的額外記憶體開銷，將輔助函數改寫為 `rob_range(start, end)`，直接用 `for i in range(start, end + 1)` 存取原陣列 `nums[i]`。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(1)
    

### **2. 顯式狀態機 DP（記錄第一間是否被偷）**

- **思路**：
    
    - 將狀態擴展為二維 `dp[i][first_robbed]`，分別記錄「第 0 間有偷」與「第 0 間沒偷」兩種平行宇宙下的轉移過程，最後在 `n - 1` 時依據 `first_robbed` 決定能否偷最後一間。邏輯本質上與破環成鏈完全等價。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(1)
    

## **總結**

- **面試首選最佳解法**：**破環成鏈 + O(1) 滾動變數動態規劃**（若面試官追問空間細節，可進一步將切片改為傳遞左右邊界索引 `start, end`）。
    
- **常見易錯點**：
    
    1. **忘記特判 `len(nums) == 1`**：當只有一間房時，`nums[:-1]` 和 `nums[1:]` 皆為空陣列 `[]`，會錯誤輸出 `0` 而非 `nums[0]`。
        
    2. **誤以為只需比較奇數項總和與偶數項總和**（當 `n` 為偶數時），這在環形陣列中同樣會漏掉「連續跳過兩間」的最佳解（例如 `[2, 1, 1, 2, 1]`）。
        

## **LeetCode / LintCode 環形陣列與打家劫舍類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 213**|House Robber II（打家劫舍 II）|**1D 環形陣列，不能偷相鄰房屋，求最大金額**|**破環成鏈：`max(rob(nums[:-1]), rob(nums[1:]))`**|
|**LeetCode 198**|House Robber（打家劫舍）|**1D 線性陣列，不能偷相鄰房屋，求最大金額**|**動態規劃 `DP[i] = max(DP[i-1], DP[i-2] + nums[i])`**|
|**LeetCode 337**|House Robber III（打家劫舍 III）|**二元樹結構，父子節點不能同時偷**|**樹形 DP（Post-order DFS 回傳 `[不偷當前, 偷當前]`）**|
|**LeetCode 918**|Maximum Sum Circular Subarray（環形子陣列的最大和）|**1D 環形連續子陣列，求最大總和**|**Kadane 變形：`max(一般最大和, 總和 - 最小子陣列和)`**|
|**LeetCode 1388**|Pizza With 3n Slices（3n 塊披薩）|**環形陣列選 `n` 個不相鄰元素的最大總和**|**破環成鏈（去掉頭或去掉尾）+ 二維選擇計數 DP**|
|**LeetCode 740**|Delete and Earn（刪除並獲得點數）|**選取數字 `x` 後必須刪除所有 `x-1` 與 `x+1`**|**統計每個數字總分後，轉化為標準 House Robber 問題**|
|**LeetCode 2560**|House Robber IV（打家劫舍 IV）|**至少偷 `k` 間不相鄰房屋，求單間最大金額的最小值**|**二分搜尋答案（Binary Search）+ 貪心驗證**|

### **延伸變形題：打家劫舍 III（LeetCode 337 - 二元樹房屋）**

如果小偷面臨的社區不再是直線或圓環，而是一棵**二元樹**（相連的父節點與子節點不能同時被偷），則需使用**樹形動態規劃（Tree DP）**，透過後序遍歷（Post-order DFS）讓每個節點向上回傳 `[不偷該節點的最大收益, 偷該節點的最大收益]`：

Python

```
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:
        def dfs(node: Optional[TreeNode]) -> tuple[int, int]:
            if not node:
                return (0, 0)  # (不偷當前節點, 偷當前節點)

            left_not_rob, left_rob = dfs(node.left)
            right_not_rob, right_rob = dfs(node.right)

            # 不偷當前節點：左右子節點可偷可不偷，各自取最大值相加
            not_rob_curr = max(left_not_rob, left_rob) + max(right_not_rob, right_rob)
            # 偷當前節點：左右子節點皆不能偷
            rob_curr = node.val + left_not_rob + right_not_rob

            return (not_rob_curr, rob_curr)

        return max(dfs(root))
```