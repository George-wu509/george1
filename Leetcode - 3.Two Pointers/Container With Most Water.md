
LeetCode 11 / LintCode 383 給定一個長度為 `n` 的整數陣列 `height`。有 `n` 條垂線，第 `i` 條線的兩個端點是 `(i, 0)` 和 `(i, height[i])`。

找出其中的兩條線，使得它們與 `x` 軸共同構成的容器可以容納最多的水。返回容器可以儲存的最大水量。
**說明**：你不能傾斜容器。

比較: 363接雨水 [[Trapping Rain Water]]

**範例 1：** 輸入：
```Python
height = [1, 8, 6, 2, 5, 4, 8, 3, 7]
```

輸出：
```Python
49
```

解釋： 圖中垂直線代表輸入陣列 `[1, 8, 6, 2, 5, 4, 8, 3, 7]`。在此情況下，容器能夠容納水（選擇索引 `1` 高度為 `8` 與索引 `8` 高度為 `7` 的兩條線）的最大值為 `(8 - 1) * min(8, 7) = 7 * 7 = 49`

**範例 2：** 輸入：
```Python
height = [1, 1]
```

輸出：
```Python
1
```

解釋： 只有兩條高度為 `1` 的線，寬度為 `1`，最大容水量為 `1 * 1 = 1`

**範例 3：** 輸入：
```Python
height = [4, 3, 2, 1, 4]
```

輸出：
```Python
16
```

解釋： 選擇最左邊（索引 `0`，高度 `4`）與最右邊（索引 `4`，高度 `4`）的線，寬度為 `4`，高度為 `4`，最大容水量為 `4 * 4 = 16`

### 解法一：相向雙指針（Optimal - $O(n)$）

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left, right = 0, len(height) - 1
        max_water = 0
        
        while left < right:
            width = right - left
            current_height = min(height[left], height[right])
            max_water = max(max_water, width * current_height)
            
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
                
        return max_water
```
解釋版
```Python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        # 1. 初始化左右指針在陣列最兩端，max_water 紀錄歷史最大水量
        left, right = 0, len(height) - 1
        max_water = 0
        
        # 2. 當兩條線還沒重疊時持續尋找
        while left < right:
            # 計算目前兩線之間的寬度
            width = right - left
            # 容器高度取決於較短的那根柱子
            current_height = min(height[left], height[right])
            # 更新最大儲水量
            max_water = max(max_water, width * current_height)
            
            # 3. 核心貪心決策：誰比較矮，就淘汰誰（往內移動一格）
            if height[left] < height[right]:
                left += 1
            else:
                right -= 1
                
        return max_water
```
解釋(my):
step1: 左右指針(left, right)從左邊右邊往中間移動
step2: 維持while left < right, 並計算每一次max_water
step3: 比對左指針跟右指針高度(height[left], height[right]). 哪個比較小就往中間移動


```
                          left(8)與 right(8)一樣高，    8 |    █ ~ ~ ~ ~ █    <--   水量 = 40
7 |    █ ~ ~ ~ ~ █   █
6 |    █ █ ~ ~ ~ █   █
5 |    █ █ ~ █ ~ █   █
4 |    █ █ ~ █ █ █   █
3 |    █ █ ~ █ █ █ █ █
2 |    █ █ █ █ █ █ █ █
1 |  █ █ █ █ █ █ █ █ █
  +--------------------
idx: 0 1 2 3 4 5 6 7 8
```

```
高度(y)
8 |  █             █
7 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 7 層：7 個 ~
6 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 6 層：7 個 ~
5 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 5 層：7 個 ~
4 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 4 層：7 個 ~
3 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 3 層：7 個 ~
2 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 2 層：7 個 ~
1 |  █ ~ ~ ~ ~ ~ ~ █  <-- 第 1 層：7 個 ~
  +------------------
idx: 1 2 3 4 5 6 7 8  (寬度 = 8 - 1 = 7 格)
```



### 解法二：暴力枚舉法（Brute Force - $O(n^2)$，會超時 TLE）

```Python
from typing import List

class Solution:
    def maxArea(self, height: List[int]) -> int:
        n = len(height)
        max_water = 0
        
        # 枚舉所有可能的左右邊界組合 (i, j)
        for i in range(n):
            for j in range(i + 1, n):
                area = (j - i) * min(height[i], height[j])
                max_water = max(max_water, area)
                
        return max_water
```

解釋:

- `step1`: 初始化左右雙指針 `left = 0`、`right = len(height) - 1`，分別指向陣列最左端與最右端（此時底邊寬度最大），並設 `max_water = 0`。
    
- `step2`: 計算當前指針構成的容器面積：`area = (right - left) * min(height[left], height[right])`，並更新 `max_water`。
    
- `step3`: 以 `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]` 為例，初始 `left = 0`（高度 `1`）、`right = 8`（高度 `7`），面積為 `(8 - 0) * min(1, 7) = 8`，`max_water = 8`。
    
- `step4`: 比較兩側高度：`height[0] (1) < height[8] (7)`，左側較短，因此將左指針右移 `left = 1`（高度 `8`）。
    
- `step5`: 第 2 輪：`left = 1`（高度 `8`）、`right = 8`（高度 `7`），面積為 `(8 - 1) * min(8, 7) = 49`，更新 `max_water = 49`。接著因為 `height[8] (7) < height[1] (8)`，右側較短，右指針左移 `right = 7`。
    
- `step6`: 重複上述過程不斷向內收縮，直到 `left == right` 時結束迴圈，最終回傳記錄的最大面積 `49`。
    

解釋: 容器的盛水量由公式 $\text{Area} = (right - left) \times \min(\text{height}[left], \text{height}[right])$ 決定。

當指針向內移動時，寬度 `(right - left)` 必然減少 `1`。若移動**較長**的那一側，新高度只會小於或等於原本的短邊，面積**一定變小**；唯有移動**較短**的那一側，才有可能遇到更高的線段，從而使面積增大。

### **LeetCode 11 / LintCode 383 - Container With Most Water（盛最多水的容器）**

這道題目是 **雙指針（Two Pointers）+ 貪心策略（Greedy）** 的經典代表題，也是 Blind 75 與 NeetCode 150 的核心必考題。題目表面上需要檢查 $O(n^2)$ 對邊界組合，但透過單調性剪枝，可以在一次線性掃描內排除所有無效狀態。

## **解法解析**

這道題可以轉化為 **從最大寬度開始，逐步排除不可能產生更大面積的短邊** 來求解。

### **思路**

1. **面積公式與木桶效應**
    
    - 選定左右兩條垂線 `left` 和 `right` 時，水面不能傾斜，因此水位高度受限於較矮的那條線：
        
        $$\text{Area}(left, right) = (right - left) \times \min(\text{height}[left], \text{height}[right])$$
        
2. **為什麼每次必須移動較短的指針？（狀態空間縮減證明）**
    
    - 假設目前 `height[left] < height[right]`，此時以 `left` 為左邊界、以任意 `k`（`left < k < right`）為右邊界的所有組合中：
        
        - 寬度一定小於 `right - left`（因為 `k < right`）。
            
        - 高度一定小於或等於 `height[left]`（因為 $\min(\text{height}[left], \text{height}[k]) \le \text{height}[left]$）。
            
    - 這意味著：**所有以當前 `left` 為左邊界、且右邊界在 `right` 左側的組合，面積絕對不可能超過 `(left, right)`**！
        
    - 因此，我們可以安全地把 `left` 整個淘汰（`left += 1`），每次移動都等價於在搜尋矩陣中直接劃掉一整行或一整列的無效狀態。
        
3. **兩側高度相等時的處理**
    
    - 當 `height[left] == height[right]` 時，移動任意一側都可以（因為無論留哪一邊，下一次的短邊上限都不會超過當前的高度，除非兩側都往內找到更高的線段，而這兩條線最終都會被雙指針相遇前遍歷到）。
        

## **具體舉例**

假設 `height = [1, 8, 6, 2, 5, 4, 8, 3, 7]`，雙指針向內收縮的完整過程如下：

|**輪次**|**left (高度)**|**right (高度)**|**寬度 right - left**|**有效高度 min**|**當前面積**|**max_water**|**移動決策**|
|---|---|---|---|---|---|---|---|
|第 1 輪|`0` (`1`)|`8` (`7`)|`8`|`1`|`8`|`8`|`1 < 7`，`left += 1`|
|第 2 輪|`1` (`8`)|`8` (`7`)|`7`|`7`|`49`|**`49`**|`8 > 7`，`right -= 1`|
|第 3 輪|`1` (`8`)|`7` (`3`)|`6`|`3`|`18`|`49`|`8 > 3`，`right -= 1`|
|第 4 輪|`1` (`8`)|`6` (`8`)|`5`|`8`|`40`|`49`|`8 == 8`，`right -= 1`|
|第 5 輪|`1` (`8`)|`5` (`4`)|`4`|`4`|`16`|`49`|`8 > 4`，`right -= 1`|
|第 6 輪|`1` (`8`)|`4` (`5`)|`3`|`5`|`15`|`49`|`8 > 5`，`right -= 1`|
|第 7 輪|`1` (`8`)|`3` (`2`)|`2`|`2`|`4`|`49`|`8 > 2`，`right -= 1`|
|第 8 輪|`1` (`8`)|`2` (`6`)|`1`|`6`|`6`|`49`|`8 > 6`，`right -= 1`|
|結束|`1`|`1`|—|—|—|**`49`**|`left == right`，終止迴圈|

## **時間與空間複雜度分析**

- **時間複雜度：$O(n)$**
    
    - 左右指針分別從陣列兩端出發，每次迭代必定有一個指針向中間移動一步，兩個指針總共移動 `n - 1` 次後相遇，因此只需一次線性遍歷，時間複雜度為 $O(n)$。
        
- **空間複雜度：$O(1)$**
    
    - 僅使用常數個變數（`left`, `right`, `max_water`, `width`, `current_height`），不需要任何額外儲存空間。
        

## **其他解法**

### **1. 雙指針 + 跳過更矮線段剪枝（常數優化）**

- **思路**：
    
    - 在移動較短指針後，如果新的線段高度比剛才淘汰的短邊還要矮（或相等），由於寬度又變小了，面積一定更小，可以直接用 `while` 迴圈連續跳過，直到遇到比剛才更高的線段再計算面積。
        
- **時間複雜度**：$O(n)$（最壞情況相同，平均常數運算次數更少）
    
- **空間複雜度**：$O(1)$
    

### **2. 暴力雙重迴圈枚舉**

- **思路**：
    
    - 窮舉所有 $0 \le i < j < n$ 的組合並計算面積取最大值。
        
- **時間複雜度**：$O(n^2)$（當 $n = 10^5$ 時會超時）
    
- **空間複雜度**：$O(1)$
    

## **總結**

- **面試最優解**：**相向雙指針（Two Pointers）**，時間 $O(n)$、空間 $O(1)$。
    
- **面試加分點**：面試官非常喜歡追問「**為什麼移動短邊不會漏掉最優解？**」能夠清晰說出「固定短邊時，寬度隨之減小且高度受限於該短邊，因此該短邊與其他所有內側邊的組合都被嚴格支配（Dominated）」，就能展現紮實的演算法推導能力。
    
- **易錯細節**：
    
    1. 迴圈條件為 `while left < right:`（兩指針重合時寬度為 `0`，無需計算）。
        
    2. 不要和「接雨水（Trapping Rain Water）」混淆：本題只需選**兩條線**構成單一容器（中間的其他線段不佔體積也不阻擋水流），而接雨水是計算**所有柱子之間凹槽**累積的總水量。
        

## **Blind 75 + NeetCode 150 雙指針（Two Pointers）類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 11**|Container With Most Water（盛最多水的容器）|**相向雙指針 + 貪心剪枝**|**左右指針從兩端向內收縮，每次移動較短邊界，時間 $O(n)$**|
|**LeetCode 42**|Trapping Rain Water（接雨水）|**相向雙指針 / 前後綴最大值**|**維護 `left_max` 與 `right_max`，每次結算較小一側的積水量，時間 $O(n)$**|
|**LeetCode 125**|Valid Palindrome（驗證回文串）|**基礎相向雙指針**|**左右指針跳過非字母數字字元，逐一比對大小寫是否一致，時間 $O(n)$**|
|**LeetCode 167**|Two Sum II - Input Array Is Sorted（兩數之和 II）|**有序陣列相向雙指針**|**利用陣列升序特性，和大於 `target` 右指針左移，小於則左指針右移，時間 $O(n)$**|
|**LeetCode 15**|3Sum（三數之和）|**排序 + 固定一點 + 相向雙指針**|**先排序，外層枚舉第一個數並去重，內層用雙指針找兩數和，時間 $O(n^2)$**|

### 延伸變形題：接雨水（LeetCode 42 - Trapping Rain Water，姐妹題）

給定 `n` 個非負整數表示每個寬度為 `1` 的柱子的高度圖，計算按此排列的柱子，下雨之後能接多少雨水（與 LeetCode 11 的差別在於：本題每個位置能裝的水取決於「左側最高柱子」與「右側最高柱子」的較小值減去當前柱子高度）：

Python

```
from typing import List

class Solution:
    def trap(self, height: List[int]) -> int:
        if not height:
            return 0
            
        left, right = 0, len(height) - 1
        left_max, right_max = height[left], height[right]
        total_water = 0
        
        while left < right:
            # 哪一側的最大值較小，該側的積水上限就已經被確定
            if left_max < right_max:
                left += 1
                left_max = max(left_max, height[left])
                total_water += left_max - height[left]
            else:
                right -= 1
                right_max = max(right_max, height[right])
                total_water += right_max - height[right]
                
        return total_water
```