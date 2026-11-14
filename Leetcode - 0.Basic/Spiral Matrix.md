

LeetCode 54 / LintCode 374
给你一个 `m` 行 `n` 列的矩阵 `matrix` ，请按照 **顺时针螺旋顺序** ，返回矩阵中的所有元素。

**样例 1：**
输入：
```python
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
````

输出：
```Python
[1, 2, 3, 6, 9, 8, 7, 4, 5]
```

解释：

外圈順時針依序走過 `1 -> 2 -> 3`（上邊）、`6 -> 9`（右邊）、`8 -> 7`（下邊）、`4`（左邊），最後收縮到正中央的 `5`。

**样例 2：**

输入：
```Python
matrix = [
    [1,  2,  3,  4],
    [5,  6,  7,  8],
    [9, 10, 11, 12]
]
```

输出：
```Python
[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]
```

解释：

非方陣（`3 x 4`）的螺旋遍歷，外圈走完後內圈只剩單一行 `[6, 7]`，從左到右讀取完畢即結束，不可重複往回讀取。

**样例 3：**

输入：
```Python
matrix = [
    [7],
    [9],
    [6]
]
```

输出：
```Python
[7, 9, 6]
```

解释：

單列矩陣（`3 x 1`），從上到下讀取完畢後左右邊界已交錯，直接結束。

```Python
# 解法一：四邊界收縮模擬法（最推薦、零重複走訪風險）
class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        res = []
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)

        while left < right and top < bottom:
            # 1. 從左到右遍歷上邊界（top 行）
            for i in range(left, right):
                res.append(matrix[top][i])
            top += 1

            # 2. 從上到下遍歷右邊界（right - 1 列）
            for i in range(top, bottom):
                res.append(matrix[i][right - 1])
            right -= 1

            # 關鍵邊界檢查：若只剩單行或單列，此時邊界已交錯，必須提前結束
            if not (left < right and top < bottom):
                break

            # 3. 從右到左遍歷下邊界（bottom - 1 行）
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[bottom - 1][i])
            bottom -= 1

            # 4. 從下到上遍歷左邊界（left 列）
            for i in range(bottom - 1, top - 1, -1):
                res.append(matrix[i][left])
            left += 1

        return res
```


```Python
# 解法二：方向向量模擬法（Direction Vector + 碰壁轉向）
class Solution:
    def spiralOrder(self, matrix: list[list[int]]) -> list[int]:
        if not matrix or not matrix[0]:
            return []
            
        rows, cols = len(matrix), len(matrix[0])
        visited = [[False] * cols for _ in range(rows)]
        res = []
        
        # 順時針四個方向：右 -> 下 -> 左 -> 上
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        r = c = d = 0  # d 為當前方向索引
        
        for _ in range(rows * cols):
            res.append(matrix[r][c])
            visited[r][c] = True
            
            # 試探下一步座標
            nr, nc = r + directions[d][0], c + directions[d][1]
            # 若越界或已訪問過，則順時針旋轉 90 度換方向
            if not (0 <= nr < rows and 0 <= nc < cols and not visited[nr][nc]):
                d = (d + 1) % 4
                nr, nc = r + directions[d][0], c + directions[d][1]
                
            r, c = nr, nc
            
        return res
```

解釋:

step1: 設定四個半開區間邊界指標：`left = 0`、`right = len(matrix[0])`、`top = 0`、`bottom = len(matrix)`，代表尚未訪問的有效矩形區域為 `[top, bottom) × [left, right)`。

step2: 當 `left < right and top < bottom` 成立時進入迴圈，依序執行四個方向的掃描：

step3: **向右走**：固定 `top` 行，遍歷 `left` 到 `right - 1`，走完後上邊界下移 `top += 1`。

step4: **向下走**：固定 `right - 1` 列，遍歷 `top` 到 `bottom - 1`，走完後右邊界左移 `right -= 1`。

step5: **關鍵防呆檢查**：因為剛才 `top` 和 `right` 都更新過了，若原矩陣是非方陣（如 `3 x 4` 的最後一層只剩一行），此時 `top == bottom` 或 `left == right`，必須立刻 `break`，否則接下來的「向左走」與「向上走」會重複讀取同一批元素。

step6: **向左走**（固定 `bottom - 1` 行，遍歷 `right - 1` 到 `left`，之後 `bottom -= 1`）與 **向上走**（固定 `left` 列，遍歷 `bottom - 1` 到 `top`，之後 `left += 1`），重複直到邊界交錯並回傳 `res`。

解釋:

這題最大的陷阱不在方陣（`n x n`），而在長方形矩陣（`m x n`，`m != n`）的最後一圈。

每走完一條邊就立刻將對應邊界向內收縮 `1` 格，並在走完「右、下」兩邊後做一次邊界交錯檢查 `if not (left < right and top < bottom): break`，即可完美處理單行、單列等所有邊界情況。

### **LeetCode 54 / LintCode 374 - Spiral Matrix（螺旋矩陣）**

這道題目是經典的 **二維矩陣邊界模擬（Matrix Boundary Simulation）** 問題。不需要複雜的演算法或資料結構，核心考驗的是**邊界條件控制的精準度**（尤其是長方形矩陣與單行/單列矩陣）。

## **解法解析**

這道題採用 **四邊界收縮法（Four-Boundary Shrinking）** 最為乾淨，能達到 `O(1)` 額外空間複雜度。

### **思路**

1. **為什麼採用「左閉右開 `[left, right)`」區間？**
    
    - 傳統寫法常用閉區間 `right = cols - 1, bottom = rows - 1`，迴圈條件要寫 `left <= right and top <= bottom`，且 `range` 經常需要 `+1` 或 `-1`，容易搞混。
        
    - 改用半開區間 `right = cols, bottom = rows` 後，迴圈條件統一為 `left < right and top < bottom`，且從左到右直接寫 `range(left, right)`，語意更一致。
        
2. **為什麼中間一定要加 `if not (left < right and top < bottom): break`？**
    
    - 在一輪迴圈中：
        
        - 步驟 1（向右）會讓 `top += 1`
            
        - 步驟 2（向下）會讓 `right -= 1`
            
    - 如果矩陣原本是 `1 x 4`（只有 1 行），步驟 1 走完所有元素後 `top` 變成 `1`（等於 `bottom`）。如果沒有中間的 `break`，步驟 3（向左）就會把同一行再倒著走一遍，導致結果變成 `[1, 2, 3, 4, 3, 2, 1]`。
        
3. **代碼執行流程**
    
    - 依序沿著矩形外框走「上邊 → 右邊 →（檢查邊界）→ 下邊 → 左邊」。
        
    - 每走完一條邊，就把該方向的邊界向內縮減 1 格，直到上下或左右邊界相遇。
        

## **具體舉例**

以 **样例 2**（`3 x 4` 矩陣 `[[1,2,3,4], [5,6,7,8], [9,10,11,12]]`）為例，追蹤邊界收縮過程：

|**輪次**|**步驟與方向**|**讀取元素**|**操作後邊界 (top, bottom, left, right)**|**說明**|
|---|---|---|---|---|
|**初始**|-|`[]`|`(0, 3, 0, 4)`|初始有效範圍 `3 x 4`|
|**第 1 輪**|1. 向右（`top=0`）|`[1, 2, 3, 4]`|`(1, 3, 0, 4)`|上邊界下移|
|**第 1 輪**|2. 向下（`right-1=3`）|`[8, 12]`|`(1, 3, 0, 3)`|右邊界左移（通過 `break` 檢查）|
|**第 1 輪**|3. 向左（`bottom-1=2`）|`[11, 10, 9]`|`(1, 2, 0, 3)`|下邊界上移|
|**第 1 輪**|4. 向上（`left=0`）|`[5]`|`(1, 2, 1, 3)`|左邊界右移|
|**第 2 輪**|1. 向右（`top=1`）|`[6, 7]`|`(2, 2, 1, 3)`|上邊界下移，此時 `top == bottom`|
|**第 2 輪**|2. 向下（`range(2, 2)`）|無|`(2, 2, 1, 2)`|**觸發 `top < bottom` 為 False，執行 `break`**|

最終輸出 `[1, 2, 3, 4, 8, 12, 11, 10, 9, 5, 6, 7]`，共 12 個元素不重不漏。

## **時間與空間複雜度分析**

- **時間複雜度：O(m × n)**
    
    - `m` 為矩陣行數，`n` 為矩陣列數。矩陣中的每一個元素都會被精準訪問並加入 `res` 一次，總時間複雜度為 **O(m × n)**。
        
- **空間複雜度：O(1)**
    
    - 除了儲存答案必須使用的輸出陣列 `res` 之外，僅使用了 `left`、`right`、`top`、`bottom` 四個整數指標，額外空間複雜度為 **O(1)**。
        

## **其他解法**

### **1. 方向向量模擬法（Direction Vector + Visited 矩陣）**

- **思路**：
    
    - 定義四個順時針方向向量 `[(0, 1), (1, 0), (0, -1), (-1, 0)]`，從 `(0, 0)` 出發走 `m × n` 步。每當下一步超出邊界或遇到 `visited[nr][nc] == True` 時，將方向索引 `d = (d + 1) % 4` 順時針右轉。
        
- **時間複雜度**：O(m × n)
    
- **空間複雜度**：O(m × n)（需要額外的 `visited` 布林矩陣；若允許修改原矩陣，可將走過的格子標記為特殊值如 `101` 以省去 `visited` 空間）
    

### **2. Python 奇技淫巧：削蘋果皮法（Pop + Zip 逆時針旋轉）**

- **思路**：
    
    - 每次用 `res += matrix.pop(0)` 拔掉第一行，再用 `list(zip(*matrix))[::-1]` 把剩餘矩陣逆時針旋轉 90 度，重複直到矩陣為空。程式碼極短（僅 4 行），但涉及多次陣列切片與重建，空間與常數時間開銷較高，不建議作為面試主力解。
        
- **時間複雜度**：O(m × n)
    
- **空間複雜度**：O(m × n)
    

## **總結**

- **面試最推薦解法**：**四邊界收縮模擬法（`left, right, top, bottom`）**，時間 `O(m × n)`、額外空間 `O(1)`，不破壞原輸入矩陣，且無需額外 `visited` 陣列。
    
- **最不容易寫錯的寫法**：**方向向量 + `visited` 陣列**，若緊張忘記四邊界收縮的 `break` 條件，用方向向量轉向最穩妥。
    
- **關鍵記憶點**：每走完「向右 + 向下」兩步後，務必檢查 **`if not (left < right and top < bottom): break`**。
    

## **LeetCode 螺旋遍歷與矩陣模擬類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 54**|Spiral Matrix（螺旋矩陣）|**2D `m × n` 矩陣順時針螺旋讀取**|**四邊界收縮法（注意中間 `break` 檢查）**，`O(1)` 空間|
|**LeetCode 59**|Spiral Matrix II（螺旋矩陣 II）|**生成 `1 ~ n²` 順時針螺旋填充方陣**|**四邊界收縮法填入遞增數字 `num += 1`**（方陣無需中間 `break`）|
|**LeetCode 885**|Spiral Matrix III（螺旋矩陣 III）|**從起點 `(rStart, cStart)` 由內向外螺旋走訪網格**|**步長規律 `1, 1, 2, 2, 3, 3...` 搭配方向向量模擬**|
|**LeetCode 2326**|Spiral Matrix IV（螺旋矩陣 IV）|**將單向鏈結串列（Linked List）螺旋填入 `m × n` 矩陣**|**預填 `-1` 後用四邊界收縮法依序填入鏈結串列節點值**|
|**LeetCode 48**|Rotate Image（旋轉圖像）|**2D 方陣原地順時針旋轉 90°**|**主對角線轉置 + 逐行左右翻轉**，`O(1)` 空間|
|**LeetCode 498**|Diagonal Traverse（對角線遍歷）|**2D 矩陣沿右上/左下方向交替對角線遍歷**|**依 `r + c` 的奇偶性決定遍歷方向與邊界轉向規則**|

### **進階變形：若題目改為「逆時針（Counter-Clockwise）螺旋輸出」，該如何修改？**

Python

```
class Solution:
    def spiralOrderCounterClockwise(self, matrix: list[list[int]]) -> list[int]:
        res = []
        left, right = 0, len(matrix[0])
        top, bottom = 0, len(matrix)

        while left < right and top < bottom:
            # 1. 從上到下遍歷左邊界（left 列）
            for i in range(top, bottom):
                res.append(matrix[i][left])
            left += 1

            # 2. 從左到右遍歷下邊界（bottom - 1 行）
            for i in range(left, right):
                res.append(matrix[bottom - 1][i])
            bottom -= 1

            # 關鍵邊界檢查
            if not (left < right and top < bottom):
                break

            # 3. 從下到上遍歷右邊界（right - 1 列）
            for i in range(bottom - 1, top - 1, -1):
                res.append(matrix[i][right - 1])
            right -= 1

            # 4. 從右到左遍歷上邊界（top 行）
            for i in range(right - 1, left - 1, -1):
                res.append(matrix[top][i])
            top += 1

        return res
```

```

<ElicitationsGroup message="繼續整理下一題 Blind 75 / NeetCode 150 筆記：">
  <Elicitation label="LeetCode 73. Set Matrix Zeroes（矩陣置零）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 73. Set Matrix Zeroes（矩陣置零）的完整筆記。"/>
  <Elicitation label="LeetCode 56. Merge Intervals（合併區間）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 56. Merge Intervals（合併區間）的完整筆記。"/>
  <Elicitation label="LeetCode 200. Number of Islands（島嶼數量）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 200. Number of Islands（島嶼數量）的完整筆記。"/>
</ElicitationsGroup>
```