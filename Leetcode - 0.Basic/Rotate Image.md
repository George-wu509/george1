

LeetCode 48 / LintCode 161
给定一个 `n × n` 的二维矩阵 `matrix` 表示一个图像。请你将图像顺时针旋转 90 度。
你必须在**原地**旋转图像，这意味着你需要直接修改输入的二维矩阵。**请不要**使用另一个矩阵来旋转图像。

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
[
    [7, 4, 1],
    [8, 5, 2],
    [9, 6, 3]
]
```

解释： 將 `3 x 3` 矩陣順時針旋轉 90 度後，原本的第一行 `[1, 2, 3]` 變成最後一列，第一列 `[1, 4, 7]` 變成第一行 `[7, 4, 1]`。

**样例 2：** 输入：
```Python
matrix = [
    [5,  1,  9,  11],
    [2,  4,  8,  10],
    [13, 3,  6,  7],
    [15, 14, 12, 16]
]
```

输出：
```Python
[
    [15, 13, 2,  5],
    [14, 3,  4,  1],
    [12, 6,  8,  9],
    [16, 7,  10, 11]
]
```

解释： 外圈與內圈元素分別順時針旋轉 90 度，全程在原矩陣上直接交換數值，不使用額外二維陣列。

```Python
# 解法一：矩陣轉置（Transpose）+ 左右翻轉（Reflect）（最推薦、最不易出錯）
class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        
        # 1. 沿主對角線轉置 (matrix[i][j] <-> matrix[j][i])
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
        # 2. 每一行左右反轉
        for i in range(n):
            matrix[i].reverse()
```

```Python
# 解法二：由外向內逐層四向輪換（Four-Way Layer Rotation）
class Solution:
    def rotate(self, matrix: list[list[int]]) -> None:
        l, r = 0, len(matrix) - 1
        
        while l < r:
            for i in range(r - l):
                top, bottom = l, r
                
                # 保存左上角數值
                top_left = matrix[top][l + i]
                
                # 左下 -> 左上
                matrix[top][l + i] = matrix[bottom - i][l]
                
                # 右下 -> 左下
                matrix[bottom - i][l] = matrix[bottom][r - i]
                
                # 右上 -> 右下
                matrix[bottom][r - i] = matrix[top + i][r]
                
                # 左上 -> 右上
                matrix[top + i][r] = top_left
                
            l += 1
            r -= 1
```

解釋: step1: 取得矩陣邊長 `n = len(matrix)`。 step2: 進行第一步「**主對角線轉置（Transpose）**」：外層迴圈 `i` 從 `0` 到 `n - 1`，內層迴圈 `j` 從 `i + 1` 開始（**只遍歷上三角區域**，避免交換兩次變回原狀），將 `matrix[i][j]` 與 `matrix[j][i]` 互換。 step3: 此時原本的「第 `j` 列」已經變成了「第 `j` 行」，但順序是反的（例如 `[1, 4, 7]` 應該是 `[7, 4, 1]`）。 step4: 進行第二步「**左右翻轉（Horizontal Flip）**」：遍歷每一行 `i`，直接呼叫 `matrix[i].reverse()`（或雙指標左右對調）。 step5: 完成上述兩步後，矩陣即精準完成順時針 90 度原地旋轉，無需回傳值。

解釋: 順時針旋轉 90 度的座標映射規律為：`(row, col) -> (col, n - 1 - row)`。

拆解成兩次線性變換：

1. **轉置**：`(row, col) -> (col, row)`
    
2. **水平鏡像翻轉**：`(col, row) -> (col, n - 1 - row)`
    
    兩者疊加剛好等於順時針旋轉 90 度，且每一步都是簡單的兩兩交換，完全不需要記憶複雜的四向索引偏移量。
    

### **LeetCode 48 / LintCode 161 - Rotate Image（旋轉圖像）**

這道題目是經典的 **二維矩陣原地操作（In-place Matrix Manipulation）** 問題。題目嚴格限制空間複雜度必須為 `O(1)`，考驗如何透過幾何對稱性或分層指標來完成座標變換。

## **解法解析**

這道題最優雅的做法是將「旋轉」拆解為兩個基礎的矩陣翻轉操作：**轉置 + 水平翻轉**。

### **思路**

1. **為什麼 `j` 要從 `i + 1` 開始？**
    
    - 在轉置矩陣時，主對角線（`i == j`）上的元素位置不變。
        
    - 如果 `j` 從 `0` 跑到 `n - 1`，那麼 `(0, 1)` 與 `(1, 0)` 會在 `i=0, j=1` 時交換一次，又在 `i=1, j=0` 時換回來，導致矩陣完全沒變。
        
    - 因此只需遍歷主對角線右上方（`j > i`）即可。
        
2. **四種常見旋轉與翻轉的組合口訣**
    
    - **順時針旋轉 90°** = 先沿主對角線轉置（Transpose） + 再左右翻轉（Reverse rows）
        
    - **逆時針旋轉 90°** = 先沿主對角線轉置（Transpose） + 再上下翻轉（Reverse matrix）
        
    - **旋轉 180°** = 先上下翻轉 + 再左右翻轉
        
3. **代碼執行流程（解法一）**
    
    - 第一階段雙層迴圈完成行列互換。
        
    - 第二階段單層迴圈對每一列執行原地反轉 `reverse()`。
        

## **具體舉例**

以 **样例 1** 的 `3 x 3` 矩陣為例，追蹤每一步的矩陣狀態變化：

|**階段**|**操作內容**|**矩陣狀態**|
|---|---|---|
|**初始狀態**|原始輸入 `matrix`|`[[1, 2, 3],`<br><br>  <br><br>`[4, 5, 6],`<br><br>  <br><br>`[7, 8, 9]]`|
|**步驟 1：轉置**|交換 `(0,1)↔(1,0)`：2 與 4 互換<br><br>  <br><br>交換 `(0,2)↔(2,0)`：3 與 7 互換<br><br>  <br><br>交換 `(1,2)↔(2,1)`：6 與 8 互換|`[[1, 4, 7],`<br><br>  <br><br>`[2, 5, 8],`<br><br>  <br><br>`[3, 6, 9]]`|
|**步驟 2：左右翻轉**|第 0 行 `[1, 4, 7]` 反轉 → `[7, 4, 1]`<br><br>  <br><br>第 1 行 `[2, 5, 8]` 反轉 → `[8, 5, 2]`<br><br>  <br><br>第 2 行 `[3, 6, 9]` 反轉 → `[9, 6, 3]`|`[[7, 4, 1],`<br><br>  <br><br>`[8, 5, 2],`<br><br>  <br><br>`[9, 6, 3]]`|

最終矩陣直接原地更新為 `[[7, 4, 1], [8, 5, 2], [9, 6, 3]]`。

## **時間與空間複雜度分析**

- **時間複雜度：O(n²)**
    
    - `n × n` 矩陣共有 `n²` 個元素。轉置步驟訪問約 `n(n-1)/2` 對元素，左右翻轉訪問 `n²/2` 對元素，每個元素最多被讀寫常數次，總時間複雜度為 **O(n²)**（已達理論最優，因為至少需要訪問每個元素一次）。
        
- **空間複雜度：O(1)**
    
    - 所有交換操作皆在原矩陣 `matrix` 內部完成，僅使用常數個臨時變數，空間複雜度為 **O(1)**。
        

## **其他解法**

### **1. 由外向內逐層四向輪換（Layer-by-Layer 4-Way Swap）**

- **思路**：
    
    - 維護左右邊界 `l` 與 `r`，將矩陣視為由外到內的同心正方形環（每次處理一圈，共 `n // 2` 圈）。對於當前圈的每個偏移量 `i`，用一個臨時變數 `top_left` 逆序覆蓋四條邊對應的四個位置。
        
- **時間複雜度**：O(n²)
    
- **空間複雜度**：O(1)（每個元素只移動一次，寫入次數比轉置法少一半，但索引邊界較容易寫錯）
    

### **2. 輔助矩陣法（不符合題目原地要求，僅作思路對照）**

- **思路**：
    
    - 建立一個新的 `n × n` 矩陣 `res`，直接套用座標公式 `res[c][n - 1 - r] = matrix[r][c]`，最後複製回 `matrix`。
        
- **時間複雜度**：O(n²)
    
- **空間複雜度**：O(n²)（違反題目 In-place 限制）
    

## **總結**

- **面試最推薦解法**：**主對角線轉置 + 逐行左右翻轉**，邏輯極度簡潔、零邊界 Bug 風險，且時間 `O(n²)`、空間 `O(1)` 完全符合最佳標準。
    
- **效能微調解法**：**四向分層輪換**，每個元素只需賦值 1 次（轉置法需交換 2 次），適合對寫入次數敏感的場景。
    
- **核心公式**：順時針 90° 的本質就是 **`Transpose + Reverse Each Row`**。
    

## **LeetCode 矩陣旋轉與原地變換類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 48**|Rotate Image（旋轉圖像）|**2D 方陣原地順時針旋轉 90°**|**主對角線轉置 + 逐行左右翻轉**（或四向分層輪換），`O(1)` 空間|
|**LeetCode 54**|Spiral Matrix（螺旋矩陣）|**2D 矩陣順時針螺旋遍歷**|**維護 `top, bottom, left, right` 四個邊界指標逐步收縮**|
|**LeetCode 59**|Spiral Matrix II（螺旋矩陣 II）|**生成 `1 ~ n²` 螺旋填充方陣**|**同上四邊界收縮模擬填數**，`O(n²)` 時間|
|**LeetCode 73**|Set Matrix Zeroes（矩陣置零）|**2D 矩陣原地行列清零**|**借用矩陣第 0 行與第 0 列作為狀態標記位**，`O(1)` 空間|
|**LeetCode 867**|Transpose Matrix（轉置矩陣）|**一般 `m × n` 矩陣轉置（非方陣）**|**建立 `n × m` 新矩陣映射 `res[c][r] = matrix[r][c]`**|
|**LeetCode 1886**|Determine Whether Matrix Can Be Obtained By Rotation|**判斷方陣旋轉 0°/90°/180°/270° 後是否與目標相同**|**最多呼叫 4 次「轉置 + 左右翻轉」並比對矩陣**|

### **進階變形：若題目改為「逆時針（Counter-Clockwise）原地旋轉 90 度」，該如何修改？**

Python

```
class Solution:
    def rotateCounterClockwise(self, matrix: list[list[int]]) -> None:
        n = len(matrix)
        
        # 1. 沿主對角線轉置 (matrix[i][j] <-> matrix[j][i])
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
                
        # 2. 上下翻轉整個矩陣的行順序（或沿副對角線翻轉）
        matrix.reverse()
```

```

<ElicitationsGroup message="繼續整理下一題 Blind 75 / NeetCode 150 筆記：">
  <Elicitation label="LeetCode 54. Spiral Matrix（螺旋矩陣）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 54. Spiral Matrix（螺旋矩陣）的完整筆記。"/>
  <Elicitation label="LeetCode 73. Set Matrix Zeroes（矩陣置零）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 73. Set Matrix Zeroes（矩陣置零）的完整筆記。"/>
  <Elicitation label="LeetCode 49. Group Anagrams（字母異位詞分組）" query="請按照我的 Obsidian 筆記模板格式，生成 LeetCode 49. Group Anagrams（字母異位詞分組）的完整筆記。"/>
</ElicitationsGroup>
```