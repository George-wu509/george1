
LeetCode 2013 設計一個演算法，可以接收來自二維平面上的資料流點座標，並計算能與給定查詢點組成 **軸對齊正方形（Axis-Aligned Squares）** 的方法總數。

**實作 `DetectSquares` 類別：**

- `DetectSquares()`：使用空資料結構初始化物件。
- `void add(int[] point)`：將一個新點 `point = [x, y]` 加入資料結構中。**允許重複的點**，且重複加入的點應被視為不同的點（各自獨立計算）。
- `int count(int[] point)`：給定查詢點 `point = [x, y]`，回傳能與該點組成 **軸對齊正方形** 且 **面積大於 0**（邊長不為 0）的方法總數。

**样例 1：** 输入：
```Python
["DetectSquares", "add", "add", "add", "count", "count", "add", "count"]
[[], [[3, 10]], [[11, 2]], [[3, 2]], [[11, 10]], [[14, 8]], [[11, 2]], [[11, 10]]]
```

输出：
```Python
[null, null, null, null, 1, 0, null, 2]
```

解释： detectSquares = DetectSquares() detectSquares.add([3, 10]) detectSquares.add([11, 2]) detectSquares.add([3, 2]) detectSquares.count([11, 10]) -> 回傳 1（可與 [3, 10]、[11, 2]、[3, 2] 組成邊長為 8 的正方形） detectSquares.count([14, 8]) -> 回傳 0（無法組成任何正方形） detectSquares.add([11, 2]) -> 允許重複加入相同座標，視為不同的點 detectSquares.count([11, 10]) -> 回傳 2（因為 [11, 2] 有 2 個，可組成 2 種組合）

**样例 2：** 输入：
```Python
["DetectSquares", "add", "add", "add", "count"]
[[], [[0, 0]], [[0, 1]], [[1, 0]], [[0, 0]]]
```

输出：
```Python
[null, null, null, null, 0]
```

解释： 雖然 [0, 0] 被加入過一次，但查詢點也是 [0, 0]，若選用同座標點會導致邊長為 0（面積為 0），不符合正方形「面積必須大於 0」的定義，故回傳 0。

```Python
from collections import defaultdict

class DetectSquares:

    def __init__(self):
        self.pts_count = defaultdict(int)
        self.pts = []

    def add(self, point: list[int]) -> None:
        self.pts_count[tuple(point)] += 1
        self.pts.append(point)

    def count(self, point: list[int]) -> int:
        res = 0
        qx, qy = point
        for px, py in self.pts:
            if abs(px - qx) != abs(py - qy) or px == qx:
                continue
            res += self.pts_count[(qx, py)] * self.pts_count[(px, qy)]
        return res
```


```Python
from collections import defaultdict

class DetectSquares:

    def __init__(self):
        self.pts_count = defaultdict(int)

    def add(self, point: list[int]) -> None:
        self.pts_count[tuple(point)] += 1

    def count(self, point: list[int]) -> int:
        res = 0
        qx, qy = point
        for (px, py), cnt in list(self.pts_count.items()):
            if abs(px - qx) == abs(py - qy) and px != qx:
                res += cnt * self.pts_count[(qx, py)] * self.pts_count[(px, qy)]
        return res
```

解釋: step1: 在 `__init__` 中建立雜湊表 `pts_count = defaultdict(int)`，用來記錄每個座標 `(x, y)` 出現的次數（處理重複點） step2: 每次呼叫 `add(point)` 時，將座標轉為不可變的元組 `tuple(point)`，並將 `pts_count[tuple(point)]` 次數加 `1` step3: 呼叫 `count(point)` 時，設查詢點為 `(qx, qy)`，遍歷所有已加入的點 `(px, py)`，把它當作正方形的 **「對角線頂點」** step4: 檢查幾何條件：若要成為軸對齊正方形的對角點，X 軸距離與 Y 軸距離必須相等（`abs(px - qx) == abs(py - qy)`），且不能重合（`px != qx`，確保邊長大於 0） step5: 一旦確定了查詢點 `(qx, qy)` 與對角點 `(px, py)`，另外兩個頂點座標就唯一確定為 `(qx, py)` 與 `(px, qy)` step6: 利用乘法原理，將三個頂點各自出現的次數相乘並累加至 `res`，遍歷結束後回傳 `res`

query = [11, 10] (已加入 [3, 10], [11, 2] 兩次, [3, 2]) -> result = 2

1. 檢查 `(px, py) = (3, 10)`：`abs(3 - 11) = 8` != `abs(10 - 10) = 0`（跳過）
    
2. 檢查 `(px, py) = (11, 2)`：`abs(11 - 11) = 0` != `abs(2 - 10) = 8`（跳過）
    
3. 檢查 `(px, py) = (3, 2)`：`abs(3 - 11) = 8 == abs(2 - 10) = 8` 且 `3 != 11`（合法對角點！）
    
4. 查詢另外兩角數量：`pts_count[(11, 2)] = 2`，`pts_count[(3, 10)] = 1`，累加 `1 * 2 * 1 = 2` -> 回傳 `2`
    

解釋: 這題的核心技巧在於 **「枚舉對角線頂點（Diagonal Enumeration）」**。 在二維平面上，如果我們嘗試同時枚舉正方形的另外三個頂點，時間複雜度會高達 $O(N^3)$。 但因為題目限定為 **軸對齊正方形**，只要固定了查詢點 `(qx, qy)` 和它的 **對角頂點 `(px, py)`**，另外兩個頂點 `(qx, py)` 與 `(px, qy)` 的位置是唯一固定的！因此我們只需花 $O(N)$ 時間枚舉對角點，再用 $O(1)$ 時間從 Hash Map 查詢另外兩個頂點的數量即可。

### **LeetCode 2013 - Detect Squares（偵測正方形）**

這道題目是 NeetCode 150 中 **Math & Geometry** 與 **Hash Map / Design** 的經典應用題。它考驗如何利用幾何性質簡化搜尋空間，並正確處理「重複座標點」的排列組合計數。

## **解法解析**

這道題可以轉化為 **「固定對角線兩端點，透過雜湊表 O(1) 查詢其餘兩角」** 來求解。

### **思路**

1. **軸對齊正方形的幾何特性**
    
    - 給定查詢點 $Q = (qx, qy)$，假設平面上存在另一點 $P = (px, py)$ 與 $Q$ 構成正方形的 **對角線**。
        
    - 因為正方形四邊與座標軸平行，對角線兩端點必須滿足：
        
        - **邊長相等**：$\vert{}px - qx\vert{} = \vert{}py - qy\vert{}$
            
        - **面積大於 0**：$px \neq qx$（若 $px = qx$，則邊長為 0，四個點會縮成同一點）
            
2. **唯一確定其餘兩頂點**
    
    - 當對角線端點 $Q(qx, qy)$ 與 $P(px, py)$ 確定後，正方形另外兩個頂點必然是：
        
        - 頂點 $A = (qx, py)$（與 $Q$ 同 X 座標，與 $P$ 同 Y 座標）
            
        - 頂點 $B = (px, qy)$（與 $P$ 同 X 座標，與 $Q$ 同 Y 座標）
            
3. **乘法原理處理重複點**
    
    - 題目允許在同一個座標重複加入多個點。
        
    - 假設對角點 $P$ 出現了 $C_P$ 次，頂點 $A$ 出現了 $C_A$ 次，頂點 $B$ 出現了 $C_B$ 次，那麼以 $P$ 為對角點所能組成的正方形總數就是：
        
        $$\text{ways} = C_P \times C_A \times C_B$$
        
    - 注意：若使用 `defaultdict(int)`，直接查詢不存在的鍵值會自動回傳 `0`，且會順便在字典中插入該鍵值（值為 0）。若在遍歷 `pts_count.items()` 時直接存取 `self.pts_count[(qx, py)]`，會引發 `RuntimeError: dictionary changed size during iteration`，因此務必用 `list(self.pts_count.items())` 或改用 `.get((qx, py), 0)` 避免字典大小在迭代中改變。
        

## **具體舉例**

假設依序執行 `add([3, 10])`、`add([11, 2])`、`add([3, 2])`、`add([11, 2])`，此時 `pts_count` 狀態為：

- `(3, 10): 1`
    
- `(11, 2): 2`
    
- `(3, 2): 1`
    

現在呼叫 `count([11, 10])`，即 $Q = (11, 10)$，遍歷所有不重複點 $P = (px, py)$ 的過程如下：

|**候選對角點 P(px,py)**|**出現次數 CP​**|**檢驗 ∣px−11∣==∣py−10∣ 且 px=11**|**推導其餘兩角 A(11,py) 與 B(px,10)**|**查詢數量 CA​×CB​**|**本輪新增正方形數**|
|---|---|---|---|---|---|
|`(3, 10)`|`1`|$\vert{}3 - 11\vert{} = 8 \neq \vert{}10 - 10\vert{} = 0$（不符）|-|-|`0`|
|`(11, 2)`|`2`|$\vert{}11 - 11\vert{} = 0 \neq \vert{}2 - 10\vert{} = 8$（不符）|-|-|`0`|
|`(3, 2)`|`1`|$\vert{}3 - 11\vert{} = 8 == \vert{}2 - 10\vert{} = 8$（**符合！**）|$A(11, 2)$ 與 $B(3, 10)$|`pts_count[(11, 2)] = 2`<br><br>  <br><br>`pts_count[(3, 10)] = 1`|$1 \times 2 \times 1 = \mathbf{2}$|

全部遍歷完成，累加結果回傳 **`2`**。

## **時間與空間複雜度分析**

假設總共呼叫了 $N$ 次 `add` 操作：

- **時間複雜度：**
    
    - **`add` 操作：O(1)**，僅需進行雜湊表更新與陣列附加操作。
        
    - **`count` 操作：O(N)**，最壞情況下需要遍歷目前儲存的所有 $N$ 個點（若使用不重複點雜湊表則為 $O(U)$，其中 $U \le N$ 為不重複座標數量），每次檢查與查詢另外兩角皆為 $O(1)$。
        
- **空間複雜度：O(N)**
    
    - 需要雜湊表 `pts_count`（與列表 `pts`）來儲存所有加入的點座標，最壞情況下所有點座標皆不重複，空間複雜度為 **O(N)**。
        

## **其他解法**

### **1. 依 X 座標分組的雙層雜湊表（Nested Hash Map by X）**

- **思路**：
    
    - 使用 `x_to_ys = defaultdict(Counter)`，將所有點依照 `x` 座標分組（記錄每個 `x` 上有哪些 `y` 以及出現次數）。
        
    - 查詢 `(qx, qy)` 時，**只遍歷與查詢點擁有相同 X 座標的點 `(qx, py)`**（即垂直邊的另一端點），計算邊長 `side = abs(py - qy)`。
        
    - 再直接檢查左邊 `(qx - side)` 與右邊 `(qx + side)` 是否同時存在 `qy` 與 `py`。
        
- **時間複雜度**：`add` 為 $O(1)$，`count` 為 **O(N_x)**（$N_x$ 為同一條垂直線 `x = qx` 上的點數，通常遠小於總點數 $N$）。
    
- **空間複雜度**：O(N)
    

### **2. 暴力三層迴圈枚舉（Brute Force）**

- **思路**：
    
    - 針對查詢點，使用三層迴圈從所有點中任選三個點，檢查這四個點是否能構成軸對齊正方形。
        
- **時間複雜度**：**O(N³)**（在資料量大時必定超時 TLE）
    
- **空間複雜度**：O(N)
    

## **總結**

- **面試最推薦解法**：**枚舉對角線頂點 + `defaultdict(int)` 計數**，程式碼僅 15 行左右，邏輯極度清晰，無需分左右兩側討論正負號，時間複雜度 `add` 為 $O(1)$、`count` 為 $O(N)$，空間為 $O(N)$。
    
- **核心避坑重點**：
    
    1. 必須排除 `px == qx` 的情況，避免面積為 0 的退化正方形被計入。
        
    2. 使用 Python `defaultdict` 迭代時，切勿在迴圈內直接對同一個字典查詢不存在的 key（會新增 key 導致迭代報錯），應使用 `.get(key, 0)` 或另外維護一個 `self.pts` 列表來遍歷。
        

## **Blind 75 & NeetCode 150 類似題型整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 2013**|Detect Squares（偵測正方形）|**二維平面資料流計數軸對齊正方形**|**枚舉對角線頂點 + Hash Map 計數**|
|**LeetCode 939**|Minimum Area Rectangle（最小面積矩形）|**給定二維點集，求軸對齊矩形的最小面積**|**枚舉對角線兩點 `(x1, y1)` 與 `(x2, y2)`，用 Set 查 `(x1, y2)` 與 `(x2, y1)`**（思維與本題幾乎一致）|
|**LeetCode 593**|Valid Square（有效的正方形）|**給定四個二維點，判斷能否組成正方形（可傾斜）**|**計算 4 點兩兩之間的 6 條距離，應有 4 條相等短邊與 2 條相等對角線**|
|**LeetCode 149**|Max Points on a Line（直線上最多的點數）|**二維平面求共線的最大點數**|**固定一點，用 Hash Map 統計其餘點與該點的最簡分數斜率 `(dx, dy)`**|
|**LeetCode 48**|Rotate Image（旋轉圖像）|**原地將 $N \times N$ 二維矩陣順時針旋轉 90 度**|**先沿主對角線轉置（Transpose），再左右翻轉每一列**|
|**LeetCode 54**|Spiral Matrix（螺旋矩陣）|**按順時針螺旋順序遍歷二維矩陣**|**維護上下左右四個邊界指標逐步內縮**|
|**LeetCode 202**|Happy Number（快樂數）|**數學模擬與循環偵測**|**Hash Set 記錄出現過的數字** 或 **快慢指標 O(1) 空間**|
|**LeetCode 43**|Multiply Strings（字串相乘）|**模擬大整數直式乘法運算**|**利用索引 `i + j` 與 `i + j + 1` 累加每位乘積與進位**|

### **進階延伸：依 X 座標分組優化寫法（減少無效遍歷）**

如果面試官追問：「**當平面上的點非常多，但大部分點都不在查詢點的垂直線上時，如何加快 `count` 的查詢速度？**」 我們可以只遍歷「與查詢點 X 座標相同」的點來決定正方形的邊長 `side`，大幅減少迴圈次數：

Python

```
from collections import defaultdict, Counter

class DetectSquaresOptimized:

    def __init__(self):
        # x_to_ys[x][y] = 座標 (x, y) 出現的次數
        self.x_to_ys = defaultdict(Counter)

    def add(self, point: list[int]) -> None:
        x, y = point
        self.x_to_ys[x][y] += 1

    def count(self, point: list[int]) -> int:
        qx, qy = point
        if qx not in self.x_to_ys:
            return 0
        
        res = 0
        # 只遍歷與查詢點同在 x = qx 這條垂直線上的所有點 (qx, py)
        for py, cnt_same_x in self.x_to_ys[qx].items():
            if py == qy:
                continue  # 邊長不能為 0
            
            side = abs(py - qy)
            # 檢查左側 (qx - side) 與右側 (qx + side) 是否能構成完整正方形
            for target_x in (qx - side, qx + side):
                if target_x in self.x_to_ys:
                    res += cnt_same_x * self.x_to_ys[target_x][qy] * self.x_to_ys[target_x][py]
                    
        return res
```