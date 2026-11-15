
x 的 n 次方 Pow(x, n)

LeetCode 50 / LintCode 428 實作 `pow(x, n)`，即計算 `x` 的整數 `n` 次冪函數（即 $x^n$）。**提示條件：**

- `-100.0 < x < 100.0`
- `-2^31 <= n <= 2^31 - 1`
- `n` 是一個整數
- 要麼 `x` 不為零，要麼 `n > 0`
- `-10^4 <= x^n <= 10^4`

**样例 1：** 输入：
```Python
x = 2.00000
n = 10
```

输出：
```Python
1024.00000
```

解释： 2.0 的 10 次方等於 1024.0

**样例 2：** 输入：
```Python
x = 2.10000
n = 3
```

输出：
```Python
9.26100
```

解释： 2.1 × 2.1 × 2.1 = 9.261

**样例 3：** 输入：
```Python
x = 2.00000
n = -2
```

输出：
```Python
0.25000
```

解释： 2^(-2) = 1 / 2^2 = 1 / 4 = 0.25
```Python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        def helper(x: float, n: int) -> float:
            if x == 0:
                return 0
            if n == 0:
                return 1

            # 遞迴計算一半的次方，將問題規模折半
            res = helper(x * x, n // 2)
            # 若 n 為奇數，需額外多乘一次底數 x
            return x * res if n % 2 else res

        res = helper(x, abs(n))
        return res if n >= 0 else 1 / res
```


```Python
class Solution:
    def myPow(self, x: float, n: int) -> float:
        if x == 0:
            return 0.0

        # 處理負指數：x^(-n) = (1/x)^n
        if n < 0:
            x = 1 / x
            n = -n

        res = 1.0
        current_product = x

        # 迭代快速冪（從二進位最低位往最高位掃描）
        while n > 0:
            # 若當前二進位最低位為 1（即 n 為奇數），將當前的底數乘入結果
            if n % 2 == 1:
                res *= current_product
            # 底數平方，指數右移一位（除以 2）
            current_product *= current_product
            n //= 2

        return res
```

解釋: step1: 如果直接用迴圈乘 `n` 次，當 `n = 2^31 - 1`（約 21 億）時會超時（TLE），因此必須利用 **快速冪（Binary Exponentiation）** 將指數每次折半 step2: 若 `n` 為負數，利用數學性質 $x^{-n} = \frac{1}{x^n}$（或 $(\frac{1}{x})^n$）將問題統一轉化為正整數次冪處理 step3: 遞迴版本中，每次將底數平方 `x * x`、指數除以二 `n // 2`：

- 若 `n` 為偶數：$x^n = (x^2)^{n/2}$
    
- 若 `n` 為奇數：$x^n = x \cdot (x^2)^{\lfloor n/2 \rfloor}$ step4: 假設 `x = 2`, `n = 10`（遞迴過程）：
    
- `helper(2, 10)` -> `n` 為偶數，呼叫 `helper(4, 5)`
    
- `helper(4, 5)` -> `n` 為奇數，返回 `4 * helper(16, 2)`
    
- `helper(16, 2)` -> `n` 為偶數，呼叫 `helper(256, 1)`
    
- `helper(256, 1)` -> `n` 為奇數，返回 `256 * helper(65536, 0)`
    
- `helper(65536, 0)` -> `n == 0`，回傳 `1` step5: 層層回傳：`1 -> 256 -> 256 -> 4 * 256 = 1024 -> 1024`
    

x = 2.0, n = -2 -> res = 0.25

1. 先取絕對值計算 helper(2.0, 2) = 4.0
    
2. 因 n < 0，回傳 1 / 4.0 = 0.25
    

解釋: 核心思想是 **分治法（Divide and Conquer）與二進位拆解**。 例如計算 $x^{10}$，$10$ 的二進位是 `1010`（即 $8 + 2$），所以 $x^{10} = x^8 \times x^2$，只需要計算 `x -> x^2 -> x^4 -> x^8` 共 4 步即可，大幅將時間複雜度從 $O(n)$ 降至 $O(\log n)$。

### **LeetCode 50 / LintCode 428 - Pow(x, n)（x 的 n 次方）**

這道題目是 **NeetCode 150（Math & Geometry 分類）** 的必考經典題，也是演算法中 **快速冪（Binary Exponentiation）** 的標準模板題，廣泛應用於大數模冪運算、矩陣快速冪（如 $O(\log n)$ 求斐波那契數列）等進階題型。

## **解法解析**

這道題如果使用線性連乘 $O(n)$ 必然超時，必須透過 **每次將指數減半** 來達成對數級時間複雜度。

### **思路**

1. **負指數轉換與整數溢位細節**
    
    - 當 $n < 0$ 時，根據指數律 $x^{-n} = \frac{1}{x^n}$。
        
    - **面試細節提醒（針對 C++ / Java）**：題目給定 $n$ 的範圍是 $[-2^{31}, 2^{31} - 1]$（即 `[-2147483648, 2147483647]`）。在 32 位元有號整數中，若直接對 `n = -2147483648` 執行 `n = -n` 會導致正數溢位！在 Python 中整數沒有上限不會溢位，但若在靜態語言面試時，需先將 `n` 轉為 `long` 型別再取負號。
        
2. **分治拆解（遞迴思路）**
    
    - 想知道 $x^n$，只要先算出 $y = x^{\lfloor n/2 \rfloor}$：
        
        - 如果 $n$ 是偶數：$x^n = y \times y$
            
        - 如果 $n$ 是奇數：$x^n = y \times y \times x$
            
    - 這樣寫能避免重複呼叫兩次 `helper(x, n // 2)`（若寫成 `helper(x, n // 2) * helper(x, n // 2)` 會退化成 $O(n)$ 時間複雜度，是常見錯誤！）。
        
3. **二進位視角（迭代思路）**
    
    - 任何正整數 $n$ 都可以寫成二進位形式。以 $n = 13$（二進位 `1101`）為例：
        
        - $x^{13} = x^{8 + 4 + 1} = x^8 \times x^4 \times x^1$
            
    - 在迴圈中維護一個不斷自乘的變數 `current_product`（依序代表 $x^1, x^2, x^4, x^8, \dots$），只要檢查 `n` 的二進位最低位是否為 `1`（`n & 1` 或 `n % 2 == 1`），就把對應的 `current_product` 乘入答案 `res` 中。
        

## **具體舉例**

假設 `x = 2.0`, `n = 13`（二進位為 `1101`），使用 **迭代快速冪** 的變數變化如下：

|**步驟**|**n（十進位 / 二進位）**|**n % 2 == 1（最低位為 1？）**|**res（更新後）**|**current_product（下輪底數）**|
|---|---|---|---|---|
|初始化|`13` (`1101`)|-|`1.0`|`2.0` ($x^1$)|
|第 1 輪|`13` (`1101`)|是（乘入 $x^1$）|`1.0 * 2.0 = 2.0`|`4.0` ($x^2$)|
|第 2 輪|`6` (`110`)|否（跳過 $x^2$）|`2.0`|`16.0` ($x^4$)|
|第 3 輪|`3` (`11`)|是（乘入 $x^4$）|`2.0 * 16.0 = 32.0`|`256.0` ($x^8$)|
|第 4 輪|`1` (`1`)|是（乘入 $x^8$）|`32.0 * 256.0 = 8192.0`|`65536.0` ($x^{16}$)|

迴圈在 `n = 0` 時結束，最終返回 `8192.0`（即 $2^{13}$）。

## **時間與空間複雜度分析**

- **時間複雜度：O(log |n|)**
    
    - 無論是遞迴還是迭代寫法，每次操作都會將指數 `n` 除以 2（右移一位），總共只會執行 $\lfloor \log_2(\vert{}n\vert{}) \rfloor + 1$ 次（最多約 31 次），因此時間複雜度為 **O(log |n|)**。
        
- **空間複雜度：**
    
    - **迭代快速冪：O(1)** — 僅使用常數個變數（`res`、`current_product`、`n`），沒有遞迴呼叫堆疊開銷，為空間最優解。
        
    - **遞迴快速冪：O(log |n|)** — 遞迴深度為 $\log_2(\vert{}n\vert{})$，系統呼叫堆疊（Call Stack）需要占用 $O(\log \vert{}n\vert{})$ 空間。
        

## **其他解法**

### **1. 暴力連乘法（Brute Force — 會超時 TLE）**

- **思路**：
    
    - 將 `1.0` 連續乘上 `x` 共 `|n|` 次，若 `n < 0` 則回傳倒數。
        
- **缺點**：
    
    - 當 `n = 2147483647` 時需要執行超過 20 億次乘法，時間複雜度為 $O(\vert{}n\vert{})$，在 LeetCode 會直接超時（Time Limit Exceeded）。
        
- **時間複雜度**：O(|n|)
    
- **空間複雜度**：O(1)
    

## **總結**

- **面試最推薦解法**：**迭代快速冪（Iterative Binary Exponentiation）**，時間 `O(log |n|)`，空間 `O(1)`，無遞迴堆疊開銷。
    
- **最簡潔直覺解法**：**遞迴快速冪（Recursive Binary Exponentiation）**，程式碼極短且邏輯清晰，面試白板題非常容易一次寫對。
    
- **三大關鍵考點**：
    
    1. 指數折半將時間從 $O(n)$ 降至 $O(\log n)$。
        
    2. 負指數 $n < 0$ 轉化為倒數計算。
        
    3. 靜態語言中 `n = -2^31` 取負號時的整數溢位邊界。
        

## **Blind 75 / NeetCode 150 數學、分治與快速冪類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 50**|Pow(x, n)（x 的 n 次方）|**Math / Divide & Conquer，快速冪運算**|**將指數 `n` 每次折半 `n // 2`，底數平方 `x * x`，時間 $O(\log n)$**|
|**LeetCode 69**|Sqrt(x)（x 的平方根）|**Math / Binary Search，整數平方根**|**在 `[0, x]` 區間使用二分搜尋尋找滿足 $mid^2 \le x$ 的最大整數**|
|**LeetCode 367**|Valid Perfect Square（有效的完全平方數）|**Math / Binary Search，判斷是否為完全平方數**|**二分搜尋 $mid^2 == num$，或利用數學性質 $1 + 3 + 5 + \dots + (2k-1) = k^2$**|
|**LeetCode 372**|Super Pow（超級次方）|**Math / Divide & Conquer，大數陣列指數模冪**|**結合快速冪與模運算性質：$(a \cdot b) \bmod k = [(a \bmod k)(b \bmod k)] \bmod k$**|
|**LeetCode 7**|Reverse Integer（整數反轉）|**Math，32 位元有號整數反轉與溢位判斷**|**取餘數 `% 10` + 整除 `/ 10`，推進前檢查 `MAX // 10` 邊界**|
|**LeetCode 202**|Happy Number（快樂數）|**Math / Two Pointers，平方和循環檢測**|**快慢指標（Floyd 判圈演算法）或 HashSet 偵測是否進入無限循環**|
|**LeetCode 43**|Multiply Strings（字串相乘）|**Math / String，大數乘法模擬**|**開長度 `m + n` 陣列模擬直式乘法，`num1[i] * num2[j]` 落在 `i + j` 與 `i + j + 1`**|
|**LeetCode 70**|Climbing Stairs（爬樓梯 — 進階解）|**Math / Matrix Exponentiation，矩陣快速冪**|**將線性遞推轉為 $2 \times 2$ 轉移矩陣，套用快速冪模板將時間從 $O(n)$ 降至 $O(\log n)$**|

### **延伸變形題：LeetCode 372 - Super Pow（超級次方）**

你的任務是計算 $a^b \bmod 1337$ 的值，其中 `a` 是一個正整數，`b` 是一個非常大的正整數且以陣列形式給出（例如 $b = [1, 0]$ 代表 $10$）。

Python

```
class Solution:
    def superPow(self, a: int, b: list[int]) -> int:
        MOD = 1337

        # 快速冪輔助函式：計算 (x^k) % MOD
        def my_pow(x: int, k: int) -> int:
            x %= MOD
            res = 1
            while k > 0:
                if k % 2 == 1:
                    res = (res * x) % MOD
                x = (x * x) % MOD
                k //= 2
            return res

        # 遞迴拆解指數陣列：例如 a^[1, 2, 3] = (a^[1, 2])^10 * a^3
        if not b:
            return 1

        last_digit = b.pop()
        part1 = my_pow(a, last_digit)
        part2 = my_pow(self.superPow(a, b), 10)

        return (part1 * part2) % MOD
```