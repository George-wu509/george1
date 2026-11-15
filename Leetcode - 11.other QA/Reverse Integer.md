
數反轉 Reverse Integer

LeetCode 7 / LintCode 413 给你一个 32 位的有符号整数 `x` ，返回将 `x` 中的数字部分反转后的结果。 如果反转后整数超过 32 位的有符号整数的范围 `[−2^31, 2^31 − 1]`（即 `[-2147483648, 2147483647]`），就返回 `0`。**假设环境不允许存储 64 位整数（有符号或无符号）。**

**样例 1：** 输入：
```Python
x = 123
```

输出：
```Python
321
```

解释： 123 反轉後為 321，未超出 32 位元有號整數範圍

**样例 2：** 输入：
```Python
x = -123
```

输出：
```Python
-321
```

解释： 保留負號，數字部分反轉為 -321

**样例 3：** 输入：
```Python
x = 120
```

输出：
```Python
21
```

解释： 反轉後前導零自動去除，結果為 21

```Python
class Solution:
    def reverse(self, x: int) -> int:
        INT_MIN, INT_MAX = -2**31, 2**31 - 1  # -2147483648, 2147483647
        res = 0
        sign = -1 if x < 0 else 1
        x = abs(x)

        while x != 0:
            digit = x % 10
            x //= 10
            # 提前檢查是否會溢位
            if res > (INT_MAX - digit) // 10:
                return 0
            res = res * 10 + digit

        res *= sign
        return res if INT_MIN <= res <= INT_MAX else 0
```


```Python
import math

class Solution:
    def reverse(self, x: int) -> int:
        MIN = -2147483648  # -2^31
        MAX = 2147483647   #  2^31 - 1
        res = 0

        while x:
            # Python 中負數取餘數需用 math.fmod，例如 fmod(-123, 10) == -3.0
            digit = int(math.fmod(x, 10))
            # Python 中負數整除需用 int(x / 10) 向零取整，例如 int(-123 / 10) == -12
            x = int(x / 10)

            if res > MAX // 10 or (res == MAX // 10 and digit > MAX % 10):
                return 0
            if res < int(MIN / 10) or (res == int(MIN / 10) and digit < int(math.fmod(MIN, 10))):
                return 0

            res = (res * 10) + digit

        return res
```

解釋: step1: 透過 `x % 10` 取出最右邊的個位數 `digit`，並用 `x // 10`（或向零取整 `int(x / 10)`）把個位數移除 step2: 每次將目前的結果 `res * 10 + digit` 往左推一位並加上新取出的數字 step3: 因為題目限制**不能使用 64 位元整數**，所以在執行 `res * 10 + digit` **之前**，必須先檢查是否會超過 `MAX (2147483647)` 或低於 `MIN (-2147483648)` step4: 當 `x = 123` 時：

- 第 1 輪：`digit = 3`, `x = 12`, `res = 0 * 10 + 3 = 3`
    
- 第 2 輪：`digit = 2`, `x = 1`, `res = 3 * 10 + 2 = 32`
    
- 第 3 輪：`digit = 1`, `x = 0`, `res = 32 * 10 + 1 = 321` step5: `x` 變為 `0` 結束迴圈，回傳 `321`
    

x = 120 -> res = 21

1. 取出 0 -> res = 0
    
2. 取出 2 -> res = 2
    
3. 取出 1 -> res = 21（前導零自然被數學運算過濾掉）
    

解釋: 把整數當成「堆疊（Stack）」來操作： `pop` 操作：`digit = x % 10`，`x /= 10`（從低位彈出數字） `push` 操作：`res = res * 10 + digit`（從高位推入新數字） 關鍵考點在於 **push 之前的溢位邊界判斷**。

### **LeetCode 7 / LintCode 413 - Reverse Integer（整數反轉）**

這道題目是 **NeetCode 150（Math & Geometry 分類）** 的經典基礎題，核心考點不是反轉邏輯本身，而是 **數學取餘數/整除的語言特性** 以及 **32 位元有號整數的溢位（Overflow）邊界檢查**。

## **解法解析**

這道題可以透過 **數學取模（Modulo）與整除（Division）** 逐位拆解並重組數字。

### **思路**

1. **逐位彈出與推入**
    
    - 想要反轉十進位數字，只要不斷對 `10` 取餘數取得最末位 `digit`，再將 `x` 除以 `10` 去掉最末位。
        
    - 同時將累積變數 `res` 乘以 `10` 並加上 `digit`，即可完成數字反轉。
        
2. **注意 Python 的負數取模與整除陷阱（重要！）**
    
    - 在 C++ / Java 中，`-123 % 10 = -3` 且 `-123 / 10 = -12`（向零取整）。
        
    - 在 Python 中，`-123 % 10 = 7` 且 `-123 // 10 = -13`（向下取整），直接寫會出錯！
        
    - **解決方法有兩種**：
        
        1. 先記錄正負號 `sign`，把 `x` 轉成 `abs(x)` 當作正數處理（最推薦、最不易出錯）。
            
        2. 使用 `int(math.fmod(x, 10))` 與 `int(x / 10)` 模擬向零取整。
            
3. **溢位提前判斷（不依賴 64 位元整數）**
    
    - 32 位元有號整數最大值為 `2147483647`（尾數為 `7`），最小值為 `-2147483648`（尾數為 `-8`）。
        
    - 在計算 `res * 10 + digit` 前：
        
        - 若 `res > 214748364`，下一輪乘以 10 必定溢位。
            
        - 若 `res == 214748364`，則 `digit` 不能大於 `7`（負數情況下不能小於 `-8`）。
            

## **具體舉例**

假設 `x = -123`（使用 `math.fmod` 寫法），變數變化如下：

|**步驟**|**x（進入迴圈前）**|**digit**|**溢位檢查**|**res（更新後）**|**x（更新後）**|
|---|---|---|---|---|---|
|初始化|`-123`|-|-|`0`|`-123`|
|第 1 輪|`-123`|`-3`|通過|`-3`|`-12`|
|第 2 輪|`-12`|`-2`|通過|`-32`|`-1`|
|第 3 輪|`-1`|`-1`|通過|`-321`|`0`|

迴圈結束，最後返回 `-321`。

## **時間與空間複雜度分析**

- **時間複雜度：O(log₁₀(x)) ≈ O(1)**
    
    - 迴圈執行的次數取決於整數 `x` 的十進位位數，即 $\lfloor \log_{10}(\vert{}x\vert{}) \rfloor + 1$。
        
    - 由於 32 位元整數最多只有 10 位數（`2147483647`），最多只會迭代 10 次，因此可視為常數時間 **O(1)**。
        
- **空間複雜度：O(1)**
    
    - 只使用了 `res`、`digit`、`MIN`、`MAX` 等常數個額外變數，沒有使用額外陣列或字串空間。
        

## **其他解法**

### **1. 字串反轉法（不符合進階限制，但面試可先提及）**

- **思路**：
    
    - 將整數轉成字串，去掉負號後用切片 `[::-1]` 反轉，再轉回整數並檢查是否落在 `[-2^31, 2^31 - 1]` 之間。
        
- **缺點**：
    
    - 轉換後的數字在檢查邊界前可能已經超過 32 位元整數上限（在 C++/Java 會直接拋出異常或溢位），違反題目「假設環境不允許儲存 64 位元整數」的要求。
        
- **時間複雜度**：O(log₁₀(x))
    
- **空間複雜度**：O(log₁₀(x))（需額外字串空間）
    

## **總結**

- **面試最推薦解法**：**先提取正負號 `sign` + `abs(x)` 的數學取模法**，時間 `O(1)`，空間 `O(1)`，既符合 32 位元限制，又能完美避開 Python 負數 `//` 與 `%` 的陷阱。
    
- **NeetCode 標準寫法**：使用 `math.fmod(x, 10)` 與 `int(x / 10)` 直接處理正負數，適合展示對語言底層取整行為的熟悉度。
    

## **Blind 75 / NeetCode 150 數學與位元運算類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 7**|Reverse Integer（整數反轉）|**Math，32 位元有號整數反轉與溢位判斷**|**取餘數 `% 10` + 整除 `/ 10`，推進前檢查 `MAX // 10` 邊界**|
|**LeetCode 9**|Palindrome Number（回文數）|**Math，判斷整數是否為回文（不轉字串）**|**只反轉後半段數字 `x <= reverted`，比較 `x == reverted` 或 `x == reverted // 10`**|
|**LeetCode 8**|String to Integer (atoi)|**String / Math，字串轉 32 位元整數與截斷**|**處理前導空白、正負號，逐位累加並在 `res > MAX // 10` 時提早截斷回傳**|
|**LeetCode 66**|Plus One（加一）|**Array / Math，陣列模擬大數加一**|**從尾端遍歷，遇 `9` 變 `0` 繼續進位，全為 `9` 則在最前面補 `1`**|
|**LeetCode 50**|Pow(x, n)（快速冪）|**Math / Divide & Conquer，計算 $x^n$**|**快速冪（Binary Exponentiation），將時間從 $O(n)$ 降至 $O(\log n)$**|
|**LeetCode 202**|Happy Number（快樂數）|**Math / Two Pointers，平方和循環檢測**|**快慢指標（Floyd 判圈演算法）或 HashSet 偵測是否進入無限循環**|
|**LeetCode 43**|Multiply Strings（字串相乘）|**Math / String，大數乘法模擬**|**開長度 `m + n` 陣列模擬直式乘法，`num1[i] * num2[j]` 落在 `p1, p2` 位置**|
|**LeetCode 190**|Reverse Bits（顛倒二進位制位）|**Bit Manipulation，32 位元無號整數二進位反轉**|迴圈 32 次，每次 `(res << 1)|
|**LeetCode 191**|Number of 1 Bits（位1的個數）|**Bit Manipulation，計算漢明重量**|**Brian Kernighan 演算法：`n = n & (n - 1)` 每次消除最低位的 `1`**|
|**LeetCode 371**|Sum of Two Integers（兩整數之和）|**Bit Manipulation，不使用 `+` `-` 算加法**|**異或 `a ^ b` 算無進位和，與運算 `(a & b) << 1` 算進位（Python 需加 `0xffffffff` 遮罩）**|

### **延伸變形題：LeetCode 9 - Palindrome Number（回文數）**

給你一個整數 `x` ，如果 `x` 是一個回文整數，返回 `true` ；否則，返回 `false` 。（進階：不將整數轉為字串，且避免整個數字反轉可能造成的溢位）

Python

```
class Solution:
    def isPalindrome(self, x: int) -> bool:
        # 負數必定不是回文；結尾為 0 的正數（除了 0 本身）也必定不是回文
        if x < 0 or (x % 10 == 0 and x != 0):
            return False

        reverted_half = 0
        # 只需要反轉「一半」的位數，即可完全避免 32 位元整數溢位問題
        while x > reverted_half:
            reverted_half = reverted_half * 10 + x % 10
            x //= 10

        # 當位數為偶數時（如 1221）：x == 12, reverted_half == 12
        # 當位數為奇數時（如 12321）：x == 12, reverted_half == 123（透過 // 10 去掉中間位）
        return x == reverted_half or x == reverted_half // 10
```