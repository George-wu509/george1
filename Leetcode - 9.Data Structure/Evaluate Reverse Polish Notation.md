
LeetCode 150 / LintCode 424 給你一個字串陣列 `tokens`，表示一個根據 **逆波蘭表示法（Reverse Polish Notation, RPN）** 寫成的算術表達式。 請你計算該表達式的值，並回傳一個整數。

**注意事項：**

- 有效的運算子為 `'+'`、`'-'`、`'*'` 和 `'/'`。
- 每個運算元（Operand）可以是一個整數，或是另一個表達式。
- 兩個整數之間的除法總是 **向零截斷（Truncate toward zero）**（例如：`6 / -132 = 0`）。
- 輸入的逆波蘭表示式總是有效的，不存在除以零的情況。

**样例 1：** 输入：
```Python
tokens = ["2", "1", "+", "3", "*"]
```

输出：
```Python
9
```

解释： 該算式轉化為常見的中綴表達式為：((2 + 1) * 3) = 9

**样例 2：** 输入：
```Python
tokens = ["4", "13", "5", "/", "+"]
```

输出：
```Python
6
```

解释： 該算式轉化為常見的中綴表達式為：(4 + (13 / 5)) = 6

**样例 3：** 输入：
```Python
tokens = ["10", "6", "9", "3", "+", "-11", "*", "/", "*", "17", "+", "5", "+"]
```

输出：
```Python
22
```

解释： ((10 * (6 / ((9 + 3) * -11))) + 17) + 5 = ((10 * (6 / (12 * -11))) + 17) + 5 = ((10 * (6 / -132)) + 17) + 5 = ((10 * 0) + 17) + 5 = 17 + 5 = 22

```Python
def evalRPN(self, tokens: list[str]) -> int:
    stack = []
    for token in tokens:
        if token in {"+", "-", "*", "/"}:
            b = stack.pop()  # 先彈出的是右運算元
            a = stack.pop()  # 後彈出的是左運算元
            if token == "+":
                stack.append(a + b)
            elif token == "-":
                stack.append(a - b)
            elif token == "*":
                stack.append(a * b)
            elif token == "/":
                stack.append(int(a / b))  # 向零截斷，不可用 a // b
        else:
            stack.append(int(token))
    return stack[0]
```


```Python
# 寫法二：使用 Dictionary + Lambda 簡化分支判斷
def evalRPN(self, tokens: list[str]) -> int:
    ops = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: int(a / b),
    }
    stack = []
    for token in tokens:
        if token in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[token](a, b))
        else:
            stack.append(int(token))
    return stack[0]
```

解釋: step1: 逆波蘭表示式的特性是「運算子寫在運算元後面」，非常適合用 **棧（Stack）** 來模擬計算。 step2: 以 `tokens = ["4", "13", "5", "/", "+"]` 為例，初始化空棧 `stack = []`。 step3: 遇到 `"4"`, `"13"`, `"5"` 皆為數字，依序轉為整數推入棧中 -> `stack = [4, 13, 5]`。 step4: 遇到 `"/"`，彈出兩個數字：`b = 5`（右運算元）、`a = 13`（左運算元），計算 `int(13 / 5) = 2` 推回棧中 -> `stack = [4, 2]`。 step5: 遇到 `"+"`，彈出兩個數字：`b = 2`、`a = 4`，計算 `4 + 2 = 6` 推回棧中 -> `stack = [6]`。 step6: 遍歷結束，棧內唯一元素 `stack[0] = 6` 即為答案。

解釋: 核心規則：遇到數字就 `push`；遇到運算子就 `pop` 出兩個數字進行運算，並將結果 `push` 回去。

特別注意 **減法與除法的順序**：先 `pop()` 出來的是右邊的數字 `b`，後 `pop()` 出來的是左邊的數字 `a`（即 `a - b` 與 `a / b`）。

### **LeetCode 150 / LintCode 424 - Evaluate Reverse Polish Notation（逆波蘭表示式求值）**

這道題目是 **棧（Stack）** 資料結構最經典的應用之一（Blind 75 & NeetCode 150 必考題）。逆波蘭表示法又稱「後綴表達式」，它的最大優勢是不需要括號就能無歧義地表示運算優先級，非常適合電腦透過單次線性掃描配合 Stack 完成求值。

## **解法解析**

這道題可以直接透過 **單調模擬（Stack Simulation）** 來求解。

### **思路**

1. **觀察規律**
    
    - 在後綴表達式中，每個運算子永遠作用於它 **緊鄰的前兩個有效運算結果**。
        
    - 例如 `["2", "1", "+", "3", "*"]`：
        
        - `2` 和 `1` 先進場等待。
            
        - 看到 `+` 時，把最近進場的 `2` 和 `1` 合併成 `3`。
            
        - 接著 `3` 進場，看到 `*` 時，把前面合併的 `3` 和剛進場的 `3` 相乘得到 `9`。
            
    - 這種「後進先出（LIFO）」的特性完全契合 **Stack**。
        
2. **關鍵細節與陷阱（Python 專屬坑點）**
    
    - **運算元順序**：對於加法 `+` 和乘法 `*` 順序無所謂，但對於減法 `-` 和除法 `/`，第一個彈出的元素是 **右運算元 `b`**，第二個彈出的才是 **左運算元 `a`**。
        
    - **向零截斷（Truncate toward zero）**：
        
        - Python 的地板除 `//` 是 **向下取整**（向負無窮截斷），例如 `-3 // 2` 會得到 `-2`。
            
        - 題目要求向零截斷（`-3 / 2` 應為 `-1`），因此在 Python 中必須使用 `int(a / b)`。
            
3. **代碼執行流程**
    
    - 初始化一個空列表 `stack = []`。
        
    - 逐一掃描 `tokens` 中的每個字串 `token`：
        
        - 若 `token` 為運算子（`+`, `-`, `*`, `/`），從 `stack` 依序彈出 `b` 和 `a`，計算 `a op b` 後將整數結果 `append` 回 `stack`。
            
        - 若 `token` 為數字，使用 `int(token)` 轉換後 `append` 進 `stack`。
            
    - 迴圈結束後，合法表達式一定只會剩下一個結果，回傳 `stack[0]`。
        

## **具體舉例**

假設 `tokens = ["4", "13", "5", "/", "+"]`，我們的變數變化如下：

|**步驟**|**當前 token**|**操作動作**|**stack 狀態**|
|---|---|---|---|
|初始化|-|建立空棧|`[]`|
|1|`"4"`|數字入棧 `push(4)`|`[4]`|
|2|`"13"`|數字入棧 `push(13)`|`[4, 13]`|
|3|`"5"`|數字入棧 `push(5)`|`[4, 13, 5]`|
|4|`"/"`|彈出 `b=5, a=13`，計算 `int(13 / 5) = 2` 入棧|`[4, 2]`|
|5|`"+"`|彈出 `b=2, a=4`，計算 `4 + 2 = 6` 入棧|`[6]`|

最後返回 `stack[0]`，即 **`6`**。

## **時間與空間複雜度分析**

- **時間複雜度：O(n)**
    
    - 其中 `n` 為陣列 `tokens` 的長度。我們只需遍歷一次 `tokens`，每個元素最多入棧一次、出棧一次，每次運算皆為 $O(1)$，總時間複雜度為 **O(n)**。
        
- **空間複雜度：O(n)**
    
    - 最壞情況下（例如所有數字都在前半段、所有運算子都在後半段），棧中最多會同時儲存 $\lceil n/2 \rceil$ 個運算元，因此空間複雜度為 **O(n)**。
        

## **其他解法**

### **1. 原地修改陣列（雙指針模擬 Stack）**

- **思路**：
    
    - 不額外開闢 `stack` 陣列，而是直接用一個寫入指針 `pos` 在原陣列 `tokens` 上覆寫計算結果（將 `tokens[0...pos-1]` 視為棧）。
        
    - 遇到數字時 `tokens[pos] = int(token); pos += 1`；遇到運算子時 `pos -= 1`，計算 `tokens[pos-1]` 與 `tokens[pos]` 並存回 `tokens[pos-1]`。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(1)（不計輸入陣列本身的空間，但會破壞原始輸入資料）
    

### **2. 從右往左遞迴（Recursion / DFS）**

- **思路**：
    
    - 逆波蘭表示式的最後一個元素必定是整個表達式的根運算子（Root Operator）。
        
    - 從陣列尾端開始往前遞迴：遇到運算子就先遞迴求右子表達式 `b`，再遞迴求左子表達式 `a`；遇到數字則直接回傳該整數。
        
- **時間複雜度**：O(n)
    
- **空間複雜度**：O(n)（遞迴呼叫棧的深度）
    

## **總結**

- **最佳解法（面試首選）**：**標準 Stack 模擬解法**，邏輯最清晰、不破壞原始輸入，時間 $O(n)$，空間 $O(n)$。
    
- **工程潔癖寫法**：使用 **Dictionary 映射 `lambda` 或 `operator` 模組**，避免冗長的 `if-elif` 分支，擴充新運算子（如次方 `^`、取餘 `%`）時只需改字典。
    
- **面試必考細節**：務必主動向面試官確認 **負數除法的截斷方向**，並在 Python 中寫出 `int(a / b)` 而非 `a // b`。
    

## **LeetCode / LintCode 棧（Stack）與表達式求值類問題整理**

|**題號**|**題目名稱**|**問題類型**|**解法**|
|---|---|---|---|
|**LeetCode 150**|Evaluate Reverse Polish Notation（逆波蘭表示式求值）|**後綴表達式求值**|**Stack 模擬**：遇數字入棧，遇運算子彈出兩數計算（注意 `int(a/b)`）|
|**LeetCode 20**|Valid Parentheses（有效的括號）|**括號匹配問題**|**Stack + HashMap**：左括號入棧，右括號檢查棧頂是否配對|
|**LeetCode 155**|Min Stack（最小棧）|**資料結構設計**|**雙棧（主棧 + 輔助最小值棧）**，實現 $O(1)$ 取得最小值|
|**LeetCode 224**|Basic Calculator（基本計算器）|**中綴表達式（含 `+`, `-`, `(`, `)`）**|**Stack 儲存括號前的結果與正負號（Sign）**|
|**LeetCode 227**|Basic Calculator II（基本計算器 II）|**中綴表達式（含 `+`, `-`, `*`, `/`，無括號）**|**Stack 處理乘除優先級**：遇 `*`/`/` 立即與棧頂計算，遇 `+`/`-` 帶符號入棧最後求和|
|**LeetCode 772**|Basic Calculator III（基本計算器 III）|**完整中綴表達式（含四則運算與括號）**|**雙棧（數字棧 + 運算子棧）** 或 **遞迴處理括號 + Stack 處理優先級**|
|**LeetCode 394**|Decode String（字串解碼）|**巢狀結構展開（如 `3[a2[c]]`）**|**雙棧（數字棧 + 字串棧）**：遇 `[` 存狀態，遇 `]` 彈出並重複拼接|
|**LeetCode 735**|Asteroid Collision（行星碰撞）|**相鄰元素抵消模擬**|**Stack 模擬碰撞**：只有棧頂向右 (`>0`) 且當前向左 (`<0`) 才會相撞|
|**LeetCode 739**|Daily Temperatures（每日溫度）|**尋找下一個更大元素（Next Greater Element）**|**單調遞減棧（Monotonic Stack）**：棧內存索引，遇更高溫則彈出計算天數差|
|**LeetCode 84**|Largest Rectangle in Histogram（柱狀圖中最大的矩形）|**尋找左右第一個更小元素**|**單調遞增棧（Monotonic Stack）**，時間複雜度 $O(n)$|

### **進階變體練習：將普通中綴表達式轉為逆波蘭表示式（Shunting Yard 調度場演算法）**

如果題目改為給你一個包含 `+`, `-`, `*`, `/` 和括號 `(`, `)` 的普通算式字串（例如 `"(2 + 1) * 3"`），要求先轉換成逆波蘭表示式再求值：

Python

```
class Solution:
    def infix_to_rpn(self, tokens: list[str]) -> list[str]:
        # 定義運算子優先級
        precedence = {"+": 1, "-": 1, "*": 2, "/": 2}
        output = []   # 存放輸出的後綴表達式
        op_stack = [] # 暫存運算子與左括號

        for token in tokens:
            if token.lstrip("-").isdigit():  # 支援負數與多位數
                output.append(token)
            elif token == "(":
                op_stack.append(token)
            elif token == ")":
                # 彈出直到遇見左括號
                while op_stack and op_stack[-1] != "(":
                    output.append(op_stack.pop())
                op_stack.pop()  # 彈出 "(" 並丟棄
            elif token in precedence:
                # 若棧頂運算子優先級 >= 當前運算子，先將棧頂彈出輸出
                while (
                    op_stack
                    and op_stack[-1] != "("
                    and precedence[op_stack[-1]] >= precedence[token]
                ):
                    output.append(op_stack.pop())
                op_stack.append(token)

        # 將棧內剩餘的運算子全部彈出
        while op_stack:
            output.append(op_stack.pop())

        return output
```