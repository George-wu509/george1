Lintcode 368
给一个用字符串表示的表达式数组，求出这个表达式的值。

**样例 1:**
```python
"""
对于表达式 `2*6-(23+7)/(1+2)`,
输入:
["2", "*", "6", "-", "(","23", "+", "7", ")", "/", "(", "1", "+", "2", ")"]
输出:
2
```
**样例 2:**
```python
"""
对于表达式 `4-(2-3)*2+5/5`,
输入:
["4", "-", "(", "2","-", "3", ")", "*", "2", "+", "5", "/", "5"]
输出:
7
```


```python
class Solution:
    def calc(self, a, operator, b):
        if operator == '+':
            return a + b
        elif operator == '-':
            return a - b
        elif operator == '*':
            return a * b
        else:
            return a // b
            
    def divide_expression(self, expression, operators):
        last_index, parens = 0, 0
        last_operator = operators[0]
        total = 1 if last_operator == '*' else 0
        can_divide = False
        
        for index, elem in enumerate(expression):
            if elem in operators and parens == 0:
                can_divide = True
                val = self.evaluate_expression(expression[last_index: index])
                total = self.calc(total, last_operator, val)
                last_operator = elem
                last_index = index + 1
            elif elem == '(':
                parens += 1
            elif elem == ')':
                parens -= 1
        
        if can_divide:
            val = self.evaluate_expression(expression[last_index:])
            total = self.calc(total, last_operator, val)
            
        return can_divide, total

    """
    @param expression: a list of strings
    @return: an integer
    """
    def evaluate_expression(self, expression):
        if not expression:
            return 0
        if len(expression) == 1:
            return int(expression[0])
            
        can_divide, total = self.divide_expression(expression, ['+', '-'])
        if can_divide:
            return total
    
        can_divide, total = self.divide_expression(expression, ['*', '/'])
        if can_divide:
            return total
            
        # must be parens around the expression
        return self.evaluate_expression(expression[1:-1])
```
pass


### **LintCode 368 - Expression Evaluation**

#### **解法分析**

本題的目標是**計算算術表達式的值**，其中表達式以**字串列表**的形式給出，包含 **加 `+`、減 `-`、乘 `*`、除 `/`** 和 **括號 `()`**。

解法主要基於 **遞歸解析** 和 **遞歸拆解運算優先級** 來解析並計算結果。

---

### **解法思路**

#### **1. 遞歸解析 `evaluate_expression(expression)`**

- **基礎情況**
    
    - 若 `expression` 為空，返回 `0`。
    - 若 `expression` 僅含一個元素，直接轉換為整數返回。
- **遞歸拆解**
    
    - 先嘗試以 **加 `+`、減 `-`** 運算符分割表達式，計算結果。
    - 若無法用加減號分割，則用 **乘 `*`、除 `/`** 運算符分割表達式並計算結果。
    - 若仍無法分割，表示 `expression` 可能被括號包圍，則遞歸解析 `expression[1:-1]`。

#### **2. 運算符優先級拆分 `divide_expression(expression, operators)`**

- **遍歷 `expression`，根據 `operators` (加減/乘除) 分割數字與運算符**
    - 使用 `parens` 計算括號層級，確保不在括號內進行拆分。
    - 遇到 `operators`（例如 `+` 或 `-`），執行 **遞歸計算左側數值**，並更新當前運算符。
    - 遍歷完畢後，計算最後一個數值部分，並返回計算結果。

#### **3. 運算函數 `calc(a, operator, b)`**

- 根據 `operator` 計算 `a op b`，支援 `+`、`-`、`*`、`//`（整數除法）。

---

### **變數說明**

|變數名稱|說明|
|---|---|
|`expression`|以字串列表形式給出的數學表達式|
|`operators`|目前用於拆分表達式的運算符 (`['+', '-']` 或 `['*', '/']`)|
|`total`|計算的當前累計值|
|`last_index`|用來記錄當前子表達式的開始索引|
|`parens`|計算括號的層級，確保不在括號內拆分|
|`last_operator`|記錄上一個運算符|
|`can_divide`|是否成功以當前 `operators` 拆分表達式|

---

### **範例**

#### **輸入**

python

複製編輯

`expression = ["2", "*", "(", "3", "+", "4", ")", "-", "5"]`

#### **處理流程**

1. **第一層解析 `evaluate_expression(expression)`**
    
    - 先嘗試 **加減 `+`、`-`** 拆分：
        - `["2", "*", "(", "3", "+", "4", ")", "-", "5"]` → `["2", "*", "(3+4)"]` `-` `["5"]`
        - 左側: `evaluate_expression(["2", "*", "(", "3", "+", "4", ")"])`
        - 右側: `evaluate_expression(["5"])`
2. **計算 `["2", "*", "(", "3", "+", "4", ")"]`**
    
    - 括號內 `["3", "+", "4"]` 遞歸計算為 `7`
    - 乘法 `2 * 7 = 14`
3. **計算 `14 - 5 = 9`**
    
    - **輸出結果：9**

---

### **輸出**

python

複製編輯

`9`

---

### **時間與空間複雜度分析**

#### **時間複雜度**

- 每次拆分表達式時，會遞歸地計算左右部分，最壞情況為完全嵌套的括號 (如 `"(((1+1)+1)+1)"`)。
- 每次計算 `divide_expression` 需要遍歷 `O(N)`，最多嵌套 `O(N)` 次，導致最壞 **O(N²)**。

#### **空間複雜度**

- 主要來自遞歸調用棧的深度，最壞情況為 `O(N)`（所有運算符都是括號嵌套情況）。

---

### **其他解法想法**

4. **使用 `Stack` 進行運算 (O(N))**
    
    - 使用 **操作數棧 (operand stack)** 和 **運算符棧 (operator stack)** 來進行運算。
    - 若遇到 `(`，壓入棧中；遇到 `)`，彈出並計算直到 `(` 為止。
    - **時間複雜度 O(N)，空間 O(N)**。
5. **轉換為後綴表達式 (Postfix) + `Stack` (O(N))**
    
    - 使用 **Shunting Yard Algorithm** 轉換中綴表達式為後綴表達式，然後使用棧計算後綴表達式的值。
    - **時間複雜度 O(N)，空間 O(N)**。
6. **遞歸下降解析 (Recursive Descent Parsing) (O(N))**
    
    - 使用解析函數解析不同優先級的運算符，例如：
        - `parse_expression()`
        - `parse_term()`（處理 `*`、`/`）
        - `parse_factor()`（處理 `+`、`-`）
    - **時間複雜度 O(N)，空間 O(N)**。

---

### **總結**

- **最優解法：`Stack` + `Operator Precedence` (O(N) 時間, O(N) 空間)**
- **若 `N` 不大，遞歸解析 (O(N²)) 也可接受**。
- **若考慮運算符優先順序，後綴表達式方法是較優選擇 (O(N))**。