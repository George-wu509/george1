Lintcode 1908
给定一个字符串代表一个仅包含`"true","false","or","and"`的布尔表达式。  
你的任务是将这个表达式的值求出，返回`"true"`或`"false"`。  
如果该表达式是错误的，则返回`"error"`

以下是對LintCode 1908題「布尔表达式求值」的詳細解法解釋，包括逐步處理的邏輯、具體例子、複雜度分析，以及其他解法簡述。

---

### **問題分析**

該問題需要對一個包含布尔值 (`true`, `false`) 和邏輯操作符 (`or`, `and`) 的字符串表達式進行求值，並且要求遵循正確的優先級進行計算：

- `and` 的優先級高於 `or`。
- 表達式中的邏輯需要正確處理，並對非法的表達式返回 `"error"`。

---

### **解法步驟**

#### **1. 轉換為逆波蘭式（RPN）**

**核心邏輯：**

- 將中序表達式轉換為後序表達式（逆波蘭式），以便簡化計算。
- 使用棧來處理運算符的優先級，確保正確處理 `and` 和 `or` 的順序。

**詳細步驟：**

1. **初始化：**
    
    - 使用 `stack` 暫存運算符，使用 `rpn` 保存結果的後序表達式。
    - 定義優先級字典：`priority = {"#": 0, "or": 1, "and": 2}`。
2. **遍歷元素：**
    
    - 如果是布尔值 (`true` 或 `false`)，直接添加到 `rpn`。
    - 如果是運算符，將棧中優先級高於等於當前運算符的元素彈出並添加到 `rpn`，然後將當前運算符壓入棧。
    - 在輸入末尾添加特殊字符 `"#"`，確保所有運算符出棧。
3. **輸入合法性檢查：**
    
    - 如果布尔值之間相鄰（例如 `"true true"`）或運算符之間相鄰（例如 `"or or"`），視為非法表達式。
    - 一旦檢測到非法輸入，在 `rpn` 中插入 `"error"`，終止運算。

---

#### **2. 計算逆波蘭式（`evaluation` 函數）**

**核心邏輯：**

- 遍歷逆波蘭式，使用棧來計算最終結果。
- 每次遇到運算符時，從棧中彈出兩個布尔值進行計算，並將結果壓入棧。

**詳細步驟：**

1. **初始化棧：**
    
    - 遍歷 `rpn` 中的每個元素。
    - 如果是布尔值，直接壓入棧。
    - 如果是運算符，從棧中彈出兩個值進行計算。
2. **非法表達式檢查：**
    
    - 如果在需要運算符操作時棧內元素不足，返回 `"error"`。
    - 最終棧中應只有一個元素，否則返回 `"error"`。
3. **計算邏輯：**
    
    - `or`: 結果為 `true`，除非兩個操作數均為 `false`。
    - `and`: 結果為 `true`，只有兩個操作數均為 `true` 時。

Example:
样例 1
输入：
"true and false"
输出：
"false"

样例 2
输入：
"true or"
输出：
"error"


```python
請中文詳細一步步解釋下列LintCode 1908 布尔表达式求值的解法並具體舉例並附上複雜度分析, 以及簡單列出其他解法.

class Solution:
    """
    @param expression: a string that representing an expression
    @return: the result of the expression
    """
    def evaluation(self, expression):
        rpn = self.getRPN(expression)
        stack = []
        
        for element in rpn:
            if element == "error":
                return "error"
            if element in ["true", "false"]:
                stack.append(element)
            else:
                if not stack:
                    return "error"
                a = stack.pop()
                if not stack:
                    return "error"
                b = stack.pop()
                stack.append(self.calculate(a, b, element))
                
        if len(stack) != 1:
            return "error"
        return stack[-1]
        
    def getRPN(self, expression):
        stack = []
        rpn = []
        priority = {"#": 0, "or": 1, "and": 2}
        elements = expression.split()
        elements.append("#")
        
        for i in range(len(elements)):
            element = elements[i]
            if element in ["true", "false"]:
                if i > 0 and elements[i - 1] in ["true", "false"]:
                    rpn.append("error")
                    break
                if i < len(elements) - 1 and elements[i + 1] in ["true", "false"]:
                    rpn.append("error")
                    break
                rpn.append(element)
            else:
                if i > 0 and elements[i - 1] in ["or", "and"]:
                    rpn.append("error")
                    break
                if i < len(elements) - 1 and elements[i + 1] in ["or", "and"]:
                    rpn.append("error")
                    break
                while stack and priority[stack[-1]] >= priority[element]:
                    rpn.append(stack.pop())
                stack.append(element)
                
        return rpn
        
    def calculate(self, a, b, operator):
        if operator == "or":
            result = not(a == b == "false")
        else:
            result = (a == b == "true")
            
        if result:
            return "true"
        return "false"
```
pass
### **具體舉例**

#### 輸入：

`expression = "true and false or true"`

#### 步驟詳解：

1. **轉換為 RPN：**
    
    - 初始 `stack = []`，`rpn = []`。
    - 遍歷輸入：
        - `"true"`：加入 `rpn` -> `rpn = ["true"]`。
        - `"and"`：壓入棧 -> `stack = ["and"]`。
        - `"false"`：加入 `rpn` -> `rpn = ["true", "false"]`。
        - `"or"`：`"and"` 優先級高，彈出棧頂 -> `rpn = ["true", "false", "and"]`，`stack = ["or"]`。
        - `"true"`：加入 `rpn` -> `rpn = ["true", "false", "and", "true"]`。
    - 輸入結束，彈出剩餘棧頂 -> `rpn = ["true", "false", "and", "true", "or"]`。
2. **計算 RPN：**
    
    - 初始 `stack = []`。
    - 遍歷 `rpn = ["true", "false", "and", "true", "or"]`：
        - `"true"`：壓入棧 -> `stack = ["true"]`。
        - `"false"`：壓入棧 -> `stack = ["true", "false"]`。
        - `"and"`：彈出 `"false", "true"` 計算 -> `stack = ["false"]`。
        - `"true"`：壓入棧 -> `stack = ["false", "true"]`。
        - `"or"`：彈出 `"true", "false"` 計算 -> `stack = ["true"]`。
3. **最終結果：**
    
    - 棧中只剩一個元素，結果為 `"true"`。

---

### **時間與空間複雜度分析**

1. **時間複雜度：**
    
    - 轉換 RPN：遍歷輸入表達式一次，為 O(n)O(n)O(n)。
    - 計算 RPN：遍歷後序表達式一次，為 O(n)O(n)O(n)。
    - 總時間複雜度為 O(n)O(n)O(n)。
2. **空間複雜度：**
    
    - RPN 轉換時棧的最大深度為 O(n)O(n)O(n)。
    - 計算 RPN 時棧的最大深度為 O(n)O(n)O(n)。
    - 總空間複雜度為 O(n)O(n)O(n)。

---

### **其他解法簡述**

1. **遞歸解析法：**
    
    - 使用遞歸來處理優先級，先處理 `and`，再處理 `or`。
    - 時間複雜度：O(n)O(n)O(n)。
2. **直接計算法：**
    
    - 不轉換為 RPN，直接遍歷表達式，根據優先級處理操作符。
    - 時間複雜度：O(n2)O(n^2)O(n2)（最壞情況下，重複處理子表達式）。
3. **帶括號的解析：**
    
    - 如果表達式包含括號，先通過棧處理括號內的子表達式，再計算結果。
    - 時間複雜度：O(n)O(n)O(n)。

