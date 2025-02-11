Lintcode 849
实现一个基本的计算器来计算一个简单的表达式字符串。
表达式字符串只包含**非负**整数、`+`, `-`, `*`, `/`操作符、左括号 `(`，右括号 `)`和空格。
您可以假设**给定的表达式总是有效的**。所有中间结果将在“[-2147483648,2147483647]”范围内。

Example:
样例 1:
输入："1 + 1"
输出：2
解释：1 + 1 = 2

样例 2:
输入：" 6-4 / 2 "
输出：4
解释：4/2=2，6-2=4

```python

PRIORITY = {
    '#': 0,
    '(': 1,
    ')': 1,
    '+': 2,
    '-': 2,
    '*': 3,
    '/': 3,
}

class Solution:
    """
    @param s: the expression string
    @return: the answer
    """
    def calculate(self, s):
        expression = []
        val = None
        for char in s:
            if char == ' ':
                continue
            if char in ['+', '-', '*', '/', '(', ')']:
                if val is not None:
                    expression.append(str(val))
                expression.append(char)
                val = None
            else:
                if val is None:
                    val = 0
                val = val * 10 + ord(char) - ord('0')
        if val is not None:
            expression.append(str(val))
        
        return self.evaluate_expression(expression)
            
    def evaluate_expression(self, expression):
        expression = [*expression, '#']
        rpe = []
        stack = []
        for elem in expression:
            if elem == '(':
                stack.append(elem)
            elif elem == ')':
                while stack and stack[-1] != '(':
                    operator = stack.pop()
                    rpe.append(operator)
                stack.pop()
            elif elem in ['+', '-', '*', '/', '#']:
                while stack and PRIORITY[stack[-1]] >= PRIORITY[elem]:
                    operator = stack.pop()
                    rpe.append(operator)
                stack.append(elem)
            else:
                rpe.append(elem)
        return self.evaluate_reverse_polish_expression(rpe)
    
    def evaluate_reverse_polish_expression(self, expression):
        stack = []
        for elem in expression:
            if elem in ['+', '-', '*', '/']:
                b = stack.pop()
                a = stack.pop()
                stack.append(self.calc(a, elem, b))
            else:
                stack.append(int(elem))
        return stack[-1]
                    
    def calc(self, a, operator, b):
        if operator == '+':
            return a + b
        elif operator == '-':
            return a - b
        elif operator == '*':
            return a * b
        else:
            return a // b
```
pass