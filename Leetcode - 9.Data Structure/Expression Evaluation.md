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