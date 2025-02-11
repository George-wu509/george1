Lintcode 980
实现一个基础计算器来计算一个简单表达式字符串。
这个表达式字符串只包含 **非负** 整数，运算符 `+`, `-`, `*`, `/` 以及空格 。 整数除法应该舍去小数。
你可以假设给出的表达式总是合理的。

Example:
例1:
输入:
"3+2*2"
输出:7

例2:
输入:
" 3/2 "
输出:1

```python
    def calculate(self, s: str) -> int:
        # Write your code here
        n = len(s)
        stack = []
        preSign = '+'
        num = 0
        for i in range(n):
            if s[i] != ' ' and s[i].isdigit():
                num = num * 10 + ord(s[i]) - ord('0')
            if i == n - 1 or s[i] in '+-*/':
                if preSign == '+':
                    stack.append(num)
                elif preSign == '-':
                    stack.append(-num)
                elif preSign == '*':
                    stack.append(stack.pop() * num)
                else:
                    stack.append(int(stack.pop() / num))
                preSign = s[i]
                num = 0
        return sum(stack)
```
pass