Lintcode 575
给出一个表达式 `s`，此表达式包括数字，字母以及方括号。在方括号前的数字表示方括号内容的重复次数(括号内的内容可以是字符串或另一个表达式)，请将这个表达式展开成一个字符串。

Example
样例1
输入: S = abc3[a]
输出: "abcaaa"

样例2
输入: S = 3[2[ad]3[pf]]xyz
输出: "adadpfpfpfadadpfpfpfadadpfpfpfxyz"
```python
    def expression_expand(self, s: str) -> str:
        stack = []
        for c in s:
            if c != ']':
                stack.append(c)
                continue
                
            strs = []
            while stack and stack[-1] != '[':
                strs.append(stack.pop())
            
            # skip '['
            stack.pop()
            
            repeats = 0
            base = 1
            while stack and stack[-1].isdigit():
                repeats += (ord(stack.pop()) - ord('0')) * base
                base *= 10
            stack.append(''.join(reversed(strs)) * repeats)
        
        return ''.join(stack)
```
pass