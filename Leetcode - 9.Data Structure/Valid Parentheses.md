
样例 1：
输入：
s = "([)]"
输出：
False

样例 2：
输入：
s = "(){}[]"
输出：
True

样例 3：
输入：
s = "({})"
输出：
True

样例 4：
输入：
s = "({[()]})"
输出：
True
```python
class Solution:
    """
    @param s: A string
    @return: whether the string is a valid parentheses
    """
    def is_valid_parentheses(self, s: str) -> bool:
        stack = []
        for ch in s:
            # 压栈
            if ch == '{' or ch == '[' or ch == '(':
                stack.append(ch)
            else:
                # 栈需非空
                if not stack:
                    return False
                # 判断栈顶是否匹配
                if ch == ']' and stack[-1] != '[' or ch == ')' and stack[-1] != '(' or ch == '}' and stack[-1] != '{':
                    return False
                # 弹栈
                stack.pop()
        return not stack
```
pass