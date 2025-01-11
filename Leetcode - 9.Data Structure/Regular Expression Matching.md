
**样例 1:**
```
输入："aa"，"a"
输出：false
解释：
无法匹配
```
**样例 2:**
```
输入："aa"，"a*"
输出：true
解释：
'*' 可以重复 a
```
**样例 3:**
```
输入："aab", "c*a*b"
输出：true
解释：
"c*" 作为一个整体匹配 0 个 'c' 也就是 ""
"a*" 作为一个整体匹配 2 个 'a' 也就是 "aa"
"b" 匹配 "b"
所以 "c*a*b" 可以匹配 "aab"
```
**样例4：**
```
输入："abcc", ".*"
输出：true
解释：
".*" 作为一个整体匹配 4 个 '.' 也就是 "...."
第一个 "." 匹配第一个字符 "a"
第二个 "." 匹配第二个字符 "b"
最后两个 "." 匹配字符 "cc"
```


```python
class Solution:
    """
    @param s: A string 
    @param p: A string includes "?" and "*"
    @return: is Match?
    """
    def is_match(self, s, p):
        return self.is_match_helper(s, 0, p, 0, {})
        
        
    # s 从 i 开始的后缀能否匹配上 p 从 j 开始的后缀
    # 能 return True
    def is_match_helper(self, s, i, p, j, memo):
        if (i, j) in memo:
            return memo[(i, j)]
        
        # s is empty
        if len(s) == i:
            return self.is_empty(p[j:])
            
        if len(p) == j:
            return False
            
        if j + 1 < len(p) and p[j + 1] == '*':
            matched = self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j, memo) or \
                self.is_match_helper(s, i, p, j + 2, memo)
        else:                
            matched = self.is_match_char(s[i], p[j]) and self.is_match_helper(s, i + 1, p, j + 1, memo)
        
        memo[(i, j)] = matched
        return matched
        
        
    def is_match_char(self, s, p):
        return s == p or p == '.'
        
    def is_empty(self, p):
        if len(p) % 2 == 1:
            return False
        
        for i in range(len(p) // 2):
            if p[i * 2 + 1] != '*':
                return False
        return True
```
pass