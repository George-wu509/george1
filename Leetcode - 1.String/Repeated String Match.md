Lintcode 1086
给定两个字符串A和B，找到A必须重复的最小次数，以使得B是它的子字符串。 如果没有这样的解决方案，返回-1。

**样例1:**
```python
"""
输入 : A = "a"     B = "b".
输出 : -1
```
**样例 2:**
```python
"""
输入 : A = "abcd"     B = "cdabcdab".
输出 :3
解释：因为将A重复3次以后 (“abcdabcdabcd”), B将成为其的一个子串 ; 而如果A只重复两次 ("abcdabcd")，B并非其的一个子串.
```


```python
class Solution:
    def strstr(self, haystack: str, needle: str) -> int:
        n, m = len(haystack), len(needle)
        if m == 0:
            return 0

        k1 = 10 ** 9 + 7
        k2 = 1337
        mod1 = random.randrange(k1) + k1
        mod2 = random.randrange(k2) + k2

        hash_needle = 0
        for c in needle:
            hash_needle = (hash_needle * mod2 + ord(c)) % mod1
        hash_haystack = 0
        for i in range(m - 1):
            hash_haystack = (hash_haystack * mod2 + ord(haystack[i % n])) % mod1
        extra = pow(mod2, m - 1, mod1)
        for i in range(m - 1, n + m - 1):
            hash_haystack = (hash_haystack * mod2 + ord(haystack[i % n])) % mod1
            if hash_haystack == hash_needle:
                return i - m + 1
            hash_haystack = (hash_haystack - extra * ord(haystack[(i - m + 1) % n])) % mod1
            hash_haystack = (hash_haystack + mod1) % mod1
        return -1

    def repeated_string_match(self, a: str, b: str) -> int:
        # write your code here
        n, m = len(a), len(b)
        index = self.strstr(a, b)
        if index == -1:
            return -1
        if n - index >= m:
            return 1
        return (m + index - n - 1) // n + 2
```
pass