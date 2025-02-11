Lintcode 213
设计一种方法，通过给重复字符计数来进行基本的字符串压缩。

例如，字符串 `aabcccccaaa` 可压缩为 `a2b1c5a3` 。而如果压缩后的字符数不小于原始的字符数，则返回原始的字符串。

可以假设字符串仅包括 a-z 的大/小写字母

**样例 1：**
```python
"""
输入：str = "aabcccccaaa"
输出："a2b1c5a3"
```
**样例 2：**
```python
"""
输入：str = "aabbcc"
输出："aabbcc"
```


```python
    def compress(self, original_string):
        if len(original_string) == 0:
            return original_string
        ans = []
        count = 1
        last = original_string[0]
        for c in  original_string[1:]:
            if c == last:
                count += 1
            else:
                ans.append(last)
                ans.append("%s" % count)
                last = c
                count = 1

        ans.append(last)
        ans.append("%s" % count)
```
pass