Lintcode 1243
计算字符串中的单词数，其中一个单词定义为不含空格的连续字符串。


```python
"""
输入: "Hello, my name is John"
输出: 5
解释：有五个字符串段落："Hello"、"my"、"name"、"is"、"John"
```


```python
    def count_segments(self, s: str) -> int:
        segment_count = 0

        for i in range(len(s)):
            if (i == 0 or s[i - 1] == ' ') and s[i] != ' ':
                segment_count += 1

        return segment_count
```
pass