
```
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