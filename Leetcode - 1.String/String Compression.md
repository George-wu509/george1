
**样例 1：**
```
输入：str = "aabcccccaaa"
输出："a2b1c5a3"
```
**样例 2：**
```
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