
```
示例:
输入:
S = "cba"
T = "abcd"
输出: "cbad"
解释: 
S 中出现了字符 "a", "b", "c", 所以 "a", "b", "c" 的顺序应该是 "c", "b", "a"
由于 "d" 没有在 S 中出现, 因此它放在 T 的末端。
```


```python
    def custom_sort_string(self, s: str, t: str) -> str:
        counter = collections.Counter(t)
        result = []

        for c in s:
            if c in counter:
                freq = counter[c]
                result += [c] * freq
                counter.pop(c)

        for c in sorted(counter.keys()):
            result += [c] * counter[c]

        return ''.join(result)
```
pass