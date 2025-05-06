Lintcode 1025
字符串 S 和 T 只包含小写字符。在S中，所有字符只会出现一次。

S 已经根据某种规则进行了排序。我们要根据 S 中的字符顺序对 T 进行排序。更具体地说，如果 S 中 x 在 y 之前出现，那么返回的字符串中 x 也应出现在 y 之前。

字符串 T 中不存在于 S 之中的字符，**按照字典序排序放在字符串最后**即可。

```python
"""
示例:
输入:
s = "cba"
t = "abcd"
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
解釋:
step1: 先將t 用collections.Counter() 統計字元出現頻率. create空result
step2: 再依s順序 (c->b->a)依次按照頻率把字元加到result. 如果不在s裡面則放result後面 