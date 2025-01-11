
```
样例 1:
	输入: "abaccdeff"
	输出:  'b'
	
	解释:
	'b' 是第一个出现一次的字符


样例 2:
	输入: "aabccd"
	输出:  'b'
	
	解释:
	'b' 是第一个出现一次的字符
```


```python
    def first_uniq_char(self, str):
        counter = {}

        for c in str:
            counter[c] = counter.get(c, 0) + 1

        for c in str:
            if counter[c] == 1:
                return c
```
pass