
**样例 1:**
```
输入: 'a'
输出: 'A'
```
**样例 2:**
```
输入: 'b'
输出: 'B'
```



```python
def lowercase_to_uppercase(self, character):
	#ASCII码中小写字母与对应的大写字母相差32
	return chr(ord(character) - 32)
```
pass