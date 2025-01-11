
**样例1**
```
输入: "abc"
输出: 3
解释:
3个回文字符串: "a", "b", "c".
```
**样例2**
```
输入: "aba"
输出: 4
解释:
4个回文字符串: "a", "b", "a", "aba".
```


```python
def count_palindromic_substrings(self, str: str) -> int:
	n = len(str)
	ans = 0
	for i in range(2 * n - 1):
		left = i // 2
		right = i // 2 + i % 2
		while left >= 0 and right < n and str[left] == str[right]:
			left -= 1
			right += 1
			ans += 1

	return ans
```
pass