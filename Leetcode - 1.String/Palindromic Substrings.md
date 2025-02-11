Lintcode 837
给定一个字符串，你的任务是数出有多少个回文子串在这个字符串内。  
一个子串不同于其他的子串，当且仅当开始和结束位置不同。
**样例1**
```python
"""
输入: "abc"
输出: 3
解释:
3个回文字符串: "a", "b", "c".
```
**样例2**
```python
"""
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