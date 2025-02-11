Lintcode 1079
给定字符串`s`，计算有相同数量的0和1的非空连续子串的数量，并且子串中所有的0和所有的1都是连续的。

相同的子串出现多次则计数多次。

**样例 1:**
```python
"""
输入: "00110011"
输出: 6
解释: 有6个符合题目的连续子串："0011", "01", "1100", "10", "0011", and "01".

注意重复的子串会记录多次。

而且, "00110011" 是不合理的子串，因为所有的0和1没有连在一起。
```
**样例 2:**
```python
"""
输入: "10101"
输出: 4
解释: 有4个合题的连续子串: "10", "01", "10", "01"。
```


```python
def count_binary_substrings(self, s: str) -> int:
	counts = list()
	ptr, n = 0, len(s)
	while ptr < n:
		c = s[ptr]
		count = 0
		while ptr < n and s[ptr] == c:
			ptr += 1
			count += 1
		counts.append(count)
	ans = 0
	for i in range(1, len(counts)):
		ans += min(counts[i], counts[i - 1])
	return ans
```
pass