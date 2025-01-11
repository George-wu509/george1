
**样例 1:**
```
输入: "00110011"
输出: 6
解释: 有6个符合题目的连续子串："0011", "01", "1100", "10", "0011", and "01".

注意重复的子串会记录多次。

而且, "00110011" 是不合理的子串，因为所有的0和1没有连在一起。
```
**样例 2:**
```
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