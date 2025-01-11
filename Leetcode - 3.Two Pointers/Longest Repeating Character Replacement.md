

**样例1**
```
输入:
"ABAB"
2
输出:
4
解释:
将两个'A’替换成两个’B’，反之亦然。
```
**样例2**
```
输入:
"AABABBA"
1
输出:
4
解释:
将中间的 'A’ 替换为 'B' 后得到 “AABBBBA"。
子字符串"BBBB" 含有最长的重复字符, 长度为4。
```


```python
def character_replacement(self, s, k):
	counter = {}
	answer = 0
	j = 0
	max_freq = 0
	for i in range(len(s)):
		while j < len(s) and j - i - max_freq <= k:
			counter[s[j]] = counter.get(s[j], 0) + 1 
			max_freq = max(max_freq, counter[s[j]])
			j += 1 
		
		# 如果替换 除出现次数最多的字母之外的其他字母 的数目>k,
		# 说明有一个不能换，答案与j-i-1进行比较；
		# 否则说明直到字符串末尾替换数目都<=k，可以全部换掉 
		# 答案与子串长度j-i进行比较
		if j - i - max_freq > k:
			answer = max(answer, j - 1 - i)
		else:
			answer = max(answer, j - i) 
			
		# 起点后移一位，当前起点位置的字母个数-1
		counter[s[i]] -= 1
	return answer
```
pass