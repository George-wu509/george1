
**样例 1:**
```
输入: S = "abcabcabca", k = 4
输出: 0
解释: 字符串中一共就只有 3 个不同的字符.
```
**样例 2:**
```
输入: S = "abcabcabcabc", k = 3
输出: 55
解释: 任意长度不小于 3 的子串都含有 a, b, c 这三个字符.
    比如,长度为 3 的子串共有 10 个, "abc", "bca", "cab" ... "abc"
    长度为 4 的子串共有 9 个, "abca", "bcab", "cabc" ... "cabc"
    ...
    长度为 12 的子串有 1 个, 就是 S 本身.
    所以答案是 1 + 2 + ... + 10 = 55.
```


```python
def k_distinct_characters(self, s, k):
	left = 0
	counter = {}
	answer = 0
	for right in range(len(s)):
		counter[s[right]] = counter.get(s[right], 0) + 1
		while left <= right and len(counter) >= k:
			counter[s[left]] -= 1
			if counter[s[left]] == 0:
				del counter[s[left]]
			left += 1
				
		if len(counter) == k - 1 and left > 0 and s[left - 1] not in counter:
			answer += left
			
	
	return answer
```
pass