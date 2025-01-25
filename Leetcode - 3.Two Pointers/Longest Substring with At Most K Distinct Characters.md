386
给定字符串_S_，找到最多有k个不同字符的最长子串_T_。

**样例 1:**
```python    
输入: S = "eceba" 并且 k = 3
输出: 4
解释: T = "eceb"
```
**样例 2:**
```python    
输入: S = "WORLD" 并且 k = 4
输出: 4
解释: T = "WORL" 或 "ORLD"
```


```python
def length_of_longest_substring_k_distinct(self, s, k):
	if not s:
		return 0
		
	counter = {}
	left = 0
	longest = 0
	for right in range(len(s)):
		counter[s[right]] = counter.get(s[right], 0) + 1
		while left <= right and len(counter) > k:
			counter[s[left]] -= 1
			if counter[s[left]] == 0:
				del counter[s[left]]
			left += 1
		
		longest = max(longest, right - left + 1)
	return longest
```
pass

解釋:
1. Create一個字典counter儲存sub字符串的字符頻率. 
2. 第一個指針right從左(id=0)開始往右移.  並往counter內加入字元並統計直到counter裡面字元>k
3. 第二個指針left從(id=0)搜尋到第一個指針right位置. 如果counter裡面字元>k, 