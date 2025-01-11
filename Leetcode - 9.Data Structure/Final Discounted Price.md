
**示例 1:**
```
输入:
Prices = [2, 3, 1, 2, 4, 2]
输出: 
[1, 2, 1, 0, 2, 2]
解释：
第0个和第1个物品右边第一个更低的价格都是1，所以实际售价需要在全价上减去1， 第3个物品右边第一个更低的价格是2，所以实际售价要在全价上面减去2。 
```
**示例 2:**
```
输入:
Prices = [1, 2, 3, 4, 5]
输出: 
[1, 2, 3, 4, 5]
解释: 
每一个物品都保持原价，他们的右边都没有等于或者更低价格的物品
```


```python
def final_discounted_price(self, prices):
	s, res = [], [prices[i] for i in range(len(prices))]

	for i in range(len(prices)):
		while len(s) != 0 and prices[s[-1]] >= prices[i]:
			index = s[-1]
			s.pop()
			res[index] = prices[index] - prices[i]
		s.append(i);
	return res
```
pass