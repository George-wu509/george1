Lintcode 613
每个学生有两个属性 `ID` 和 `scores`。找到每个学生最高的5个分数的平均值。

例1:
```python
"""
输入: 
[[1,91],[1,92],[2,93],[2,99],[2,98],[2,97],[1,60],[1,58],[2,100],[1,61]]
输出:
1: 72.40
2: 97.40
```
例2:
```python
"""
输入:
[[1,90],[1,90],[1,90],[1,90],[1,90],[1,90]]
输出: 
1: 90.00
```


```python
def highFive(self, results):
	hash = dict()
	for r in results:
		if r.id not in hash:
			hash[r.id] = []

		hash[r.id].append(r.score)
		if len(hash[r.id]) > 5:
			index = 0
			for i in range(1, 6):
				if hash[r.id][i] < hash[r.id][index]:
					index = i

			hash[r.id].pop(index)

	answer = dict()
	for id, scores in hash.items():
		answer[id] = sum(scores) / 5.0

	return answer
```
pass