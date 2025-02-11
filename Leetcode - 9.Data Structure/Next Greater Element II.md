Lintcode 1201
给定一个环形数组（最后一个元素的下一个元素是数组的第一个元素），为每个元素打印下一个更大的元素。 数字x的下一个更大的数是数组中下一个遍历顺序中出现的第一个更大的数字，这意味着您可以循环搜索以查找其下一个更大的数字。 如果它不存在，则为此数字输出-1。


例1:
```python
"""
输入: [1,2,1]
输出: [2,-1,2]
解释：第一个1的下一个更大的数字是2;
数字2找不到下一个更大的数字;
第二个1的下一个更大的数字需要循环搜索，答案也是2。
```
例2:
```python
"""
输入: [1]
输出: [-1]
解释：
数字1找不到下一个更大的数字
```


```python
def next_greater_elements(self, nums: List[int]) -> List[int]:
	n = len(nums)
	ret = [-1] * n
	stk = list()

	for i in range(n * 2 - 1):
		while stk and nums[stk[-1]] < nums[i % n]:
			ret[stk.pop()] = nums[i % n]
		stk.append(i % n)
	
	return ret
```
pass