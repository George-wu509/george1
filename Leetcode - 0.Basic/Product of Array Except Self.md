
**样例1**
```
输入: [1,2,3,4]
输出: [24,12,8,6]
解释:
2*3*4=24
1*3*4=12
1*2*4=8
1*2*3=6
```
**样例2**
```
输入: [2,3,8]
输出: [24,16,6]
解释:
3*8=24
2*8=16
2*3=6
```


```python
def product_except_self(self, nums: List[int]) -> List[int]:
	# write your code here
	length = len(nums)
	answer = [0]*length
	
	# answer[i] 表示索引 i 左侧所有元素的乘积
	# 因为索引为 '0' 的元素左侧没有元素， 所以 answer[0] = 1
	answer[0] = 1
	for i in range(1, length):
		answer[i] = nums[i - 1] * answer[i - 1]
	
	# R 为右侧所有元素的乘积
	# 刚开始右边没有元素，所以 R = 1
	R = 1;
	for i in reversed(range(length)):
		# 对于索引 i，左边的乘积为 answer[i]，右边的乘积为 R
		answer[i] = answer[i] * R
		# R 需要包含右边所有的乘积，所以计算下一个结果时需要将当前值乘到 R 上
		R *= nums[i]
	
	return answer
```
pass