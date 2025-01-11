
**样例 1:**
```
输入: [10,13,12,14,15]
输出: 2
```

解析:  
对于起点 i = 0，我们能跳跃到 i = 2 因为 `A[2]` 是 `A[0]` 右边大于等于`A[0]`的值中 `(A[1], A[2], A[3], A[4])` 最小的，  
然后无法继续跳跃，因为 `A[2]` 的右边不存在小于等于 `A[2]` 的位置。  
对于起点 i = 1 和 i = 2，第一次我们都可以跳跃至 i = 3，但是无法从 i = 3 进行第二次跳跃。  
对于起点 i = 3 ，我们可以跳跃至 i = 4，到达末尾，i = 4 合法。  
对于起点 i = 4 ，我们已经在末尾了，i = 4 合法。  
所以有两个个合法的索引 (i = 3, i = 4)，答案是 2。

**样例 2:**
```
输入: [2,3,1,1,4]
输出: 3
```

解析:  
对于起点 i = 0：  
第一次跳跃 (奇数跳), 我们可以从 i = 0 跳到 i = 1, 因为 `A[1]` 是`A[0]`右边大于等于`A[0] (A[1], A[4])` 当中最小的，  
第二次跳跃 (偶数跳), 我们可以从 i = 1 跳到 i = 2, 因为 `A[2]` 是`A[1]`右边小于等于`A[1] (A[2], A[3])` 当中最大的，虽然 `A[3]` 也是最大的, 但是 下标2 比 3 小，要选择更小的下标 。  
第三次跳跃 (奇数跳), 我们可以从 i = 2 跳到 i = 3, 因为 `A[3] 是A[2]`右边大于等于`A[2] (A[3], A[4])` 当中最小的。  
但是这时候我们没有办法继续进行第四次跳跃。

对于起点 i = 1：第一次跳跃可以从 i = 1 跳到 i = 4，到达终点，i = 1 合法。  
对于起点 i = 2：第一次跳跃可以从 i = 2 跳到 i = 3，但是无法进行第二次跳跃。  
对于起点 i = 3：第一次跳跃可以从 i = 3 跳到 i = 4，到达终点，i = 3 合法。  
对于起点 i = 4：无需进行跳跃，已经在数组末尾，i = 4 合法。  
所以共有三个起始下标合法：1，3，4。所以答案为 3。


```python
def odd_even_jumps(self, a: List[int]) -> int:
	N = len(a)

	def make(b):
		ans = [None] * N
		stack = []  # invariant: stack is decreasing
		for i in b:
			while stack and i > stack[-1]:
				ans[stack.pop()] = i
			stack.append(i)
		return ans

	b = sorted(range(N), key = lambda i: a[i])
	oddnext = make(b)
	b.sort(key = lambda i: -a[i])
	evennext = make(b)

	odd = [False] * N
	even = [False] * N
	odd[N-1] = even[N-1] = True

	for i in range(N-2, -1, -1):
		if oddnext[i] is not None:
			odd[i] = even[oddnext[i]]
		if evennext[i] is not None:
			even[i] = odd[evennext[i]]

	return sum(odd)
```
pass