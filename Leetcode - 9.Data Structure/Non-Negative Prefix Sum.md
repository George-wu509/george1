
Lintcode 3542

给定一个环形整数数组 `nums`，其长度为 `n`。对于所有下标 i0,i1,...,in−1i0​,i1​,...,in−1​，以每个下标为起始位置遍历所有数组元素，计算该环形数组的**前缀和**，请问有多少个下标可以使得该**前缀和**的所有元素均为**非负数**。

**样例 1:**
```python
输入：
nums = [-3,5,1,2]
输出：
2
解释：
[-3,5,1,2] -> 前缀和为 [-3,2,3,5]
[5,1,2,-3] -> 前缀和为 [5,6,8,5]
[1,2,-3,5] -> 前缀和为 [1,3,0,5]
[2,-3,5,1] -> 前缀和为 [2,-1,4,5]
```

**样例 2:**
```python
输入：
nums = [1,-1,1,-1,1,-1]
输出：
3
解释：
[1,-1,1,-1,1,-1] -> 前缀和为 [1,0,1,0,1,0]
[-1,1,-1,1,-1,1] -> 前缀和为 [-1,0,-1,0,-1,0]
```


```python

class Solution:
    """
    @param nums: 
    @return: count of non-negative prefix sum
    """
    def get_prefix_sum(self, nums: List[int]) -> int:
        # write your code here
        n = len(nums)
        ans = 0
        prefix = [0] * (2 * n)
        q = collections.deque()
        for i in range(1, 2 * n):
            prefix[i] = prefix[i - 1] + nums[(i - 1) % n]

        for i, j in zip(range(-n + 1, n + 1), range(2 * n)):
            if i > 0 and q[0] == prefix[i - 1]:
                q.popleft()
            while q and q[-1] > prefix[j]:
                q.pop()

            q.append(prefix[j])

            if i > 0:
                if q[0] - prefix[i - 1] >= 0:
                    ans += 1
        return ans
```
pass



#### 方法：单调队列

首先对于环形数组，我们可以做一次预处理，来**让一个线性数组拥有与环形数组相同的处理逻辑**，对于本题，我们可以将 `nums` 复制一份并接在给定的 `nums` 之后，但考虑到不处理输入数据，我们也可以重新构造一个长度为 `2 * nums.length` 的数组用于处理。

![13.jpg](https://media-lc.lintcode.com/u_394541/202304/3c4bd6e2585047d3aa970c6c8a0f0945/13.jpg)

这样，我们就能根据新的数组下标 i=0,1,...,2n−1i=0,1,...,2n−1 来做进一步处理，而不必再执行取余操作，同时也可以看到我们只会使用 i=0,1,...,2n−2i=0,1,...,2n−2 的值，因为 nums[n],nums[n+1],...,nums[2n−1]nums[n],nums[n+1],...,nums[2n−1] 与原数组 nums[0],nums[1],...,nums[n−1]nums[0],nums[1],...,nums[n−1] 相同。

我们对处理后的 `nums` 数组做前缀和处理：

![16.png](https://media-lc.lintcode.com/u_394541/202304/c18cff0c54874324b3ea0b37e2bd49f2/16.png)

可以看到，对于每一个下标的前缀和，都有以下性质：

new_prefix[i][x−1]=prefix[i+x]−prefix[i](1≤x≤n,0≤i<n)new_prefix[i][x−1]=prefix[i+x]−prefix[i](1≤x≤n,0≤i<n)

因此我们可以维护一个单调队列，**每次单调队列内存放 prefix 的 4 个元素，每次右侧新元素入队，根据单调队列的性质记录队列中的最小值，并让左侧元素出队，并检测队列中的最小值减去出队元素是否为负数。**

这样，我们就能动态地记录连续范围内的最小值，并且通过这个最小值来判断弹出的元素是否会导致变换后的前缀和是否存在负数。