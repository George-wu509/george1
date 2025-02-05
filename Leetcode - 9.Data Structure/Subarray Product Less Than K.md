
```python
样例 1:
	输入:  nums = [10, 5, 2, 6], k = 100
	输出:  8
	
	解释:
	这8个子段是: [10], [5], [2], [6], [10, 5], [5, 2], [2, 6], [5, 2, 6].
	[10, 5, 2] 没有严格小于100所以不算。

	
解释 2:
	输入: nums = [5,10,2], k = 10
	输出:  2
	
	解释:
	只有 [5] 和 [2].
```


```python
    def num_subarray_product_less_than_k(self, nums: List[int], k: int) -> int:
        ans, prod, i = 0, 1, 0
        for j, num in enumerate(nums):
            prod *= num
            while i <= j and prod >= k:
                prod //= nums[i]
                i += 1
            ans += j - i + 1
        return ans
```
pass