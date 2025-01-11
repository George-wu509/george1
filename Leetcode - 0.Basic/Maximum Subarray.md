

**样例 1：**
输入：
```
nums = [−2,2,−3,4,−1,2,1,−5,3]
```
输：
```
6
```
解释：
符合要求的子数组为[4,−1,2,1]，其最大和为 6。

**样例 2：**
输入：
```
nums = [1,2,3,4]
```
输出：
```
10
```
解释：
符合要求的子数组为[1,2,3,4]，其最大和为 10。



```python
class Solution:
    """
    @param nums: A list of integers
    @return: A integer indicate the sum of max subarray
    """
    def max_sub_array(self, nums: List[int]) -> int:
        #prefix_sum记录前i个数的和，max_Sum记录全局最大值，min_Sum记录前i个数中0-k的最小值
        min_sum, max_sum = 0, -sys.maxsize
        prefix_sum = 0
        
        for num in nums:
            prefix_sum += num
            max_sum = max(max_sum, prefix_sum - min_sum)
            min_sum = min(min_sum, prefix_sum)
            
        return max_sum
```
pass