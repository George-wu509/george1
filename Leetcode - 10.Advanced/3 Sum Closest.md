

**样例 1：**
输入：
```
numbers = [2,7,11,15]
target = 3
```
输出：
```
20
```
解释：
2+7+11=20  

**样例 2：**
输入：
```
numbers = [-1,2,1,-4]
target = 1
```
输出：
```
2
```
解释：
-1+2+1=2


```python
    def three_sum_closest(self, numbers, target):
        numbers.sort()
        ans = None
        for i in range(len(numbers)):
            left, right = i + 1, len(numbers) - 1
            while left < right:
                sum = numbers[left] + numbers[right] + numbers[i]
                if ans is None or abs(sum - target) < abs(ans - target):
                    ans = sum
                    
                if sum <= target:
                    left += 1
                else:
                    right -= 1
        return ans
```
pass