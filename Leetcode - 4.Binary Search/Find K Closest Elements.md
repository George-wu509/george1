Lintcode 460
在升序排列A中找与target最接近的k个整数

**样例 1:**
```
输入: A = [1, 2, 3], target = 2, k = 3
输出: [2, 1, 3]
```
**样例 2:**
```
输入: A = [1, 4, 6, 8], target = 3, k = 3
输出: [4, 1, 6]
```

```python
class Solution:
    def k_closest_numbers(self, a, target, k):
        if not a or k == 0:
            return []
        n = len(a)
        left, right = 0, n-1
        results = []
        while left+1 < right:
            mid = left + (right-left)//2
            if a[mid] > target:
                right = mid
            else:
                left = mid
                          
        while len(results) < k:
            if left < 0:
                results.append(a[right])
                right += 1
            elif right >= n:
                results.append(a[left])
                left -= 1
            else:
                if abs(a[left] - target) <= abs(a[right] - target):
                    results.append(a[left])
                    left -= 1
                else:
                    results.append(a[right])
                    right += 1
        
        return results
```
pass