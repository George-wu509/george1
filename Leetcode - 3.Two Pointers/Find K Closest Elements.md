

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
    def k_closest_numbers(self, a, target, k):
        # 找到 a[left] < target, a[right] >= target
        # 也就是最接近 target 的两个数，他们肯定是相邻的
        right = self.findUpperClosest(a, target)
        left = right - 1
    
        # 两根指针从中间往两边扩展，依次找到最接近的 k 个数
        results = []
        for _ in range(k):
            if self.isLeftCloser(a, target, left, right):
                results.append(a[left])
                left -= 1
            else:
                results.append(a[right])
                right += 1
        
        return results
    
    def findUpperClosest(self, a, target):
        # find the first number >= target in a
        start, end = 0, len(a) - 1
        while start + 1 < end:
            mid = (start + end) // 2
            if a[mid] >= target:
                end = mid
            else:
                start = mid
        
        if a[start] >= target:
            return start
        
        if a[end] >= target:            
            return end
        
        # 找不到的情况
        return len(a)
        
    def isLeftCloser(self, a, target, left, right):
        if left < 0:
            return False
        if right >= len(a):
            return True
        return target - a[left] <= a[right] - target
```
pass