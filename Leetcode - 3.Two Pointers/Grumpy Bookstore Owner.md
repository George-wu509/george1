

**样例 1:**
```
输入:
[1,0,1,2,1,1,7,5]
[0,1,0,1,0,1,0,1]
3
输出: 
16
解释: 
书店老板在最后3天保持好脾气。
感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.
```


```python
    def max_satisfied(self, customers: List[int], grumpy: List[int], x: int) -> int:
        n = len(customers)
        not_satisfied = customers[:]
        ans = 0
        for i in range(n):
            not_satisfied[i] = customers[i] * grumpy[i]
            ans += customers[i] * (1 - grumpy[i])
        now_sum = 0
        left = 0
        right = x - 1
                
        for i in range(right + 1):
            now_sum += not_satisfied[i]
        
        maximum = now_sum
        
        while right + 1 < n:
            now_sum -= not_satisfied[left]
            left += 1 
            right += 1
            now_sum += not_satisfied[right]
            
            
            maximum = max(maximum, now_sum)
        
        ans += maximum
        return ans
```
pass