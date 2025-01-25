1849
有一个书店，在接下来的n天中的第i天会有customer[i]个顾客到来，并且在这一天结束后离开。但是书店老板的脾气时好时坏，我们用一个数组grumpy表示他每一天的脾气好坏，若grumpy[i]=1, 则表示第i天老板的脾气很不好；若grumpy[i]=0, 则表示第i天老板的脾气很好。

若某一天书店老板的脾气不好，则会导致所有当天来的所有顾客会给书店差评。但如果某一天脾气好，那么当天所有顾客都会给书店好评。老板想要尽量增加给书店好评的人数数量，想了一个方法。他可以保持连续XX天的好脾气。但这个方法只能使用一次。

那么在这n天这个书店最多能有多少人离开时给书店好评？

**样例 1:**
```python
输入:
customer = [1,0,1,2,1,1,7,5]
grumpy   = [0,1,0,1,0,1,0,1]
x = 3
输出: 
16
解释: 
书店老板在最后3天保持好脾气。
感到满意的最大客户数量 = 1 + 1 + 1 + 1 + 7 + 5 = 16.
```


```python
    def max_satisfied(customers, grumpy, x):
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