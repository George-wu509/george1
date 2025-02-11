Lintcode 153
给定一个数组 `num` 和一个整数 `target`. 找到 `num` 中所有的数字之和为 `target` 的组合.


**样例 1:**
```python
"""
输入: num = [7,1,2,5,1,6,10], target = 8
输出: [[1,1,6],[1,2,5],[1,7],[2,6]]
```
**样例 2:**
```python
"""
输入: num = [1,1,1], target = 2
输出: [[1,1]]
解释: 解集不能包含重复的组合
```


```python
    def combination_sum2(self, num, target): 
        # write your code here
        num.sort()        
        self.ans, tmp, use = [], [], [0] * len(num)        
        self.dfs(num, target, 0, 0, tmp, use)        
        return self.ans    
    def dfs(self, can, target, p, now, tmp, use):        
        if now == target:            
            self.ans.append(tmp[:])            
            return        
        for i in range(p, len(can)):            
            if now + can[i] <= target and (i == 0 or can[i] != can[i-1] or use[i-1] == 1):                
                tmp.append(can[i])
                use[i] = 1                
                self.dfs(can, target, i+1, now + can[i], tmp, use)                
                tmp.pop()                
                use[i] = 0
```
pass