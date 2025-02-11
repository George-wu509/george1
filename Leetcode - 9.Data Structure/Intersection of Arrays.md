Lintcode 793
给出多个数组，求它们的交集。输出他们交集的大小。


**样例 1:**
```python
"""
输入:  [[1,2,3],[3,4,5],[3,9,10]]
输出:  1
	
解释:
只有 3 出现在三个数组中。
```
**样例 2:**
```python
"""
输入: [[1,2,3,4],[1,2,5,6,7],[9,10,1,5,2,3]]
输出:  2
	
解释:
交集是 [1,2]。
```


```python
    def intersection_of_arrays(self, arrs):
        count = {}
        # 记录每个数的出现次数
        for arr in arrs:
            for x in arr:
                if x not in count:
                    count[x] = 0
                count[x] += 1
        
        # 某个数出现次数等于数组个数，代表它在所有数组中都出现过
        result = 0
        for x in count.keys():
            if count[x] == len(arrs):
                result += 1
        return result
```
pass