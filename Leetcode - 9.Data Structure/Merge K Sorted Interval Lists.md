
**样例1**
```
输入: [
  [(1,3),(4,7),(6,8)],
  [(1,2),(9,10)]
]
输出: [(1,3),(4,8),(9,10)]
```
**样例2**
```
输入: [
  [(1,2),(5,6)],
  [(3,4),(7,8)]
]
输出: [(1,2),(3,4),(5,6),(7,8)]
```





```python
    def mergeKSortedIntervalLists(self, intervals):
        arr = []
        for i in intervals:
            for j in i:
                arr.append(j)
        arr = sorted(arr, key=lambda o: o.start)
        ans = []
        if (len(arr) == 0) :
            return ans 
        ans.append(arr[0])
        for i in range(1, len(arr)):
            if (ans[len(ans) - 1].end >= arr[i].start):
                ans[len(ans) - 1].end = max(ans[len(ans) - 1].end, arr[i].end)
            else :
                ans.append(arr[i])
        return ans
```
pass