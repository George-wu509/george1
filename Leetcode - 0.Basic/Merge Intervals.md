

**样例1:**

```
输入: [(1,3)]
输出: [(1,3)]
```

**样例 2:**

```
输入:  [(1,3),(2,6),(8,10),(15,18)]
输出: [(1,6),(8,10),(15,18)]
```

```python
    def merge(self, intervals):
        intervals = sorted(intervals, key=lambda x: x.start)
        result = []
        for interval in intervals:
            if len(result) == 0 or result[-1].end < interval.start:
                result.append(interval)
            else:
                result[-1].end = max(result[-1].end, interval.end)
        return result
```
pass