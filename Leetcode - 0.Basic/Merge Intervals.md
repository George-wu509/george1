Lintcode 156
我们以一个 `Interval` 类型的列表 `intervals` 来表示若干个区间的集合，其中单个区间为 `(start, end)`。你需要**合并所有重叠的区间**，并返回一个**不重叠的区间数组**，该数组需**恰好覆盖输入中的所有区间**。

**样例1:**

```python
#输入: [(1,3)]
#输出: [(1,3)]
```

**样例 2:**

```python
#输入:  [(1,3),(2,6),(8,10),(15,18)]
#输出: [(1,6),(8,10),(15,18)]
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