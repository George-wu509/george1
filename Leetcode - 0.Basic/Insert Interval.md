
**样例 1：**
输入：
```
区间列表 = [(1,2), (5,9)]
新的区间 = (2, 5)
```
输出：
```
[(1,9)]
```
解释：
插入后区间有重叠，需要合并区间。

**样例 2：**
输入：
```
区间列表 = [(1,2), (5,9)]
新的区间 = (3, 4)
```
输出：
```
[(1,2), (3,4), (5,9)]
```
解释：
区间按起始端点有序。




```python
class Solution:
    """
    @param intervals: Sorted interval list.
    @param new_interval: new interval.
    @return: A new interval list.
    """
    def insert(self, intervals, new_interval):
        new_intervals = []
        if len(intervals) == 0:
            new_intervals.append(new_interval)
        for i in range(len(intervals)):
            # 如果新区间的结束值小于区间开始值，插在这里，后面续上
            if new_interval.end < intervals[i].start:
                new_intervals.append(new_interval)
                for j in range(i,len(intervals)):
                    new_intervals.append(intervals[j])
                break
            # 如果新区间的开始值大于区间结束值，把当前区间加进去
            elif new_interval.start > intervals[i].end:
                new_intervals.append(intervals[i]);
            # 出现交叉，需要合并
            else :
                new_interval.start = min(new_interval.start, intervals[i].start);
                new_interval.end = max(new_interval.end, intervals[i].end);
            # 最后只剩一个数据了，添加进去
            if i == len(intervals) - 1:
                new_intervals.append(new_interval)
        return new_intervals
```
pass