Lintcode 30
给出一个无重叠的按照区间起始端点排序的区间列表。  
在列表中插入一个新的区间，你要确保列表中的区间仍然有序且不重叠（如果有必要的话，可以合并区间）。

**样例 1：**
输入：
```python
#输入: intervals = [(1,2),(3,5),(6,7),(8,10),(12,16)], 
#newInterval = (4,9) 
#输出: [(1,2),(3,10),(12,16)]
```
输出：
```python
#[(1,9)]
```
解释：
插入后区间有重叠，需要合并区间。

**样例 2：**
输入：
```python
#区间列表 = [(1,2), (5,9)]
#新的区间 = (3, 4)
```
输出：
```python
#[(1,2), (3,4), (5,9)]
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


输入: intervals = [(1,2),(3,5),(6,7),(8,10),(12,16)], newInterval = (4,9) 
输出: [(1,2),(3,10),(12,16)]
解釋:
step1. 從intervals的每個小區間開始譬如(1,2), 每個居間的頭尾跟newInterval的頭尾比較, 在判斷如何插入

Time complexity should be O(n)