
**样例1**
```
输入: intervals = [(0,30),(5,10),(15,20)]
输出: 2
解释:
需要两个会议室
会议室1:(0,30)
会议室2:(5,10),(15,20)
```
**样例2**
```
输入: intervals = [(2,7)]
输出: 1
解释:
只需要1个会议室就够了
```



```python
class Solution:

    def min_meeting_rooms(self, intervals: List[Interval]) -> int:
        points = []
        for interval in intervals:
            points.append((interval.start, 1))
            points.append((interval.end, -1))
            
        meeting_rooms = 0
        ongoing_meetings = 0
        for _, delta in sorted(points):
            ongoing_meetings += delta
            meeting_rooms = max(meeting_rooms, ongoing_meetings)
            
        return meeting_rooms
```
pass