Lintcode 1178
给定一个字符串表示学生出勤记录，记录仅由下列三个字符组成：

- **'A'** : 缺席（Absent）.
- **'L'** : 迟到（Late）.
- **'P'** : 到场（Present）.

如果学生的出勤情况不包含 **超过一个'A'(缺席)** 或者 **超过连续两个'L'(迟到)** ，那么他就应该受到奖励。
返回该学生是否会受到奖励。

**样例 1:**
```python
"""
输入: "PPALLP"
输出: True
```
**样例 2:**
```python
"""
输入: "PPALLL"
输出: False
```


```python
    def check_record(self, s: str) -> bool:
        absents = lates = 0
        for i, c in enumerate(s):
            if c == "A":
                absents += 1
                if absents >= 2:
                    return False
            if c == "L":
                lates += 1
                if lates >= 3:
                    return False
            else:
                lates = 0
        
        return True
```
pass