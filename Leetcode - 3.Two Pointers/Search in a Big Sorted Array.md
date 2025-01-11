
**样例 1:**
```
输入: [1, 3, 6, 9, 21, ...], target = 3
输出: 1
```
**样例 2:**
```
输入: [1, 3, 6, 9, 21, ...], target = 4
输出: -1
```


```python
    def searchBigSortedArray(self, reader, target):
        kth = 1
        while reader.get(kth - 1) < target:
            kth = kth * 2
                
        # start 也可以是 kth // 2，但是我习惯比较保守的写法
        # 因为写为 0 也不会影响时间复杂度
        start, end = 0, kth - 1
        while start + 1 < end:
            mid = start + (end - start) // 2
            if reader.get(mid) < target:
                start = mid
            else:
                end = mid
        if reader.get(start) == target:
            return start
        if reader.get(end) == target:
            return end
        return -1
```
pass