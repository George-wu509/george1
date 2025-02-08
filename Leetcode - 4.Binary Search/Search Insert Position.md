
**样例 1：**
输入：
```
数组 = [1,3,5,6]
target = 5
```
输出：
```
2
```
解释：
5在数组中索引为2。

**样例 2：**
输入：
```
数组 = [1,3,5,6]
target = 2
```
输出：
```
1
```
解释：
2应该被插到索引为1的位置。

**样例 3：**
输入：
```
数组 = [1,3,5,6]
target = 7
```
输出：
```
4
```
解释：
7应该被插到索引为4的位置。

**样例 4：**
输入：
```
数组 = [1,3,5,6]
target = 0
```
输出：
```
0
```
解释：
0应该被插到索引为0的位置。


```python
    def search_insert(self, a, target):
        if len(a) == 0:
            return 0
            
        start, end = 0, len(a) - 1
        # first position >= target
        while start + 1 < end:
            mid = (start + end) // 2
            if a[mid] >= target:
                end = mid
            else:
                start = mid
        
        if a[start] >= target:
            return start
        if a[end] >= target:
            return end
        return len(a)
```
pass