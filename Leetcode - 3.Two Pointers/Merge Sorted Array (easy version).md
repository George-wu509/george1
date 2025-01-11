

**样例 1：**
输入：
```
A = [1,2,3]
m = 3
B = [4,5]
n = 2
```
输出：
```
[1,2,3,4,5]
```
解释：
经过合并新的数组为[1,2,3,4,5]  

**样例 2：**
输入：
```
A = [1,2,5]
m = 3
B = [3,4]
n = 2
```
输出：
```
[1,2,3,4,5]
```
解释：
经过合并新的数组为[1,2,3,4,5]

```python

    def mergeSortedArray(self, A, m, B, n):
        sorted = []
        p1, p2 = 0, 0
        while p1 < m or p2 < n:
            if p1 == m:
                sorted.append(B[p2])
                p2 += 1
            elif p2 == n:
                sorted.append(A[p1])
                p1 += 1
            elif A[p1] < B[p2]:
                sorted.append(A[p1])
                p1 += 1
            else:
                sorted.append(B[p2])
                p2 += 1
        A[:] = sorted
```
pass