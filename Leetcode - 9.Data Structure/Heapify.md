
_**样例 1**_

```
输入 : [3,2,1,4,5]
输出 : [1,2,3,4,5]
解释 : 返回任何一个合法的堆数组，因此 [1,3,2,4,5] 也是一个正确的结果
```



```python
class Solution:

    def heapify(self, A: List[int]):
        for i in range(len(A) // 2, -1, -1):
            self.siftdown(A, i)
            
    def siftdown(self, A, index):
        n = len(A)
        while index < n:
            left = index * 2 + 1
            right = index * 2 + 2
            minIndex = index
            if left < n and A[left] < A[minIndex]:
                minIndex = left
            if right < n and A[right] < A[minIndex]:
                minIndex = right

            if minIndex == index:
                break
            
            A[minIndex], A[index] = A[index], A[minIndex]
            index = minIndex
```
pass