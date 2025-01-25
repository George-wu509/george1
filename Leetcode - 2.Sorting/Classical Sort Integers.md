
```python
样例  1:
	输入:  [3, 2, 1, 4, 5]
	输出:  [1, 2, 3, 4, 5]
	
	样例解释: 
	返回排序后的数组。

样例 2:
	输入:  [1, 1, 2, 1, 1]
	输出:  [1, 1, 1, 1, 2]
	
	样例解释: 
	返回排好序的数组。
```


```python
class Solution:
    def sort_integers(self, a):
        if a == None or len(a) == 0:
            return
        self.quickSort(a, 0, len(a) - 1)
        
    def quickSort(self, a, start, end):
        if start >= end:
            return
        
        left, right = start, end
        pivot = a[(start+end)//2]
        
        while left <= right:
            while left <= right and a[left] < pivot:
                left += 1
            while left <= right and a[right] > pivot:
                right -= 1
                
            if left <= right:
                a[left], a[right] = a[right], a[left]
                left += 1
                right -= 1
                
        self.quickSort(a, start, right)
        self.quickSort(a, left, end)
```
pass