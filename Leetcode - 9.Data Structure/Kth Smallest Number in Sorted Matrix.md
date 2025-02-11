Lintcode


**样例 1:**
```python
"""
输入:
[
  [1 ,5 ,7],
  [3 ,7 ,8],
  [4 ,8 ,9],
]
k = 4
输出: 5
```
**样例 2:**
```python
"""
输入: 
[
  [1, 2],
  [3, 4]
]
k = 3
输出: 3
```


```python
    def kth_smallest(self, matrix: List[List[int]], k: int) -> int:
        nums=[]
        doing=0
        while doing<len(matrix):
            did=0
            while did<len(matrix[doing]):
                nums.append(matrix[doing][did])
                did+=1
            doing+=1
        nums.sort()
        return nums[k-1]
```
pass