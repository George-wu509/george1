Lintcode 6
将按升序排序的整数数组A和B合并，新数组也需有序。

**样例 1：**
输入：
```python
A = [1]
B = [1]
```
输出：
```python
[1,1]
```
解释：
返回合并后的数组。

**样例 2：**
输入：
```python
A = [1,2,3,4]
B = [2,4,5,6]
```
输出：
```python
[1,2,2,3,4,4,5,6]
```


```python
    def merge_sorted_array(self, a, b):
        i, j = 0, 0
        C = []
        while i < len(a) and j < len(b):
            if a[i] < b[j]:
                C.append(a[i])
                i += 1
            else:
                C.append(b[j])
                j += 1
        while i < len(a):
            C.append(a[i])
            i += 1
        while j < len(b):
            C.append(b[j])
            j += 1
            
        return C
```
pass