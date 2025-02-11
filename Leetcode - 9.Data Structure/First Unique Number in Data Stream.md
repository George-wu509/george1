Lintcode 960
我们需要实现一个叫 `DataStream` 的数据结构。并且这里有 `两` 个方法需要实现：

1. `void add(number)` // 加一个新的数
2. `int firstUnique()` // 返回第一个独特的数

**样例1**
```python
""" 
[1, 2, 2, 1, 3, 4, 4, 5, 6]
5
输出： 3
```
**样例2**
```python
"""
输入：
[1, 2, 2, 1, 3, 4, 4, 5, 6]
7
输出： -1
```
**样例3**
```python
"""
输入：
[1, 2, 2, 1, 3, 4]
3
输出： 3
```


```python
    def first_unique_number(self, nums, number):
        counter = {}
        for num in nums:
            counter[num] = counter.get(num, 0) + 1
            if num == number:
                break
        else:
            return -1
            
        for num in nums:
            if counter[num] == 1:
                return num
            if num == number:
                break

        return -1
```
pass