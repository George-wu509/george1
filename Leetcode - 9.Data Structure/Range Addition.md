Lintcode 903
假设你有一个长度为`n`的数组，数组的所有元素初始化为`0`，并且给定`k`个更新操作。

每个更新操作表示为一个三元组：`[startIndex, endIndex, inc]`。这个更新操作给子数组 `A[startIndex ... endIndex]`（包括startIndex和endIndex）中的每一个元素增加 `inc`。

返回执行`k`个更新操作后的新数组。


```python
"""
给定：
长度 = 5,
更新操作 = 
[
[1,  3,  2],
[2,  4,  3],
[0,  2, -2]
]
返回 [-2, 0, 3, 5, 3]

解释:
初始状态：
[ 0, 0, 0, 0, 0 ]
完成 [1, 3, 2]操作后：
[ 0, 2, 2, 2, 0 ]
完成[2, 4, 3]操作后：
[ 0, 2, 5, 5, 3 ]
完成[0, 2, -2]操作后：
[-2, 0, 3, 5, 3 ]
```


```python
    def get_modified_array(self, length, updates):
        
        result = [0 for i in range(length)]
        operation = result + [0]
        # O(k) - k operations
        for start, end, val in updates :
            operation[start] += val
            operation[end + 1] -= val
        # O(n)
        for index in range(len(result)) :
            if index == 0 :
                result[index] = operation[index]
                continue
            result[index] = operation[index] + result[index - 1]
         
        return result 
```
pass