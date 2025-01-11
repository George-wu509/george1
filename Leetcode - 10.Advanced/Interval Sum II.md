

**样例1**
```plain
输入:
[1,2,7,8,5]
[query(0,2),modify(0,4),query(0,1),modify(2,1),query(2,4)]
输出: [10,6,14]
说明:
给定数组 A = [1,2,7,8,5].
在query(0, 2)后, 1 + 2 + 7 = 10,
在modify(0, 4)后, 将 A[0] 修改为 4， A = [4,2,7,8,5].
在query(0, 1)后, 4 + 2 = 6.
在modify(2, 1)后, 将 A[2] 修改为 1，A = [4,2,1,8,5].
After query(2, 4), 1 + 8 + 5 = 14.
```
**样例2**
```plain
输入:
[1,2,3,4,5]
[query(0,0),query(1,2),quert(3,4)]
输出: [1,5,9]
说明:
1 = 1
2 + 3 = 5
4 + 5 = 9
```



```python
class Solution:
    def __init__(self, A):
        self.n = len(A)
        self.A = A
        self.s = [0] * (4 * len(A))
        self.build_tree(0, 0, len(A) - 1)

    def build_tree(self, s_index, left, right):
        if left > right: return
        if left == right:
            self.s[s_index] = self.A[left]
            return
        mid = (left + right) // 2
        self.build_tree(s_index * 2 + 1, left, mid)
        self.build_tree(s_index * 2 + 2, mid + 1, right)
        self.s[s_index] = self.s[s_index * 2 + 1] + self.s[s_index * 2 + 2]

    def query(self, start, end):
        return self.query_tree(0, 0, self.n - 1, start, end)

    def query_tree(self, s_index, left, right, start, end):
        if left == start and right == end:
            return self.s[s_index]
        mid = (left + right) // 2
        # 完全在左子树
        if end <= mid:
            return self.query_tree(s_index * 2 + 1, left, mid, start, end)
        # 完全在右子树
        if start >= mid + 1:
            return self.query_tree(s_index * 2 + 2, mid + 1, right, start, end)
        # 被 mid 劈成了两个区间
        left_sum = self.query_tree(s_index * 2 + 1, left, mid, start, mid)
        right_sum = self.query_tree(s_index * 2 + 2, mid + 1, right, mid + 1, end)
        return left_sum + right_sum

    def modify(self, index, value):
        self.modify_tree(0, 0, self.n - 1, index, value)
    
    def modify_tree(self, s_index, left, right, index, value):
        if left == right:
            self.s[s_index] = value
            return
        mid = (left + right) // 2
        if index <= mid:
            self.modify_tree(s_index * 2 + 1, left, mid, index, value)
        else:
            self.modify_tree(s_index * 2 + 2, mid + 1, right, index, value)
        self.s[s_index] = self.s[s_index * 2 + 1] + self.s[s_index * 2 + 2]
```
pass