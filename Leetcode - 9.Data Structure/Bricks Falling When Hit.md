
**样例 1:**
```
输入: grid = [[1,0,0,0],[1,1,1,0]], hits = [[1,0]]
输出: [2]
解释: 消除 (1, 0) 处的砖块时, 位于 (1, 1) 和 (1, 2) 的砖块会掉落, 所以返回 2.
```
**样例 2:**
```
输入: grid = [[1,0,0,0],[1,1,0,0]], hits = [[1,1],[1,0]]
输出: [0,0]
解释: 当我们消除 (1, 0) 的砖块时, (1, 1) 的砖块已经在前一次操作中被消除了.
```


```python
```python
from typing import (
2    List,
3)
4
5class UnionFind:
6    def __init__(self, n):
7        # 当前结点的父亲结点
8        self.father = {}
9        # 以当前结点为根结点的子树的结点总数
10        self.size = {}
11        for i in range(n):
12            self.father[i] = i
13            self.size[i] = 1
14
15    # 路径压缩，只要求每个不相交集合的「根结点」的子树包含的结点总数数值正确即可
16    # 因此在路径压缩的过程中不用维护数组 size
17    def find(self, x):
18        if x != self.father[x]:
19            self.father[x] = self.find(self.father[x])
20        return self.father[x]
21
22    def union(self, x, y):
23        root_x, root_y = self.find(x), self.find(y)
24        if root_x == root_y:
25            return
26
27        self.father[root_x] = root_y
28        # 在合并的时候维护数组 size
29        self.size[root_y] += self.size[root_x]
30
31    # 在并查集的根结点的子树包含的结点总数
32    def get_size(self, x):
33        return self.size[self.find(x)]
34
35class Solution:
36    def hit_bricks(self, grid: List[List[int]], hits: List[List[int]]) -> List[int]:
37        DIRECTIONS = ((0, 1), (1, 0), (-1, 0), (0, -1))
38        rows = len(grid)
39        cols = len(grid[0])
40
41        # 第 1 步：把 grid 中的砖头全部击碎，通常算法问题不能修改输入数据
42        # 这一步非必需，可以认为是一种答题规范
43        copy = [[grid[i][j] for j in range(len(grid[0]))] for i in range(len(grid))]
44
45        # 把 copy 中的砖头全部击碎
46        for hit in hits:
47            copy[hit[0]][hit[1]] = 0
48
49        # 第 2 步：建图，把砖块和砖块的连接关系输入并查集
50        # size 表示二维网格的大小，也表示虚拟的「屋顶」在并查集中的编号
51        size = rows * cols
52        union_find = UnionFind(size + 1)
53
54        # 将下标为 0 的这一行的砖块与「屋顶」相连
55        for j in range(cols):
56            if copy[0][j] == 1:
57                union_find.union(j, size)
58
59        # 其余网络，如果是砖块向上、向左看一下，如果也是砖块，在并查集中进行合并
60        for i in range(1, rows):
61            for j in range(cols):
62                if copy[i][j] == 1:
63                    # 如果上方也是砖块
64                    if copy[i - 1][j] == 1:
65                        # 二维坐标转换为一维坐标
66                        union_find.union((i - 1) * cols + j, i * cols + j)
67                    # 如果左边也是砖块
68                    if (j > 0 and copy[i][j - 1] == 1):
69                        # 二维坐标转换为一维坐标
70                        union_find.union(i * cols + j - 1, i * cols + j)
71
72        # 第 3 步：按照 hits 的逆序，在 copy 中 补回砖块
73        # 把每一次因为补回砖块而与屋顶相连的砖块的增量记录到 res 数组中
74        hits_len = len(hits)
75        res = [0] * hits_len
76        for i in range(hits_len - 1, - 1, -1):
77            x = hits[i][0]
78            y = hits[i][1]
79
80            # 注意：这里不能用 copy，语义上表示，如果原来在 grid 中，这一块是空白
81            # 这一步不会产生任何砖块掉落
82            # 逆向补回的时候，与屋顶相连的砖块数量也肯定不会增加
83            if grid[x][y] == 0:
84                continue
85
86            # 补回之前与屋顶相连的砖块数
87            origin = union_find.get_size(size)
88
89            # 注意：如果补回的这个结点在第 1 行，要告诉并查集它与屋顶相连（逻辑同第 2 步）
90            if x == 0:
91                union_find.union(y, size)
92
93            # 在 4 个方向上看一下，如果相邻的 4 个方向有砖块，合并它们
94            for direction in DIRECTIONS:
95                new_x = x + direction[0]
96                new_y = y + direction[1]
97
98                # 输入坐标在二维网格中是否越界
99                if (0 <= new_x < rows and 0 <= new_y < cols) and copy[new_x][new_y] == 1:
100                    union_find.union(x * cols + y, new_x * cols + new_y)
101
102            # 补回之后与屋顶相连的砖块数
103            current = union_find.get_size(size)
104            # 减去的 1 是逆向补回的砖块（正向移除的砖块），与 0 比较大小
105            # 是因为存在一种情况，添加当前砖块，不会使得与屋顶连接的砖块数量更多
106            res[i] = max(0, current - origin - 1)
107
108            # 真正补上这个砖块
109            copy[x][y] = 1
110        return res
```
```
pass