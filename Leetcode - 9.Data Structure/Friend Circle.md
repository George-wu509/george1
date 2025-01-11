
**样例 1:**
```
输入：[[1,1,0],[1,1,0],[0,0,1]]
输出：2
解释：
0号和1号学生是直接朋友，所以他们位于一个朋友圈内。
2号学生自己位于一个朋友圈内。所以返回2.
```
**样例 2:**
```
输入：[[1,1,0],[1,1,1],[0,1,1]]
输出：1
解释：
0号和1号学生是直接朋友，1号和2号学生处于同一个朋友圈。
所以0号和2号是间接朋友。所有人都处于一个朋友圈内，所以返回1。
```


```python
class Solution:
    def dfs(self, x, m, visisted):
        for i in range(len(m)):
            if (m[x][i] == 1 and visisted[i] == False):
                visisted[i] = True
                self.dfs(i, m, visisted)
    def begindfs(self, m):
        # 人数
        n = len(m)
        # 答案
        ans = 0
        # 标记是否访问过
        visisted = {}
        for i in range(n):
            visisted[i] = False
        # 遍历每个人，如果这个人还没访问过 就从这个人开始做一遍dfs
        for i in range(n):
            if (visisted[i] == False):
                ans += 1
                visisted[i] = True
                self.dfs(i, m, visisted)
        return ans
    """
    @param m: a matrix
    @return: the total number of friend circles among all the students
    """
    def find_circle_num(self, m):
        # Write your code here
        ansdfs = self.begindfs(m)
        return ansdfs
```
pass