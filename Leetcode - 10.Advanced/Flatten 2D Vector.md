
例1:
```
输入:[[1,2],[3],[4,5,6]]
输出:[1,2,3,4,5,6]
```
例2:
```
输入:[[7,9],[5]]
输出:[7,9,5]
```



```python
```python
class Vector2D(object):
2
3    # @param vec2d {List[List[int]]}
4    def __init__(self, vec2d):
5        self.vec2d = vec2d
6        self.row, self.col = 0, -1
7        self.next_elem = None
8
9    # @return {int} a next element
10    def next(self):
11        if self.next_elem is None:
12            self.hasNext()
13            
14        temp, self.next_elem = self.next_elem, None
15        return temp
16
17    # @return {boolean} true if it has next element
18    # or false
19    def hasNext(self):
20        if self.next_elem:
21            return True
22        
23        self.col += 1
24        while self.row < len(self.vec2d) and self.col >= len(self.vec2d[self.row]):
25            self.row += 1
26            self.col = 0
27            
28        if self.row < len(self.vec2d) and self.col < len(self.vec2d[self.row]):
29            self.next_elem = self.vec2d[self.row][self.col]
30            return True
31            
32        return False
```
```
pass