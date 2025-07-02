Lintcode 589
给一个图中的`n`个节点, 记为 `1` 到 `n` . 在开始的时候图中没有边。  
你需要完成下面两个方法:

1. `connect(a, b)`, 添加连接节点 `a`, `b` 的边.
2. `query(a, b)`, 检验两个节点是否联通

例1:
```python
"""
输入:
ConnectingGraph(5)
query(1, 2)
connect(1, 2)
query(1, 3) 
connect(2, 4)
query(1, 4) 
输出:
[false,false,true]
```
例2:
```python
"""
输入:
ConnectingGraph(6)
query(1, 2)
query(2, 3)
query(1, 3)
query(5, 6)
query(1, 4)

输出:
[false,false,false,false,false]
```



```python
class ConnectingGraph:

    def __init__(self, n):
        self.father = {}
        for i in range(1, n + 1):
            self.father[i] = i

    def connect(self, a, b):
        self.father[self.find(a)] = self.find(b)

    def query(self, a, b):
        return self.find(a) == self.find(b)

    def find(self, node):
        path = []
        while self.father[node] != node:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node
```
pass

解釋:
step1.  ConnectingGraph(5),   father = {1:1, 2:2, 3:3, 4:4, 5:5}
step2.  query(1, 2), 比較find(a)跟find(b)
	find(a) = 1,  find(b) = 2 
step3. connect(1, 2)
	 father[find(1)] = find(2)    
	 father = {1:2, 2:2, 3:3, 4:4, 5:5}
step4. connect(4,2)
	 father = {1:2, 2:2, 3:3, 4:2, 5:5}




这个 `ConnectingGraph` 类是使用**并查集（Union-Find Set）**数据结构来解决连接图的问题。并查集是一种树形数据结构，用于处理一些不相交集合的合并及查询问题。它的主要思想是，每个集合用一棵树来表示，树的根节点就是这个集合的代表元素。

---

### **核心概念：并查集**

1. **集合（Set）**: 一组不相交的元素。
    
2. **代表元素（Representative/Root）**: 每个集合都会有一个唯一的代表元素，通常是树的根节点。
    
3. **查找（Find）操作**: 给定一个元素，找出它所属集合的代表元素。
    
4. **合并（Union）操作**: 将两个不相交的集合合并为一个集合。
    

---

### **代码解释**

让我们逐个方法来分析这段代码：

#### **`__init__(self, n)`：初始化**

Python

```
    def __init__(self, n):
        self.father = {}
        for i in range(1, n + 1):
            self.father[i] = i
```

- **`self.father = {}`**: 这是一个字典（或者哈希表），用来存储每个节点的“父节点”。在并查集中，`self.father[i]` 表示节点 `i` 的父节点。
    
- **`for i in range(1, n + 1): self.father[i] = i`**: 在初始化时，图中有 `n` 个节点，编号从 `1` 到 `n`。刚开始时，没有任何边，所以每个节点都独立成为一个集合。这意味着每个节点都是它自己集合的代表元素（即它的父节点是它自己）。
    

#### **`connect(self, a, b)`：连接节点**

Python

```
    def connect(self, a, b):
        self.father[self.find(a)] = self.find(b)
```

- 这个方法用来添加连接节点 `a` 和 `b` 的边，实际上是**合并**节点 `a` 所在的集合和节点 `b` 所在的集合。
    
- `self.find(a)`: 找到节点 `a` 所属集合的代表元素（根节点）。
    
- `self.find(b)`: 找到节点 `b` 所属集合的代表元素（根节点）。
    
- `self.father[self.find(a)] = self.find(b)`: 这行代码将节点 `a` 所在集合的根节点的父节点设置为节点 `b` 所在集合的根节点。这样，原本两个独立的集合就被合并成了一个新的集合。
    

#### **`query(self, a, b)`：查询节点是否联通**

Python

```
    def query(self, a, b):
        return self.find(a) == self.find(b)
```

- 这个方法用来检验两个节点 `a` 和 `b` 是否联通。
    
- 在并查集中，如果两个节点属于同一个集合，那么它们就是联通的。而判断它们是否属于同一个集合，只需要检查它们的代表元素（根节点）是否相同。
    
- `self.find(a)`: 找到 `a` 的根节点。
    
- `self.find(b)`: 找到 `b` 的根节点。
    
- 如果 `self.find(a)` 等于 `self.find(b)`，说明它们共享同一个根节点，因此它们联通；否则，它们不联通。
    

#### **`find(self, node)`：查找根节点（带路径压缩）**

Python

```
    def find(self, node):
        path = []
        while self.father[node] != node:
            path.append(node)
            node = self.father[node]
            
        for n in path:
            self.father[n] = node
            
        return node
```

这是并查集最核心也是最巧妙的部分，它实现了**查找**操作，并且带有**路径压缩**优化。

1. **查找根节点**:
    
    - `path = []`: 初始化一个列表 `path`，用来存储从当前 `node` 到根节点路径上的所有节点。
        
    - `while self.father[node] != node:`: 从当前 `node` 开始，不断向上遍历，直到找到它的父节点就是它自己的那个节点——这就是根节点。
        
    - `path.append(node)`: 在遍历过程中，将当前节点加入 `path`。
        
    - `node = self.father[node]`: 移动到当前节点的父节点，继续向上遍历。
        
2. **路径压缩**:
    
    - `for n in path: self.father[n] = node`: 当找到根节点 `node` 后，遍历 `path` 中收集到的所有节点。
        
    - 将这些节点（从原始 `node` 到根节点路径上的所有非根节点）的父节点都直接指向这个**根节点**。
        
    - 这样做的目的是为了优化后续的 `find` 操作。下次再查询 `path` 中的任何一个节点时，它就能直接找到根节点，大大减少了遍历的深度，提高了效率。
        
    - 最后，`return node` 返回找到的根节点。
        

---

### **总结**

这个 `ConnectingGraph` 类通过并查集有效地解决了图的连接和查询问题：

- **`__init__`**: 初始化每个节点为一个独立的集合。
    
- **`connect`**: 合并两个节点所在的集合。
    
- **`query`**: 判断两个节点是否在同一个集合（是否联通）。
    
- **`find`**: 查找节点的代表元素（根节点），并优化了查找路径。
    

并查集在处理这类动态连通性问题时非常高效，尤其是在图的边不断增加且需要频繁查询连通性时。