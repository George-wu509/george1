
**样例 1：**
输入：
```
LFUCache(3)
set(2,2)
set(1,1)
get(2)
get(1)
get(2)
set(3,3)
set(4,4)
get(3)
get(2)
get(1)
get(4)
```
输出：
```
2
1
2
-1
2
1
4
```
解释：
在set（4，4）时，由于值为3的缓存没有被使用，所以将其替换掉。



```python
```python
class Node:
2    def __init__(self, key, val, pre=None, nex=None, freq=0):
3        self.pre = pre
4        self.nex = nex
5        self.freq = freq
6        self.val = val
7        self.key = key
8        
9    def insert(self, nex):
10        nex.pre = self
11        nex.nex = self.nex
12        self.nex.pre = nex
13        self.nex = nex
14    
15def create_linked_list():
16    head = Node(0, 0)
17    tail = Node(0, 0)
18    head.nex = tail
19    tail.pre = head
20    return (head, tail)
21
22class LFUCache:
23    def __init__(self, capacity: int):
24        self.capacity = capacity
25        self.size = 0
26        self.minFreq = 0
27        self.freqMap = collections.defaultdict(create_linked_list)
28        self.keyMap = {}
29
30    def delete(self, node):
31        if node.pre:
32            node.pre.nex = node.nex
33            node.nex.pre = node.pre
34            if node.pre is self.freqMap[node.freq][0] and node.nex is self.freqMap[node.freq][-1]:
35                self.freqMap.pop(node.freq)
36        return node.key
37        
38    def increase(self, node):
39        node.freq += 1
40        self.delete(node)
41        self.freqMap[node.freq][-1].pre.insert(node)
42        if node.freq == 1:
43            self.minFreq = 1
44        elif self.minFreq == node.freq - 1:
45            head, tail = self.freqMap[node.freq - 1]
46            if head.nex is tail:
47                self.minFreq = node.freq
48
49    def get(self, key: int) -> int:
50        if key in self.keyMap:
51            self.increase(self.keyMap[key])
52            return self.keyMap[key].val
53        return -1
54
55    def set(self, key: int, value: int) -> None:
56        if self.capacity != 0:
57            if key in self.keyMap:
58                node = self.keyMap[key]
59                node.val = value
60            else:
61                node = Node(key, value)
62                self.keyMap[key] = node
63                self.size += 1
64            if self.size > self.capacity:
65                self.size -= 1
66                deleted = self.delete(self.freqMap[self.minFreq][0].nex)
67                self.keyMap.pop(deleted)
68            self.increase(node)
```
```
pass