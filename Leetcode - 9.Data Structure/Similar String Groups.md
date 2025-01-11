
**样例 1:**
```
输入: ["tars","rats","arts","star"]
输出: 2
```
**样例2:**
```
输入: ["omv","ovm"]
输出:1
```


```python
```python
class UnionFind:
2    def __init__(self):
3        self.parent = {}
4        self.num_of_set = 0
5        self.size_of_set = {}
6    
7    def add(self, node):
8        if node in self.parent:
9            return
10        self.parent[node] = node
11        self.num_of_set += 1
12        self.size_of_set[node] = 1
13
14    def find(self, node):
15        root = node
16        while self.parent[root] != root:
17            root = self.parent[root]
18
19        while self.parent[node] != root:
20            temp = self.parent[node]
21            self.parent[node] = root
22            node = temp
23
24        return root
25
26    def union(self, n1, n2):
27        r1 = self.find(n1)
28        r2 = self.find(n2)
29        if r1 != r2:
30            self.parent[r2] = r1
31            self.num_of_set -= 1
32            self.size_of_set[r1] += self.size_of_set[r2]
33
34class Solution:
35    """
36    @param A: a string array
37    @return: the number of groups 
38    """
39    def numSimilarGroups(self, A):
40        # Write your code here
41        if len(A) == 0:
42            return 0
43        n = len(A)
44
45        uf = UnionFind()
46        for word in A:
47            uf.add(word)
48
49        words = list(uf.parent.keys())
50        N = len(words)
51        L = len(words[0])
52
53        complexity1 = N * N * L
54        complexity2 = N * L * L * L
55        
56        if complexity1 < complexity2:
57            # O(N * N * L)
58            for i in range(N - 1):
59                for j in range(i + 1, N):
60                    s1 = words[i]
61                    s2 = words[j]
62                    if uf.find(s1) != uf.find(s2) and self.isSimilar(s1, s2):
63                        uf.union(s1, s2)
64        else:
65            # O(N * L * L * L)
66            for word in words:
67                chars = list(word)
68                for i in range(0, L - 1):
69                    for j in range(i + 1, L):
70                        chars[i], chars[j] = chars[j], chars[i]
71                        wordNew = ''.join(chars)
72                        if word != wordNew and wordNew in uf.parent and uf.find(word) != uf.find(wordNew) and self.isSimilar(word, wordNew):
73                            uf.union(word, wordNew)
74                        chars[i], chars[j] = chars[j], chars[i]
75
76        return uf.num_of_set
77
78    def isSimilar(self, w1, w2):
79        count = 0
80        for i in range(len(w1)):
81            if w1[i] != w2[i]:
82                count += 1
83                if count > 2:
84                    return False
85        return True
```
```
pass