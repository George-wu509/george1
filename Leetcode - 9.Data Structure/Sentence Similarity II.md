
**样例 1:**
```
输入:
["7", "5", "4", "11", "13", "15", "19", "12", "0", "10"]
["16", "1", "7", "3", "15", "10", "13", "2", "19", "8"]
[["6", "18"], ["8", "17"], ["1", "13"], ["0", "8"], ["9", "14"], ["11", "17"], ["11", "19"], ["13", "16"], ["0", "18"], ["3", "11"], ["1", "9"], ["2", "11"], ["2", "4"], ["0", "19"], ["8", "12"], ["8", "19"], ["16", "19"], ["1", "11"], ["2", "18"], ["0", "16"], ["7", "11"], ["6", "8"], ["9", "17"], ["8", "16"], ["3", "13"], ["7", "9"], ["7", "10"], ["3", "6"], ["15", "19"], ["1", "5"], ["2", "14"], ["1", "18"], ["8", "15"], ["14", "19"], ["3", "17"], ["6", "10"], ["5", "17"], ["10", "15"], ["1", "10"], ["4", "6"]]
输出:
true
```
**样例 2:**
```
输入:
["great","acting","skills"]
["fine","drama","talent"]
[["great","good"],["fine","good"],["drama","acting"],["skills","talent"]]
输出:
true
```



```python
class Solution:
    def are_sentences_similar_two(self, words1: List[str], words2: List[str], pairs: List[List[str]]) -> bool:
        if len(words1) != len(words2): return False
        import itertools
        index = {}
        count = itertools.count()
        dsu = DSU(2 * len(pairs))
        for pair in pairs:
            for p in pair:
                if p not in index:
                    index[p] = next(count)
            dsu.union(index[pair[0]], index[pair[1]])

        return all(w1 == w2 or
                   w1 in index and w2 in index and
                   dsu.find(index[w1]) == dsu.find(index[w2])
                   for w1, w2 in zip(words1, words2))
class DSU:
    def __init__(self, N):
        self.par = list(range(N))
    def find(self, x):
        if self.par[x] != x:
            self.par[x] = self.find(self.par[x])
        return self.par[x]
    def union(self, x, y):
        self.par[self.find(x)] = self.find(y)
```
pass