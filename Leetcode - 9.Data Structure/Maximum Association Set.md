Lintcode 805
亚麻卖书，每本书都有与其关联性很强的书，给出ListA与ListB，表示`ListA[i]`与`ListB[i]`有关联，输出互相关联的最大集合。(输出任意顺序)，题目保证只有一个最大的集合。


```python
"""
样例 1:
	输入:  ListA = ["abc","abc","abc"], ListB = ["bcd","acd","def"]
	输出:  ["abc","acd","bcd","def"]
	解释:
	"abc" 和其他书均有关联，全集就是最大集合。
	
样例 2:
	输入:  ListA = ["a","b","d","e","f"], ListB = ["b","c","e","g","g"]
	输出:  ["d","e","f","g"]
	解释:
	关联的集合有 [a, b, c] 和 [d, e, g, f], 最大的是 [d, e, g, f]
```


```python
    def maximum_association_set(self, list_a, list_b):
        # Write your code here
        self.father = {}
        for i in range(len(list_a)):
            if list_a[i] not in self.father:
                self.father[list_a[i]] = list_a[i]
            if list_b[i] not in self.father:
                self.father[list_b[i]] = list_b[i]
            self.union(list_a[i], list_b[i])
           
        result = {} #key: root of the group, values: the book under the same father
        for key in self.father:
            root = self.find(key)
            if root not in result:
                result[root] = []
                
            result[root].append(key)
        
        max_size = 0
        for key in result:
            if max_size < len(result[key]):
                max_size = len(result[key])
                max_key = key 
        
        return result[max_key]
        
    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a != root_b:
            self.father[root_a] = root_b
            
    def find(self, x):
        j = x 
        while self.father[j] != j:
            j = self.father[j]
        while self.father[x] != j:
            fx = self.father[x]
            self.father[x] = j 
            x = fx 
        return j
```
pass