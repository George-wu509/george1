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

## 算法

并查集

### 算法分析

- 题目给出两个书籍名字列表，两个列表中相同位置的书籍代表其相互关联。题目要求找到书籍之间互相关联的最大集合。
    
- 这道题的本质，其实是一个不相交集合的合并和查询问题。对于这一种问题，普通的数据结构不能很好地兼顾时间复杂度和空间复杂度。
    
- 面对这一种问题，我们可以选择并查集这种数据结构。其查找和合并功能，能很好地解决这个问题。
    

### 算法步骤

1. 数据预处理：遍历两个列表，使用hash将每个书名与一个唯一整数关联。

2. 初始化并查集数组：f[i]=if[i]=i,即每本书籍与自己相关联。

3. 并查集合并操作：对两个列表中相同位置的书籍，调用findfind函数后更新并查集数组。

4. 并查集查找操作：统计并查集中出现次数最多数字，即最大的关联集合。

5. 根据数字重新转换为书籍名字，生成书籍名字列表后输出。

## 复杂度分析

- 时间复杂度：近似为O(n)O(n), nn为书籍名字个数
    
    - 数据预处理和最后重新转换为字符串的时间复杂度均为O(n)O(n)，并查集的时间复杂度O(n⋅Alpha(n))O(n⋅Alpha(n))（Alpha是Ackerman函数的某个反函数）
- 空间复杂度：O(n)O(n), nn为书籍名字个数
    
    - 需要将字符串和数字相互转换的两个hashmaphashmap字典以及一个并查集数组。长度均为书籍名字个数。