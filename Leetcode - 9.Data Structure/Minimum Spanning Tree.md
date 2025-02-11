Lintcode 629
给定一个Connections，即Connection类（边缘两端的城市名称和它们之间的开销），找到可以连接所有城市并且花费最少的边缘。  
如果可以连接所有城市，则返回连接方法。  
如果城市无法全部连通，则返回空列表。


**样例 1:**
```python
"""
输入:
["Acity","Bcity",1]
["Acity","Ccity",2]
["Bcity","Ccity",3]
输出:
["Acity","Bcity",1]
["Acity","Ccity",2]
```
**样例 2:**
```python
"""
输入:
["Acity","Bcity",2]
["Bcity","Dcity",5]
["Acity","Dcity",4]
["Ccity","Ecity",1]
输出:
[]

解释:
没有办法连通
```


```python
import functools

def comp(a, b):
    if a.cost != b.cost:
        return a.cost - b.cost
    
    if a.city1 != b.city1:
        if a.city1 < b.city1:
            return -1
        else:
            return 1

    if a.city2 == b.city2:
        return 0
    elif a.city2 < b.city2:
        return -1
    else:
        return 1

class Solution:
    # @param {Connection[]} connections given a list of connections include two cities and cost
    # @return {Connection[]} a list of connections from results
    def lowestCost(self, connections):
        # Write your code here
        connections.sort(key=functools.cmp_to_key(comp))
        hash = {}   
        n = 0
        for connection in connections:
            if connection.city1 not in hash:
                n += 1
                hash[connection.city1] = n
            
            if connection.city2 not in hash:
                n += 1
                hash[connection.city2] = n

        father = [0 for _ in range(n + 1)] 

        results = []
        for connection in connections:
            num1 = hash[connection.city1]
            num2 = hash[connection.city2]

            root1 = self.find(num1, father)
            root2 = self.find(num2, father)
            if root1 != root2:
                father[root1] = root2
                results.append(connection)

        if len(results)!= n - 1:
            return []
        return results
    
    def find(self, num, father):
        if father[num] == 0:
            return num
        father[num] = self.find(father[num], father)
        return father[num]
```
pass