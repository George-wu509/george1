

```
输入: insert("apple", 3), 输出: Null
输入: sum("ap"), 输出: 3
输入: insert("app", 2), 输出: Null
输入: sum("ap"), 输出: 5
```





```python
class TrieNode:
    def __init__(self):
        self.val = 0
        self.next = [None for _ in range(26)]

class MapSum:
    def __init__(self):
        # write your code here
        self.root = TrieNode()
        self.map = {}
    
    """
    @param key: 
    @param val: 
    @return: nothing
    """
    def insert(self, key, val):
        # write your code here
        delta = val
        if key in self.map:
            delta -= self.map[key]
        self.map[key] = val
        node = self.root
        for c in key:
            if node.next[ord(c) - ord('a')] is None:
                node.next[ord(c) - ord('a')] = TrieNode()
            node = node.next[ord(c) - ord('a')]
            node.val += delta

    """
    @param prefix: 
    @return: nothing
    """
    def sum(self, prefix):
        # write your code here
        node = self.root
        for c in prefix:
            if node.next[ord(c) - ord('a')] is None:
                return 0            
            node = node.next[ord(c) - ord('a')]
        return node.val
```
pass