

**样例1**
```plain
输入: [1, 2, 3, 4]
输出: 7
说明:
子数组[3, 4]有最大的异或值
```
**样例2**
```plain
输入: [8, 1, 2, 12, 7, 6]
输出: 15
说明:
子数组[1, 2, 12]有最大的异或值
```
**样例3**
```plain
输入: [4, 6]
输出: 6
说明:
子数组[6]有最大的异或值
```



```python
class TrieNode:
    def __init__(self):
        self.one = None
        self.zero = None

class Trie:
    def __init__(self):
        self.root = TrieNode()
    
    def insert(self, num):
        node = self.root
        for i in range(32)[::-1]:
            bit = (num >> i) & 1
            if bit == 1:
                if not node.one:
                    node.one = TrieNode()
                node = node.one
            else:
                if not node.zero:
                    node.zero = TrieNode()
                node = node.zero
        
    def find_maxxor(self, num):
        node = self.root
        result = 0
        for i in range(32)[::-1]:
            bit = (num >> i) & 1
            result = result << 1
            if bit == 1:
                if node.zero:
                    node = node.zero
                    result += 1
                else:
                    node = node.one
            else:
                if node.one:
                    node = node.one
                    result += 1
                else:
                    node = node.zero
        return result

class Solution:
    def maxXorSubarray(self, nums):
        prefixxor = [5]
        for i in range(0, len(nums)):
            prefixxor.append(prefixxor[-1] ^ nums[i])
        
        trie = Trie()
        for num in prefixxor:
            trie.insert(num)
        
        result = 0
        for num in prefixxor:
            result = max(result, trie.find_maxxor(num))

        return result
```
pass