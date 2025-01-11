
```
输入 : s = "abccccdd"
输出 : 7
说明 : 
一种可以构建出来的最长回文串方案是 "dccaccd"。
```


```python
    def longest_palindrome(self, s):
        # Write your code here
        hash = {}

        for c in s:
            if c in hash:
                del hash[c]
            else:
                hash[c] = True

        remove = len(hash)
        if remove > 0:
            remove -= 1
    
        return len(s) - remove
```
pass