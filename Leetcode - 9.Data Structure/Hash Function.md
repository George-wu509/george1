**样例 1:**
```
输入:  key = "abcd", size = 1000
输出: 978	
样例解释：(97 * 33^3 + 98*33^2 + 99*33 + 100*1)%1000 = 978
```
**样例 2:**
```
输入:  key = "abcd", size = 100
输出: 78	
样例解释：(97 * 33^3 + 98*33^2 + 99*33 + 100*1)%100 = 78
```



```python
    def hash_code(self, key, size):
        ans = 0
        for x in key:
            ans = (ans * 33 + ord(x)) % size
        return ans
```
pass