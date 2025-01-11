
**样例 1：**
输入：
```
A = [1,1,2,2,3,4,4]
```
输出：
```
3
```
解释：
仅3出现一次  

**样例 2：**
输入：
```
A = [0,0,1]
```
输出：
```
1
```
解释：
仅1出现一次


```python
    def single_number(self, a):
        ans = 0;
        for x in a:
            ans = ans ^ x
        return ans
```
pass