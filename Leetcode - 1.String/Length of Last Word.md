
**样例 1：**
```
输入："Hello World"
输出：5
```
**样例 2：**
```
输入："Hello LintCode"
输出：8
```


```python
    def length_of_last_word(self, s):
        return len(s.strip().split(' ')[-1])
```
pass