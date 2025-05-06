Lintcode 422
给定一个字符串 `s`， 包含大小写字母、空格 `' '`，请返回其最后一个单词的长度。
**单词** 是指仅由字母组成，且不包含空格字符的 **最大子字符串**

**样例 1：**
```python
"""
输入："Hello World"
输出：5
```
**样例 2：**
```python
"""
输入："Hello LintCode"
输出：8
```


```python
    def length_of_last_word(self, s):
        return len(s.strip().split(' ')[-1])
```
pass
解釋: 
strip() 將string的頭尾空格消掉, 譬如  "  Hello  " -> "Hello"
split(' ')將string依照輸入的' '譬如空格, 將string切割成數個子string的列表
 譬如:  "hello world" -> ["hello" , "world"]
