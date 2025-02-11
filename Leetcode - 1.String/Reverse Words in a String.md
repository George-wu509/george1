Lintcode 53
给定一个字符串，逐个翻转字符串中的每个单词

**样例 1：**
输入：
```python
"""
s = "the sky is blue"
```
输出：
```python
"""
"blue is sky the"
```
解释：
返回逐字反转的字符串。  

**样例 2：**
输入：
```python
"""
s = "hello world"
```
输出：
```python
"""
"world hello"
```
解释：
返回逐字反转的字符串。


```python
    def reverse_words(self, s):
        #strip()去掉s头尾的空格，split()按照空格分割字符串，reversed翻转，''.join按照空格连接字符串
        return ' '.join(reversed(s.strip().split()))
```
pass