Lintcode 1193
给定一个单词，你需要判断其中大写字母的使用是否正确。

当下列情况之一成立时，我们将单词中大写字母的用法定义为正确：

这个单词中的所有字母都是大写字母，如“USA”。  
这个单词中的所有字母都不是大写字母，如“lintcode”。  
如果它有多个字母，例如“Google”，那么这个单词中的第一个字母就是大写字母。  
否则，我们定义该单词没有以正确的方式使用大写字母

**样例 1：**
```python
"""
输入: "USA"
输出: True
```
**样例 2：**
```python
"""
输入: "FlaG"
输出: False
```


```python
    def detect_capital_use(self, word: str) -> bool:
        # 若第 1 个字母为小写，则需额外判断第 2 个字母是否为小写
        if len(word) >= 2 and word[0].islower() and word[1].isupper():
            return False
        
        # 无论第 1 个字母是否大写，其他字母必须与第 2 个字母的大小写相同
        return all(word[i].islower() == word[1].islower() for i in range(2, len(word)))
```
pass