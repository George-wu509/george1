Lintcode 1394
给定句子S，其由用空格分隔的单词组成。 每个单词仅包含小写和大写字母。

我们想将句子转换为“Goat Latin”（一种类似于Pig Latin的伪造语言）。

Goat Latin的规则如下：

如果一个单词以元音（a，e，i，o或u）开头，则在单词的末尾附加“ma”。  
例如，“apple”这个词就变成了“applema”。

如果一个单词以辅音（即不是元音）开头，则删除第一个字母并追加它到最后，然后添加“ma”。  
例如，“山羊”这个词就变成了“oatgma”。

在句子的每个单词的末尾添加一个字母'a'，从1开始。  
例如，第一个单词将“a”添加到结尾，第二个单词将“aa”添加到结尾，依此类推。

返回将从S到Goat Latin的转换后的最终语句

**样例1**
```python
"""
输入: "I speak Goat Latin"
输出: "Imaa peaksmaaa oatGmaaaa atinLmaaaaa"
```
**样例2**
```python
"""
输入: "The quick brown fox jumped over the lazy dog"
输出: "heTmaa uickqmaaa rownbmaaaa oxfmaaaaa umpedjmaaaaaa overmaaaaaaa hetmaaaaaaaa azylmaaaaaaaaa ogdmaaaaaaaaaa"
```


```python
    def to_goat_latin(self, s: str) -> str:
        vowels = {"a", "e", "i", "o", "u", "A", "E", "I", "O", "U"}
        n = len(s)
        i, cnt = 0, 1
        words = list()

        while i < n:
            j = i
            while j < n and s[j] != " ":
                j += 1      
            cnt += 1
            if s[i] in vowels:
                words.append(s[i:j] + "m" + "a" * cnt)
            else:
                words.append(s[i+1:j] + s[i] + "m" + "a" * cnt)
            
            i = j + 1
        
        return " ".join(words)
```
pass