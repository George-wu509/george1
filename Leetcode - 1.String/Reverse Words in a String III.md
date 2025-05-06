Lintcode 1173
给定一个字符串句子，反转句子中每一个单词的所有字母，同时保持空格和最初的单词顺序。

```python
"""
输入: "Let's take LeetCode contest"
输出: "s'teL ekat edoCteeL tsetnoc"
```


```python
    def reverse_words(self, s):
        word = ""
        answer = ""
        for c in s:
            if c == ' ':
                if word != "":
                    answer += word[::-1]
                word = ""
                answer += c
            else:
                word += c
        if word != "":
            answer += word[::-1]
        return answer
```
pass

```python
    def reverse_words(self, s):
        s_list = s.strip().split()
		for i in range(len(s_list)):
		    s_list[i] = s_list[i][::-1]
		
		answer = " ".join(s_list)
        return answer
```
pass