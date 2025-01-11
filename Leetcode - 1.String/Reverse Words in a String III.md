
```
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