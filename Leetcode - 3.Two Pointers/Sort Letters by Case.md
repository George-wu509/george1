Lintcode 49
给定一个只包含字母的字符串 `chars`，按照先小写字母后大写字母的顺序进行排序。  
对于不同的语言，`chars`将以不用的形式给出，例如对于字符串 `"abc"` ，将以下面的形式给出

- Java: char[] chars = {'a', 'b', 'c'};
- Python：chars = ['a', 'b', 'c']
- C++：string chars = "abc";

你需要实现原地算法解决这个问题，不需要返回任何值，我们会根据排序后的chars判断你的结果是否正确。

**样例 1：**
输入
```python
chars = "abAcD"
```
输出：
```python
"acbAD"
```
解释：
你也可以返回"abcAD"或者"cbaAD"或者其他正确的答案。 

**样例 2：**
输入：
``` python
chars = "ABC"
```
输出：
```python
"ABC"
```
解释：
你也可以返回"CBA"或者"BCA"或者其他正确的答案。


```python
    def sort_Letters(self, chars):
        # 定义左右指针并初始化
        left = 0
        right = len(chars) - 1

        # 两指针相向移动，交会则结束
        while left <= right:
            # 左指针向右移动，直到找到第一个大写字母
            while left <= right and chars[left] >= 'a' and chars[left] <= 'z':
                left += 1

            # 右指针向左移动，直到找到第一个小写字母
            while left <= right and chars[right] >= 'A' and chars[right] <= 'Z':
                right -= 1

            # 将左边的大写字母和右边的小写字母交换位置
            if left <= right:
                tmp = chars[left]
                chars[left] = chars[right]
                chars[right] = tmp
                left += 1
                right -= 1
```
pass