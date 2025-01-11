
**样例 1：**
输入
```
chars = "abAcD"
```
输出：
```
"acbAD"
```
解释：
你也可以返回"abcAD"或者"cbaAD"或者其他正确的答案。 

**样例 2：**
输入：
```
chars = "ABC"
```
输出：
```
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