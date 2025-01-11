
**样例 1：**
输入：
```
数组 = []
```
输出：
```
0
```
解释：
空数组，长度为0.

**样例 2：**
输入：
```
数组 = [1,1,1,2,2,3]
```
输出：
```
5
```
解释：
长度为 5， 数组为：[1,1,2,2,3]


```python
    def removeDuplicates(self, nums):
        B = []
        before = None
        countb = 0
        for number in nums:
            if(before != number):
                B.append(number)
                before = number
                countb = 1
            elif countb < 2:
                B.append(number)
                countb += 1
        p = 0
        for number in B:
            nums[p] = number
            p += 1
        return p
```
pass