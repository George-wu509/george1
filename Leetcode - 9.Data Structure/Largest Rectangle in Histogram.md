
**样例 1：**
输入：
```
height = [2,1,5,6,2,3]
```
输出：
```
10
```
解释：
第三个和第四个直方图截取矩形面积为2*5=10。

**样例 2：**
输入：
```
height = [1,1]
```
输出：
```
2
```
解释：
第一个和第二个直方图截取矩形面积为2*1=2。


```python
    def largest_rectangle_area(self, heights):
        indices_stack = []
        area = 0
        for index, height in enumerate(heights + [0]):
            while indices_stack and heights[indices_stack[-1]] >= height:		#如果列表尾部高度大于当前高度
                popped_index = indices_stack.pop()
                left_index = indices_stack[-1] if indices_stack else -1		
                width = index - left_index - 1		#如果列表为空，则宽度为index，否则为index-indices_stack[-1]-1
                area = max(area, width * heights[popped_index])
                
            indices_stack.append(index)		#压入列表中
            
        return area
```
pass