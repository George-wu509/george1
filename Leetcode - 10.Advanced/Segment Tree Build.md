
**样例 1:**
```
输入：[1,4]
输出："[1,4][1,2][3,4][1,1][2,2][3,3][4,4]"
解释：
	               [1,  4]
	             /        \
	      [1,  2]           [3, 4]
	      /     \           /     \
	   [1, 1]  [2, 2]     [3, 3]  [4, 4]
```
**样例 2:**
```
输入：[1,6]
输出："[1,6][1,3][4,6][1,2][3,3][4,5][6,6][1,1][2,2][4,4][5,5]"
解释：
	       [1,  6]
             /        \
      [1,  3]           [4,  6]
      /     \           /     \
   [1, 2]  [3,3]     [4, 5]   [6,6]
   /    \           /     \
[1,1]   [2,2]     [4,4]   [5,5]
```




```python
def build(self, start, end):
	#首先start > end的错误输入直接终止
	if start > end: return
	#如果 start 等于 end, 那么该节点是叶子节点，不再有左右儿子
	if start == end:
		seg = SegmentTreeNode(start, end)
		return seg
	else:
		seg = SegmentTreeNode(start, end)
		#计算左右叶节点
		pt = int((start + end) / 2)
		left = self.build(start, pt)
		seg.left = left
		right = self.build(pt + 1, end)
		seg.right = right
		return seg
```
pass