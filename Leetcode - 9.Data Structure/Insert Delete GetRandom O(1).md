
```
// 初始化空集set
RandomizedSet randomSet = new RandomizedSet();

// 1插入set中。返回正确因为1被成功插入
randomSet.insert(1);

// 返回错误因为2不在set中
randomSet.remove(2);

// 2插入set中，返回正确，set现在有[1,2]。
randomSet.insert(2);

// getRandom 应该随机的返回1或2。
randomSet.getRandom();

// 从set中移除1，返回正确。set现在有[2]。
randomSet.remove(1);

// 2已经在set中，返回错误。
randomSet.insert(2);

// 因为2是set中唯一的数字，所以getRandom总是返回2。
randomSet.getRandom();
```


```python
import random

class RandomizedSet(object):

    def __init__(self):
        # do initialize if necessary    
        self.nums, self.val2index = [], {}
        
    # @param {int} val Inserts a value to the set
    # Returns {bool} true if the set did not already contain the specified element or false
    def insert(self, val):
        if val in self.val2index:
            return False
        
        self.nums.append(val)
        self.val2index[val] = len(self.nums) - 1
        return True
        
    # @param {int} val Removes a value from the set
    # Return {bool} true if the set contained the specified element or false
    def remove(self, val):
        # Write your code here
        if val not in self.val2index:
            return False
        
        index = self.val2index[val]
        last = self.nums[-1]
        
        # move the last element to index
        self.nums[index] = last
        self.val2index[last] = index
        
        # remove last element
        self.nums.pop()
        del self.val2index[val]
        return True
    
    # return {int} a random number from the set
    def getRandom(self):
        return self.nums[random.randint(0, len(self.nums) - 1)]
```
pass