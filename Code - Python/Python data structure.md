
|                      |                                                          |
| -------------------- | -------------------------------------------------------- |
| List 列表<br>[1,'A',4] | ['physics', 'chemistry', 1997, 2000]<br>[1, 2, 3, 4, 5 ] |
| Tuple 元祖             | ('physics', 'chemistry', 1997, 2000)<br>(1, 2, 3, 4, 5 ) |
| Set 集合               | {1,2,3}                                                  |
| Dict 字典              | {key1: value1, key2: value2}                             |

列表  [1, 2, 3],  元祖 (1, 2, 3),  集合 {1, 2, 3},  字典  {key1: value1, key2: value2}

集合  Set()  [无序的不重复元素序列](https://www.runoob.com/python3/python3-set.html)

Create:   S = {'ab', 'cd', 'ef'}

                S = set()    創建一個空集合必須用set()而不是{}. 因為{}是創建空字典

                S = set("google")

--> S = {'e', 'o', 'g', 'l'}

     S = set([1, 2, 3, 4, 6])

           --> S = {1, 2, 3, 4, 6}

     S = set(['ab', 'cd', 'ef'])

           --> S = {'ab', 'cd', 'ef'}

Operation:

     S.add()       S.update()       S.remove()       S.discard()

     S.pop()       Len(S)              S.clear()

字典  Dict         [Python 字典(Dictionary)](https://www.runoob.com/python/python-dictionary.html)

         Create:  d = {'a':1, 'b':2, 'c':3}

                        d = {'Alice': '2341', 'Beth': '9102', 'Cecil': '3258'}

                        d = { 'abc': 123, 98.6: 37 }

         Operation:    d = {'Name': 'Runoob', 'Age': 7, 'Class': 'First'}

                        d['Name'] = 'Runoob'

                        d['Age'] = 8    -->   d['Age']=8

  d.get('Age') = 8

                        del d['Name']  --> d = {'Age': 7, 'Class': 'First'}

                        d.clear()

                        del d