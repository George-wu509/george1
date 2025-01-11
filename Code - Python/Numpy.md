
Reference:

(1) NumPy基礎 — 初步理解NumPy Array(陣列) 

[https://medium.com/seaniap/numpy%E5%9F%BA%E7%A4%8E-%E5%88%9D%E6%AD%A5%E7%90%86%E8%A7%A3numpy-array-%E9%99%A3%E5%88%97-34263c3f58f0](https://medium.com/seaniap/numpy%E5%9F%BA%E7%A4%8E-%E5%88%9D%E6%AD%A5%E7%90%86%E8%A7%A3numpy-array-%E9%99%A3%E5%88%97-34263c3f58f0)

(2)

List1 = [1,2,3]

A = np. array ( List1 )

B1 = np. zeros (3)

B2 = np. zeros ((2,4))

C1 = np. ones (4) 

C2 = np. ones ((3,4)) 

D1 = np. arange(5) = array([0,1,2,3,4])

D2 = np. arange (0,11,2) = array([0,2,4,6,8,10]

D3 = np. linspace (0,12,3) = array([0, 6, 12])

E = np. Eye(5)

F1 = np. random. rand (3) = array([0.924, 0.620, 0.257])   --> [0,1]

F2 = np. random. rand (3,4)

F3 = np. random . randn(5,5)   --> [normal distribution]

F4 = np. random. randint(1,100,10) = array([70, 2, 17, 80, 38, 39,  2, 14, 83, 19])

F5 = np. random. randint(1,100,(2,3))  = array([[40, 29, 30],[67,  4, 63]])

[3]

M1 = np.arange(25)

M1 .shape = (25,)

M2 = M1.reshape(5,5)

M2.shape = (5,5)

M3 = array([38, 18, 22, 10, 10, 23, 35, 39, 23,  2])

M3.max() = 39

M3.min() = 2

M3.argmax = 7

M3.argmin = 9

M3.dtype = 'int64'