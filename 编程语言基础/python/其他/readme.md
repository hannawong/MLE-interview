## 1. 类

Python是面向对象语言，所有允许定义类并且可以继承和组合。Python没有访问标识如在C++中的**public, private**, 这就非常信任程序员的素质，相信每个程序员都是“成人”了~

类的定义：

```python
class Employee:
   empCount = 0
 
   def __init__(self, name, salary):
      self.name = name
      self.salary = salary
      Employee.empCount += 1
   
   def displayCount(self):
     print "Total Employee %d" % Employee.empCount
 
   def displayEmployee(self):
      print "Name : ", self.name,  ", Salary: ", self.salary

```

## 2. python运行速度

 Python跑起来会比编译型语言慢。幸运的是，Python允许使用C扩展写程序，所以瓶颈可以得到处理。Numpy库就是一个很好例子，因为很多代码不是Python直接写的，所以运行很快。



## 3. 深拷贝浅拷贝

- copy.copy() 是藕断丝连，两者指向同一个地址，**改变copy的内容也会影响原来的内容**。拷贝父对象，不会拷贝对象的内部的子对象。

- copy.deepcopy()是完全切断关系，新创建对象。完全拷贝了父对象及其子对象。

对于那些包含了其他object(例如list，class)的复合对象(compound objects), 深浅拷贝是有区别的。

![图片描述](https://segmentfault.com/img/bVbrl56?w=310&h=227)





![img](https://pic2.zhimg.com/80/02090688c635f7168994ea351f66a2f4_1440w.jpg?source=1940ef5c)



#### 创建二维数组 以及 python中[0]* n与[0 for _ in range(n)]的区别与联系



[ 0 ] * n 是浅拷贝， 也就是把一个列表重复了 n 次；[[0]\*n]\*m 这种方式是直接将 [0]\*n 复制了m遍
[0 for _ in range(n)] 才是创建，深拷贝

```python
n = 4
dp1 = [0] * n
dp2 = [0 for _ in range(n) ]
print('dp1:',dp1)
print('dp2:',dp2)
```


对于一维数组来说，这两者的效果是一样的

```python
dp1: [0, 0, 0, 0]
dp2: [0, 0, 0, 0]
```

二维数组，创建一个3*4的矩阵，元素全为0，修改（0，2）个元素的值为3，则提供三种方法如下：

```python
m,n = 3,4
dp1 = [[0] * n ] * m
dp2 = [[0 for _ in range(n) ] for _ in range(m)]
dp3 = [[0] * n for _ in range(m)]
dp1[0][2] = 3
dp2[0][2] = 3
dp3[0][2] = 3
print('dp1:',dp1)
print('dp2:',dp2)
print('dp2:',dp3)
```


结果为：

```python
dp1: [[0, 0, 3, 0], [0, 0, 3, 0], [0, 0, 3, 0]]
dp2: [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
dp2: [[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
```

第一种方法不行，每一行的改变都会改变其他行
第二种、第三种方法均可。

又如：

```python
dp = [[]*10]
dp[1].append("t")
```

你会惊讶的发现这个操作把dp中所有的list都append了一个"t"! 这显然不是我们想要的。这是因为*n操作是浅拷贝，对于列表这种复杂数据结构，并没有开辟另外的内存空间，而只是给了一个别名。所以，**我们应该使用for _ in range(n)**。