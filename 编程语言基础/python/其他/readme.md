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

- copy.copy() 是藕断丝连，两者指向同一个地址，**改变copy的内容也会影响原来的内容**

- copy.deepcopy()是完全切断关系，新创建对象。

对于那些包含了其他object(例如list，class)的复合对象(compound objects), 深浅拷贝是有区别的。

![图片描述](https://segmentfault.com/img/bVbrl56?w=310&h=227)





![img](https://pic2.zhimg.com/80/02090688c635f7168994ea351f66a2f4_1440w.jpg?source=1940ef5c)



