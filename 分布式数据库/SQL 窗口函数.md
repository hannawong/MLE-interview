# SQL 窗口函数

**一.窗口函数有什么用？**

在日常工作中，经常会遇到需要**在每组内排名**，比如下面的业务需求：

> 排名问题：每个部门按业绩来排名
> topN问题：找出每个部门排名前N的员工进行奖励

面对这类需求，就需要使用sql的高级功能窗口函数了。

**二.什么是窗口函数？**

窗口函数的基本语法如下：

```text
<窗口函数> over (partition by <用于分组的列名>
                order by <用于排序的列名>)
```

那么语法中的<窗口函数>都有哪些呢？

<窗口函数>的位置，可以放以下两种函数：

1） 专用窗口函数，包括后面要讲到的rank, dense_rank, row_number等专用窗口函数。

2） 聚合函数，如sum. avg, count, max, min等

因为窗口函数是对where或者group by子句处理后的结果进行操作，所以**窗口函数原则上只能写在select子句中**。

**三.如何使用？**

接下来，就结合实例，给大家介绍几种窗口函数的用法。

**1.专用窗口函数rank**

例如下图，是班级表中的内容

![img](https://pic2.zhimg.com/80/v2-f8c3b3deb99122d75bb506fdbea81c8d_1440w.jpg)

如果我们想在每个班级内按成绩排名，得到下面的结果。

![img](https://pic3.zhimg.com/80/v2-3285d1d648de9f90864000d58847087a_1440w.jpg)

以班级“1”为例，这个班级的成绩“95”排在第1位，这个班级的“83”排在第4位。上面这个结果确实按我们的要求在每个班级内，按成绩排名了。

得到上面结果的sql语句代码如下：

```text
select *,
   rank() over (partition by 班级
                 order by 成绩 desc) as ranking
from 班级表
```

我们来解释下这个sql语句里的select子句。rank是排序的函数。要求是“每个班级内按成绩排名”，这句话可以分为两部分：

1）每个班级内：**按班级分组**

**partition by**用来对表分组。在这个例子中，所以我们指定了按“班级”分组（partition by 班级）

2）按成绩排名

**order by**子句的功能是对分组后的结果进行排序，默认是按照升序（asc）排列。在本例中（order by 成绩 desc）是按成绩这一列排序，加了desc关键词表示降序排列。

通过下图，我们就可以理解partiition by（分组）和order by（在组内排序）的作用了。



![img](https://pic2.zhimg.com/80/v2-451c70aa24c68aa7142693fd27c85605_1440w.jpg)



**简单来说，窗口函数有以下功能：**

1）同时具有分组和排序的功能

2）不减少原表的行数

3）语法如下：

```text
<窗口函数> over (partition by <用于分组的列名>
                order by <用于排序的列名>)
```



**2.其他专业窗口函数**

专用窗口函数rank, dense_rank, row_number有什么区别呢？

它们的区别我举个例子，你们一下就能看懂：

```text
select *,
   rank() over (order by 成绩 desc) as ranking,
   dense_rank() over (order by 成绩 desc) as dese_rank,
   row_number() over (order by 成绩 desc) as row_num
from 班级表
```

得到结果：

![img](https://pic2.zhimg.com/80/v2-ad1d86f5a5b9f0ef684907b20b341099_1440w.jpg)

最后，需要强调的一点是：在上述的这三个专用窗口函数中，函数后面的括号不需要任何参数，保持()空着就可以。

现在，大家对窗口函数有一个基本了解了吗？

**3.聚合函数作为窗口函数**

聚和窗口函数和上面提到的专用窗口函数用法完全相同，只需要把聚合函数写在窗口函数的位置即可，但是函数后面括号里面不能为空，需要指定聚合的列名。

我们来看一下窗口函数是聚合函数时，会出来什么结果：

```text
select *,
   sum(成绩) over (order by 学号) as current_sum,
   avg(成绩) over (order by 学号) as current_avg,
   count(成绩) over (order by 学号) as current_count,
   max(成绩) over (order by 学号) as current_max,
   min(成绩) over (order by 学号) as current_min
from 班级表
```

得到结果：

![img](https://pic2.zhimg.com/80/v2-c48f0218306f65049fcf9f98c184226d_1440w.jpg)

不仅是sum求和，平均、计数、最大最小值，也是同理，都是**针对自身记录、以及自身记录之上的**所有数据进行计算，现在再结合刚才得到的结果（下图），是不是理解起来容易多了？

如果想要知道所有人成绩的总和、平均等聚合结果，看最后一行即可。

**这样使用窗口函数有什么用呢？**

聚合函数作为窗口函数，可以在每一行的数据里直观的看到，截止到本行数据，统计数据是多少（最大值、最小值等）。同时可以看出每一行数据，对整体统计数据的影响。

**四.注意事项**

partition子句可是省略，省略就是**不指定分组**，结果如下，只是按成绩由高到低进行了排序：

```text
select *,
   rank() over (order by 成绩 desc) as ranking
from 班级表
```

得到结果：

![img](https://pic1.zhimg.com/80/v2-c589fe21dd785ff5996174684cc4de84_1440w.jpg)

但是，这就失去了窗口函数的功能，所以一般不要这么使用。