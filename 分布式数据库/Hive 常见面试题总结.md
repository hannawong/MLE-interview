# Hive 常见面试题总结 



**1、Hive表关联查询，如何解决数据倾斜的问题？（☆☆☆☆☆）**

在执行任务的时候，任务进度长时间维持在99%左右，查看任务监控页面，发现只有少量（1个或几个）**reduce**子任务未完成。因为其处理的数据量和其他reduce差异过大。

1）倾斜原因： map输出数据按key Hash的分配到reduce中，由于**key分布不均匀**造成不同reducer上的数据量差异过大。

![img](https://pic1.zhimg.com/80/v2-6fff06e6b4341f385c901c74b7be0eb0_1440w.jpg)

2）解决方案

（1）调优：**参数调节**：

hive.map.aggr = true

hive.groupby.skewindata=true ##--有数据倾斜的时候进行负载均衡

有数据倾斜的时候进行负载均衡，当选项设定位true,生成的查询计划会有两个MR Job。第一个MR Job中，Map的输出结果集合会随机分布到Reduce中，每个Reduce做部分聚合操作，并输出结果。这样，相同的Group By Key有可能被分发到不同的Reduce中，从而达到负载均衡的目的；第二个MR Job再根据预处理的数据结果按照Group By Key 分布到 Reduce 中（这个过程可以保证相同的 Group By Key 被分布到同一个Reduce中），最后完成最终的聚合操作。

（2）SQL 语句调节：

① 选用join key分布**最均匀**的表作为**驱动表**。做好列裁剪和filter操作，以达到两表做join 的时候，数据量相对变小的效果。

② 大小表Join：

使用map-side join,让小的维度表（1000 条以下的记录条数）**先进内存**。在map端完成reduce，这样就不用再reduce了

③ 大表Join大表：

把**空值**的key变成一个字符串加上随机数，把倾斜的数据分到不同的reduce上，由于null 值**关联不上**，处理后并不影响最终结果。

④ count distinct大量相同特殊值:

count distinct 时，将值为空的情况单独处理，如果是计算**count distinct**，可以不用处理，直接过滤，在最后结果中**加1**。如果还有其他计算，需要进行group by，**可以先将值为空的记录单独处理**，再和其他计算结果进行**union**。



**2、Hive的HiveQL转换为MapReduce的过程？（☆☆☆☆☆）**

HiveSQL ->AST(抽象语法树) -> QB(查询块) ->OperatorTree（操作树）->**优化**后的操作树->mapreduce**任务**树->优化后的mapreduce任务树

![img](https://pic2.zhimg.com/80/v2-8e366f5b648fce344c02b107b2959de1_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-4c1097ff510415c9c25c44e2c1eba058_1440w.jpg)

过程描述如下：

SQL **Parser**：**Antlr**定义SQL的语法规则，完成SQL的词法分析、语法分析，将SQL转化为**抽象语法树**AST Tree；

Semantic Analyzer：遍历AST Tree，抽象出查询的基本组成单元QueryBlock；

Logical plan（逻辑计划）：遍历QueryBlock，翻译为执行操作树OperatorTree；

Logical plan optimizer: 逻辑层优化器进行OperatorTree变换，合并不必要的ReduceSinkOperator，减少shuffle数据量；

**Physical plan**：遍历OperatorTree，翻译为MapReduce任务；

Logical plan optimizer：物理层优化器进行MapReduce任务的变换，生成最终的执行计划。



**3、Hive底层与数据库交互原理？（☆☆☆☆☆）**

由于Hive的元数据可能要面临不断地更新、修改和读取操作，所以它显然不适合使用Hadoop文件系统进行存储。目前Hive将元数据存储在RDBMS中，比如存储在MySQL、Derby中。元数据信息包括：**存在的表、表的列、权限**和更多的其他信息。

![img](https://pic4.zhimg.com/80/v2-cb60019ae2881e7c4081eaf3528bcb87_1440w.jpg)



**4、Hive的两张表关联，使用MapReduce怎么实现？（☆☆☆☆☆）**

如果其中有一张表为小表，直接使用map-side join的方式（map端内存加载小表）进行聚合。

如果两张都是大表，那么采用**联合key**（也就是两个key），联合key的第一个组成部分是 join on中的**公共字段**("join on A.id = B.id",那么公共字段是id)，第二部分是一个flag，0代表表A，1代表表B，由此让Reduce区分两个表；

在Mapper中同时处理两张表的信息，将join on公共字段相同的数据划分到同一个reducer中，然后在Reduce中实现聚合。



**5、请谈一下Hive的特点，Hive和RDBMS有什么异同？**

hive是基于Hadoop的一个数据仓库工具，可以将结构化的数据文件映射为一张数据库表，并提供完整的sql查询功能，可以将sql语句转换为MapReduce任务进行运行。其优点是学习成本低，可以通过类SQL语句快速实现简单的MapReduce统计，不必开发专门的MapReduce应用，十分适合数据仓库的统计分析，但是**Hive不支持实时查询**。

Hive与关系型数据库的区别：

![img](https://pic3.zhimg.com/80/v2-64953645f5acf37701fe13f22e2457d2_1440w.jpg)

**6、请说明hive中 Sort By，Order By，Cluster By，Distrbute By各代表什么意思？**

order by：会对输入做**全局**排序，因此**只有一个reducer**（一个reducer需要处理所有数据，多个reducer无法保证全局有序）。只有一个reducer，会导致当输入规模较大时，需要较长的计算时间。

sort by：不是全局排序，只能保证每个reducer本身是有序的。

distribute by：按照指定的字段对数据进行划分**输出到不同的reduce中**。

cluster by：= sort by ... distributed by。



**7、写出hive中split、coalesce及collect_list函数的用法（可举例）？**

split将字符串转化为数组，即：split('a,b,c,d' , ',') ==> ["a","b","c","d"]。

coalesce(T v1, T v2, …) 返回参数中的**第一个非空值**；如果所有值都为 NULL，那么返回NULL。

collect_list**列出该字段所有的值，不去重** => select collect_list(id) from table。



**8、Hive有哪些方式保存元数据，各有哪些特点？**

Hive支持三种不同的元存储服务器，分别为：内嵌式元存储服务器、本地元存储服务器、远程元存储服务器，每种存储方式使用不同的配置参数。

内嵌式元存储主要用于单元测试，在该模式下每次只有一个进程可以连接到元存储，Derby是内嵌式元存储的默认数据库。

在本地模式下，每个Hive客户端都会打开到数据存储的连接并在该连接上请求SQL查询。

在远程模式下，所有的Hive客户端都将打开一个到元数据服务器的连接，该服务器依次查询元数据，元数据服务器和客户端之间使用Thrift协议通信。



**9、Hive内部表和外部表的区别？**

创建表时：创建内部表时，会将**数据移动到数据仓库指向的路径**；若创建外部表，仅**记录**数据所在的路径，不对数据的位置做任何改变。

删除表时：在删除表的时候，内部表的**元数据和数据**会被一起删除， 而外部表**只删除元数据**，不删除数据。这样外部表相对来说更加安全些，数据组织也更加灵活，方便共享源数据。



**10、Hive 中的压缩格式TextFile、SequenceFile、RCfile 、ORCfile各有什么区别？**

1、TextFile

默认格式，存储方式为行存储，数据**不做压缩**，磁盘开销大。可结合Gzip、Bzip2进行压缩和解压缩，但使用这种方式，压缩后的文件不是**可分割的**，从而无法对数据进行**并行**操作。并且在反序列化过程中，必须逐个字符判断是不是分隔符和行结束符，因此反序列化开销会比SequenceFile高几十倍。

2、SequenceFile

SequenceFile是Hadoop API提供的一种二进制文件支持，存储方式为行存储，其具有使用方便、**可分割**、可压缩的特点。

SequenceFile支持三种压缩选择：NONE**，RECORD，BLOCK。**Record压缩率低，一般建议使用BLOCK压缩。

3、RCFile

存储方式：数据按行**分块**，**每块按列存储**。结合了行存储和列存储的优点：

首先，RCFile 保证同一行的数据位于同一节点，因此元组重构的开销很低；

其次，像列存储一样，RCFile 能够利用**列维度的数据压缩**，并且**能跳过不必要的列读取**；

4、ORCFile

存储方式：数据按行分块 每块按照列存储。

压缩快、快速列存取。

效率比rcfile高，**是rcfile的改良版本**。

【总结】相比TEXTFILE和SEQUENCEFILE，RCFILE由于**列式**存储方式，数据加载时性能消耗较大，但是具有较好的压缩比和**查询**响应（因此适用于OLAP）。

数据仓库的特点是**一次写入、多次读取**，因此，整体来看，RCFILE相比其余两种格式具有较明显的优势。



**11、所有的Hive任务都会有MapReduce的执行吗？**

不是，从Hive0.10.0版本开始，对于简单的不需要聚合的类似SELECT \<col\> from \<table\> LIMIT n语句，不需要起MapReduce job，直接通过Fetch task获取数据。因为这个查询只需要遍历文件，然后输出即可了。



**12、Hive的函数：UDF、UDAF、UDTF的区别？**

UDF：单行进入，单行输出

UDAF (表聚合函数，如sum、avg)：多行进入，单行输出

UDTF（表生成函数，如explode）：单行输入，多行输出



**13、说说对Hive桶表的理解？**

桶表是对数据进行**哈希取值**，然后放到**不同文件**中存储。

数据加载到桶表时，会对字段取hash值，**然后与桶的数量取模**。把数据放到对应的文件中。**物理上**，每个桶就是表(或分区）目录里的一个**文件**，一个作业产生的桶(输出文件)和reduce任务个数相同。

桶表专门用于**抽样查询**，是很专业性的，不是日常用来存储数据的表，需要抽样查询时，才创建和使用桶表。