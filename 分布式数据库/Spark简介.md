# Spark简介

来源：https://zhuanlan.zhihu.com/p/258483133

2014，是个久远的年代，那个时候，大数据江湖群雄并起，门派林立。论内功，有少林派的 Hadoop，Hadoop 可谓德高望重、资历颇深，2006 年由当时的互联网老大哥 Yahoo！开源并迅速成为 Apache 顶级项目。所谓天下武功出少林，Hadoop 的三招绝学：**HDFS（分布式文件系统）、YARN（分布式调度系统）、MapReduce（分布式计算引擎）**，为各门各派武功绝学的发展奠定了坚实基础。论阵法，有武当派的 Hive，Hive 可谓是开源分布式数据仓库的鼻祖。论剑法，有峨眉派的 Mahout，峨眉武功向来“一树开五花、五花八叶扶”，Mahout 在分布式系统之上提供主流的经典机器学习算法实现。论轻功，有昆仑派的 Storm，在当时，Storm 轻巧的分布式流处理框架几乎占据着互联网流计算场景的半壁江山。

Spark 师从 Hadoop，习得 MapReduce 内功心法，因天资聪慧、勤奋好学，年纪轻轻即独创内功绝学：Spark Core —— **基于内存的分布式计算引擎**。青，出于蓝而胜于蓝；冰，水为之而寒于水。凭借扎实的内功，Spark 练就一身能为：

- Spark SQL —— 分布式数据分析
- Spark Streaming —— 分布式流处理
- Spark MLlib —— 分布式机器学习
- Spark GraphX —— 分布式图计算

自恃内功深厚、招式变幻莫测，Spark 初涉江湖便立下豪言壮语：One stack to rule them all —— 剑锋直指各大门派。小马乍行嫌路窄，大鹏展翅恨天低。各位看官不禁要问：Spark 何以傲视群雄？Spark 修行的内功心法 Spark Core，与老师 Hadoop 的 MapReduce 绝学相比，究竟有何独到之处？

**Hadoop MapReduce**

欲探究竟，还需从头说起。在 Hadoop 出现以前，数据分析市场的参与者主要由以 IOE（IBM、Oracle、EMC）为代表的传统 IT 巨头构成，Share-nothing 架构的分布式计算框架大行其道。传统的 Share-nothing 架构凭借其预部署、高可用、高性能的特点在金融业、电信业大放异彩。然而，随着互联网行业飞速发展，瞬息万变的业务场景对于分布式计算框架的灵活性与扩展性要求越来越高，笨重的 Share-nothing 架构无法跟上行业发展的步伐。2006 年，Hadoop 应运而生，MapReduce 提供的分布式计算抽象，结合分布式文件系统 HDFS 与分布式调度系统 YARN，完美地诠释了“数据不动代码动”的新一代分布式计算思想。

![img](https://pic3.zhimg.com/80/v2-9a24b99564936f1ddcc44f42de32dd4a_1440w.jpg)

顾名思义，MapReduce 提供两类计算抽象，即 Map 和 Reduce。Map 抽象用于封装数据映射逻辑，开发者通过实现其提供的 map 接口来定义**数据转换**流程；Reduce 抽象用于封装**数据聚合**逻辑，开发者通过实现 reduce 接口来定义数据汇聚过程。Map 计算结束后，往往需要对数据进行**分发**才能启动 Reduce 计算逻辑来执行数据聚合任务，数据分发的过程称之为 **Shuffle**。MapReduce 提供的分布式任务调度让开发者专注于业务逻辑实现，**而无需关心依赖管理、代码分发等分布式实现问题**。在 MapReduce 框架下，为了完成端到端的计算作业，Hadoop 采用 **YARN 来完成分布式资源调度**从而充分利用廉价的硬件资源，采用 HDFS 作为计算抽象之间的**数据接口**来规避廉价磁盘引入的系统稳定性问题。

由此可见，Hadoop 的“三招一套”自成体系，MapReduce 搭配 YARN 与 HDFS，几乎可以实现任何分布式批处理任务。然而，近乎完美的组合也不是铁板一块，每一只木桶都有它的短板。HDFS 利用副本机制实现数据的高可用从而提升系统稳定性，但额外的分片副本带来更多的磁盘 I/O 和网络 I/O 开销，众所周知，I/O 开销会严重损耗端到端的执行性能。更糟的是，一个典型的批处理作业往往需要多次 Map、Reduce 迭代计算来实现业务逻辑，因此上图中的计算流程会被重复多次，直到最后一个 Reduce 任务输出预期的计算结果。我们来想象一下，完成这样的批处理作业，在整个计算过程中需要多少次**落盘、读盘（IO）、发包、收包（网络传输）**的操作？因此，随着 Hadoop 在互联网行业的应用越来越广泛，人们对其 MapReduce 框架的执行性能诟病也越来越多。

**Spark Core**

时势造英雄，Spark 这孩子不仅天资过人，学起东西来更是认真刻苦。当别人都在抱怨老师 Hadoop 的 MapReduce 心法有所欠缺时，他居然已经开始盘算如何站在老师的肩膀上推陈出新。在 Spark 拜师学艺三年后的 2009 年，这孩子提出了“基于内存的分布式计算引擎”—— Spark Core，此心法一出，整个武林为之哗然。Spark Core 最引入注目的地方莫过于“内存计算”，这一说法几乎镇住了当时所有的初学者，大家都认为 Spark Core 的**全部计算都在内存中完成**，人们兴奋地为之奔走相告。兴奋之余，大家开始潜心研读 Spark Core 内功心法，才打开心法的手抄本即发现一个全新的概念 —— RDD。

**RDD**

RDD（Resilient Distributed Datasets），全称是“弹性分布式数据集”。全称本身并没能很好地解释 RDD 到底是什么，本质上，RDD 是 Spark 用于对分布式数据进行抽象的数据模型。简言之，RDD 是一种**抽象的数据模型**，这种数据模**型用于囊括、封装所有内存中和磁盘中的分布式数据实体**。对于大部分 Spark 初学者来说，大家都有一个共同的疑惑：Spark 为什么要提出这么一个新概念？与其正面回答这个问题，不如我们来反思另一个问题：Hadoop 老师的 MapReduce 框架，到底欠缺了什么？有哪些可以改进的地方？前文书咱们提到：MapReduce 计算模型**采用 HDFS 作为算子（Map 或 Reduce）之间的数据接口**，**所有算子的临时计算结果都以文件的形式存储到 HDFS 以供下游算子消费**。下游算子从 HDFS 读取文件并将其转化为键值对（江湖人称 KV），用 Map 或 Reduce 封装的计算逻辑处理后，再次以文件的形式存储到 HDFS。不难发现，问题就出在数据接口上。HDFS 引发的计算效率问题我们不再赘述，那么，有没有比 HDFS 更好的数据接口呢？如果能够将所有中间环节的数据文件以某种统一的方式归纳、抽象出来，那么所有 map 与 reduce 算子是不是就可以更流畅地衔接在一起，从而不再需要 HDFS 了呢？—— Spark 提出的 RDD 数据模型，恰好能够实现如上设想。

为了弄清楚 RDD 的基本构成和特性，我们从它的 5 大核心属性说起。

![img](https://pic1.zhimg.com/80/v2-09208543edf5e3266ab8f894fe54e290_1440w.jpg)

对于 RDD 数据模型的抽象，我们只需关注前两个属性，即 dependencies 和 compute。任何一个 RDD 都不是凭空产生的，**每个 RDD 都是基于一定的“计算规则”从某个“数据源”转换而来**。dependencies 指定了生成该 RDD 所需的“数据源”，术语叫作依赖或**父 RDD**；compute 描述了**从父 RDD 经过怎样的“计算规则”得到当前的 RDD**。这两个属性看似简单，实则大有智慧。

与 MapReduce 以算子（Map 和 Reduce）为第一视角、以外部数据为衔接的设计方式不同，Spark Core 中 RDD 的设计以数据作为第一视角，不再强调算子的重要性，算子仅仅是 RDD 数据转换的一种计算规则，map 算子和 reduce 算子纷纷被弱化、稀释在 Spark 提供的茫茫算子集合之中。dependencies 与 compute 两个核心属性实际上抽象出了“从哪个数据源经过怎样的计算规则和转换，从而得到当前的数据集”。父与子的关系是相对的，将思维延伸，如果当前 RDD 还有子 RDD，那么从当前 RDD 的视角看过去，子 RDD 的 dependencies 与 compute 则描述了“从当前 RDD 出发，再经过怎样的计算规则与转换，可以获得新的数据集”。

![img](https://pic3.zhimg.com/80/v2-f9ad7c787c74d9d5ca216959d9b4a49a_1440w.jpg)

不难发现，所有 RDD 根据 dependencies 中指定的依赖关系和 compute 定义的计算逻辑构成了一条从起点到终点的数据转换路径。这条路径在 Spark 中有个专门的术语，叫作 Lineage —— 血缘。Spark Core 依赖血缘进行依赖管理、阶段划分、任务分发、失败重试，任意一个 Spark 计算作业都可以析构为一个 Spark Core 血统。关于血统，到后文书再展开讨论，我们继续介绍 RDD 抽象的另外 3 个属性，即 partitions、partitioner 和 preferredLocations。相比 dependencies 和 compute 属性，这 3 个属性更“务实”一些。

在分布式计算中，**一个 RDD 抽象可以对应多个数据分片实体，所有数据分片构成了完整的 RDD 数据集。partitions 属性记录了 RDD 的每一个数据分片**，方便开发者灵活地访问数据集。partitioner 则描述了 RDD 划分数据分片的规则和逻辑，采用不同的 partitioner 对 RDD 进行划分，能够以不同的方式得到不同数量的数据分片。因此，partitioner 的选取，直接决定了 partitions 属性的分布。preferredLocations —— 位置偏好，该属性与 partitions 属性一一对应，定义了每一个数据分片的物理位置偏好。具体来说，每个数据分片可以有以下几种不同的位置偏好：

- 本地内存：数据分片已存储在当前计算节点的**内存**中，可就地访问
- 本地磁盘：数据分片在当前计算节点的**磁盘**中有副本，可就地访问
- 本机架磁盘：当前节点没有分片副本，但是同机架**其他机器**的磁盘中有副本
- 其他机架磁盘：当前机架所有节点都没有副本，但**其他机架**的机器上有副本
- 无所谓：当前数据分片没有位置偏好

根据**“数据不动代码动”**的原则，Spark Core 优先尊重数据分片的本地位置偏好，尽可能地**将计算任务分发到本地计算节点去处理**。显而易见，本地计算的优势来源于网络开销的大幅减少，进而从整体上提升执行性能。

RDD 的 5 大属性从“虚”与“实”两个角度刻画了对数据模型的抽象，任何数据集，无论格式、无论形态，都可以被 RDD 抽象、封装。前面提到，任意分布式计算作业都可以抽象为血统，而血统由不同 RDD 抽象的依次转换构成，因此，**任意的分布式作业都可以由 RDD 抽象之间的转换来实现**。理论上，**如果计算节点内存足够大，那么所有关于 RDD 的转换操作都可以放到内存中来执行，这便是“内存计算”的由来**。

