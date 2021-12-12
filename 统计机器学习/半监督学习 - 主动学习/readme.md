# 主动学习



只在实习中接触了一点点，所以不做深入展开，只介绍基本概念。面试官倘若提起能说上几句就好。



### 0x01. 背景

机器学习的研究领域包括**有监督学习（Supervised Learning）**，**无监督学习（Unsupervised Learning）**，**半监督学习（Semi-supervised Learning）**和**强化学习（Reinforcement Learning）**等诸多内容。针对**有监督学习和半监督学习**，都需要一定数量的标注数据，也就是说在训练模型的时候，全部或者部分数据需要带上相应的标签才能进行模型的训练。但是在实际的业务场景或者生产环境中，工作人员获得样本的成本其实是比较高的，那么如何通过较少成本 来获得较大价值的标注数据，进一步地提升算法的效果就是值得思考的问题了。

在很多特殊的业务场景上，其实都没有一个大规模的公开数据集。从业人员需要想尽办法去获取业务标注数据。在安全风控领域，黑产用户相对于正常用户是偏少的，因此，如何通过极少的黑产用户来建立模型则是值得思考的问题之一。在业务运维领域，服务器/app 的故障时间相对于正常运行的时间也是偏少的，必然会出现样本不均衡的情况。因此，在这些业务领域，要想获得样本和构建模型，就必须要通过人力的参与。那么如何通过一些机器学习算法来降低人工标注的成本就是从业者需要关注的问题了。毕竟需要标注 100 个样本和需要标注成千上万的样本所需要的人力物力是截然不同的。

在学术界，同样有学者在关注这方面的问题，学者们通过一些技术手段或者数学方法来降低人们标注的成本，学者们把这个方向称之为**主动学习（Active Learning）**。在整个机器学习建模的过程中有人工参与的部分和环节，并且通过机器学习方法**筛选出合适的候选集给人工标注**的过程。主动学习（Active Learning）的大致思路就是：通过机器学习的方法获取到那些比较**“难”**分类的样本数据，让人工再次确认和审核，然后将人工标注得到的数据再次使用有监督学习模型或者半监督学习模型进行训练，逐步提升模型的效果，将人工经验融入机器学习的模型中。

在没有使用主动学习（Active Learning）的时候，通常来说系统会从样本中**随机**选择或者使用一些人工规则的方法来提供待标记的样本供人工进行标记。这样虽然也能够带来一定的效果提升，但是其标注成本总是相对大的。

用一个例子来比喻，一个高中生通过做高考的模拟试题以希望提升自己的考试成绩，那么在做题的过程中就有几种选择。一种是随机地从历年高考和模拟试卷中随机选择一批题目来做，以此来提升考试成绩。但是这样做的话所需要的时间也比较长，针对性也不够强；另一种方法是每个学生建立自己的错题本，用来记录自己容易做错的习题，反复地巩固自己做错的题目，通过多次复习自己做错的题目来巩固自己的易错知识点，逐步提升自己的考试成绩。其主动学习的思路就是选择一批容易被错分的样本数据，让人工进行标注，再让机器学习模型训练的过程。

那么主动学习（Active Learning）的整体思路究竟是怎样的呢？在机器学习的建模过程中，通常包括样本选择，模型训练，模型预测，模型更新这几个步骤。在主动学习这个领域则需要把标注候选集提取和人工标注这两个步骤加入整体流程，也就是：

1. 机器学习模型：包括机器学习模型的训练和预测两部分；
2. 待标注的数据候选集提取：依赖主动学习中的**查询函数**（Query Function）；
3. 人工标注：专家经验或者业务经验的提炼；
4. 获得候选集的标注数据：获得更有价值的样本数据；
5. 机器学习模型的更新：通过增量学习或者重新学习的方式更新模型，从而将人工标注的数据融入机器学习模型中，提升模型效果。

![img](https://pic2.zhimg.com/80/v2-2ae5752e375e831b152457ae013f22f9_1440w.jpg)



其应用的领域包括：

1. 个性化的垃圾邮件，短信，内容分类：包括营销短信，订阅邮件，垃圾短信和邮件等等；
2. 异常检测：包括但不限于安全数据异常检测，黑产账户识别，时间序列异常检测等等。

（在实习中的应用场景是数据库组件化每个字段的类型预测）



### 0x02. 查询策略

查询策略（Query Strategy Frameworks）就是主动学习的核心之处。只介绍不确定性采样的查询（**Uncertainty Sampling**）

##### **不确定性采样（Uncertainty Sampling）**

顾名思义，不确定性采样的查询方法就是将模型中**难以区分**的样本数据提取出来，提供给业务专家或者标注人员进行标注，从而达到以较快速度提升算法效果的能力。而不确定性采样方法的关键就是如何描述样本或者数据的不确定性，通常有以下几种思路：

1. 置信度最低（Least Confident）；
2. 边缘采样（Margin Sampling）；
3. 熵方法（Entropy）；

##### **(1) Least Confident**

对于二分类或者多分类的模型，通常它们都能够对每一个数据进行打分，判断它究竟更像哪一类。例如，在二分类的场景下，有两个数据分别被某一个分类器预测，其对两个类别的预测概率分别是：(0.9,0.1) 和 (0.51, 0.49)。在此情况下，第一个数据被判定为第一类的概率是 0.9，第二个数据被判定为第一类的概率是 0.51，于是第二个数据明显更“难”被区分，因此更有被继续标注的价值。所谓 Least Confident 方法就是选择那些最大概率最小的样本进行标注，用数学公式描述就是：

![[公式]](https://www.zhihu.com/equation?tex=x_%7BLC%7D%5E%7B%2A%7D%3Dargmax_%7Bx%7D%281-P_%7B%5Ctheta%7D%28%5Chat%7By%7D%7Cx%29%29%3Dargmin_%7Bx%7DP_%7B%5Ctheta%7D%28%5Chat%7By%7D%7Cx%29) ,

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D%3Dargmax_%7By%7DP_%7B%5Ctheta%7D%28y%7Cx%29) ，这里的 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta) 表示一个已经训练好的机器学习模型参数集合。 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D) 对于 ![[公式]](https://www.zhihu.com/equation?tex=x) 而言是模型预测概率最大的类别。Least Confident 方法考虑那些模型预测概率最大但是可信度较低的样本数据。



##### (2)Margin Sampling

边缘采样（[margin sampling](https://www.zhihu.com/search?q=margin+sampling&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A239756522%7D)）指的是选择那些极容易被判定成两类的样本数据，或者说这些数据被判定成两类的概率相差不大。边缘采样就是选择模型预测最大和第二大的[概率差值](https://www.zhihu.com/search?q=%E6%A6%82%E7%8E%87%E5%B7%AE%E5%80%BC&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A239756522%7D)最小的样本，用数学公式来描述就是：

![[公式]](https://www.zhihu.com/equation?tex=x_%7BM%7D%5E%7B%2A%7D%3Dargmin_%7Bx%7D%28P_%7B%5Ctheta%7D%28%5Chat%7By%7D_%7B1%7D%7Cx%29-P_%7B%5Ctheta%7D%28%5Chat%7By%7D_%7B2%7D%7Cx%29%29) ,

其中 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7B1%7D) 和 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat%7By%7D_%7B2%7D)分别表示对于 ![[公式]](https://www.zhihu.com/equation?tex=x) 而言，模型预测为最大可能类和第二大可能类。

特别地，如果针对二分类问题，[least confident](https://www.zhihu.com/search?q=least+confident&search_source=Entity&hybrid_search_source=Entity&hybrid_search_extra=%7B%22sourceType%22%3A%22article%22%2C%22sourceId%22%3A239756522%7D) 和 margin sampling 其实是等价的。

##### **(3) Entropy**

在数学中，可以使用熵（Entropy）来衡量一个系统的不确定性，熵越大表示系统的不确定性越大，熵越小表示系统的不确定性越小。因此，在二分类或者多分类的场景下，可以选择那些**熵比较大**的样本数据作为待定标注数据。用数学公式表示就是：

![[公式]](https://www.zhihu.com/equation?tex=x_%7BH%7D%5E%7B%2A%7D%3Dargmax_%7Bx%7D-%5Csum_%7Bi%7DP_%7B%5Ctheta%7D%28y_%7Bi%7D%7Cx%29%5Ccdot+%5Cln+P_%7B%5Ctheta%7D%28y_%7Bi%7D%7Cx%29) ,

相较于 least confident 和 margin sample 而言，entropy 的方法考虑了该模型对某个 ![[公式]](https://www.zhihu.com/equation?tex=x) 的所有类别判定结果。而 least confident 只考虑了最大的概率，margin sample 考虑了最大的和次大的两个概率。