 # 词性标注(POS tagging) 入门级教程 -- HMM和CRF

这是【命名实体识别】相关的文章第一篇，主要介绍一些POS tagging入门级的内容，包括HMM与Viterbi算法、CRF的知识。相信读完这一篇之后，会对POS tagging的baseline方法有一个基本的认识。

本文参考：https://zhuanlan.zhihu.com/p/104562658

https://www.freecodecamp.org/news/a-deep-dive-into-part-of-speech-tagging-using-viterbi-algorithm-17c8de32e8bc

----

词性标注任务就是给每个词语标上动词、名词、副词、形容词这样的标签，就像下面的图这样：

![img](https://miro.medium.com/max/993/1*AyPfvdmnz5uF2UIiOX2ctA.png)

POS tagging给出了一个词在句子中的**词性角色**，因此可以用来做命名实体识别(NER)，指代消解等。

#### 1. Markov Chain

例如，一个不完整的句子：“Why not tell … ”，下面的一个单词最可能是什么词性呢？熟悉英语的人都知道，下面大概率是一个名词。如果我们假设每个位置的词性只和**上一个词**的词性有关，这就符合**一阶**markov假设了。转移矩阵如下所示：

![img](https://miro.medium.com/max/927/1*toVjZDV525ptkAQhoM6e4A.png)

转移矩阵中每个位置的值都是由**统计**的方式得到的。

#### 2. Hidden Markov Model

带隐藏状态的Markov 过程则称为Hidden Markov Model。在POS tagging问题中，**隐藏状态则是词语的POS tagging**，因为它们是不能直接从句子中看出来的。和Markov Model一样，HMM也有转移概率（描述从一个隐藏状态转移到另一个隐藏状态的概率），除此之外还有**emission probability**（从隐藏状态，即POS tagging，转移到可观察状态，即词语，的概率）

![img](https://miro.medium.com/max/866/1*sSfQ0DJnQNLC-illR3UdSQ.png)

emission probability的矩阵：

![img](https://miro.medium.com/max/1035/1*ITb6TnMqjNhVaiXQBfvibQ.png)

记得我们之前讲过生成式模型和判别式模型吧，对于生成式模型，我们是要求 $argmax_{y} P(x,y)$ . 即是求label y，使得P(x,y)最大。

对于POS问题，$(x_1,x_2,...x_n)$ 表示文本，$(y_1,y_2,...y_n)$ 表示文本每个词对应的POS。那么**第一个问题**就是，如何求 $P(x_1,x_2,...x_n,y_1,y_2,...y_n)$ 呢？就像下面的这个句子和对应的tagging，我们可以求出它的likelihood嘛？

![35SQfert2ZVmpA4biNBYbdh18x1E8CaxpfYI](https://cdn-media-1.freecodecamp.org/images/35SQfert2ZVmpA4biNBYbdh18x1E8CaxpfYI)

直接上公式：

![QXdufboQ1sB0ZSP3vta1yiteOpT47xDCy6xf](https://cdn-media-1.freecodecamp.org/images/QXdufboQ1sB0ZSP3vta1yiteOpT47xDCy6xf)

注意这里我们是用**二阶markov**假设，所以 $q(y_i|y_{i-2},y_{i-1})$ 指的是前两个POS状态分别是 $y_{i-2},y_{i-1}$ 的情况下，下一个状态为 $y_i$ 的概率。$e(x_i|y_i)$ 是emission probability。

需要指出的是，$q(y_i|y_{i-2},y_{i-1})$和$e(x_i|y_i)$都是可以很容易的用训练数据估计出来的，只要用统计方法就可以了：

![c9IK15ggYYCC2jj7xqv49szM4T9z865wOYZW](https://cdn-media-1.freecodecamp.org/images/c9IK15ggYYCC2jj7xqv49szM4T9z865wOYZW)

一个数据集的实例如下所示，蓝色框框出来的就是三元组，用来训练$q(y_i|y_{i-2},y_{i-1})$的，而红色框框出来部分用来训练$e(x_i|y_i)$。

![fjB8BXYUF0A3PMGLF1Hwt2E4ueO0VwLfhea8](https://cdn-media-1.freecodecamp.org/images/fjB8BXYUF0A3PMGLF1Hwt2E4ueO0VwLfhea8)

###### Finding the most probable sequence — Viterbi Algorithm

上面我们已经知道了怎么去求 $P(x_1,x_2,...x_n,y_1,y_2,...y_n)$ 。那么一个暴力的方法就是遍历所有可能的POS tagging序列 $(y_1,y_2,...y_n)$ , 然后求出使得 $P(x_1,x_2,...x_n,y_1,y_2,...y_n)$ 最大的那个序列$(y_1,y_2,...y_n)$. 但是，这样的做法复杂度是**指数级别**的。但是好消息是，我们可以用基于动态规划的Viterbi算法，来把复杂度降到多项式时间复杂度。

首先，我们来看前k个词语，它的$P(x_1,x_2,...x_k,y_1,...y_k)$为：

​                                                    $r(y_1,...,y_k) = \prod_{i=1}^k q(y_i|y_{i-2},y_{i-1}) \prod_{i=1}^k e(x_i|y_i)$ 

$S(k, u, v)$定义为序列长度为k，且隐藏状态以(u, v)为结尾的那些序列的集合。

$π(k, u, v)$ 定义为序列长度为k，且隐藏状态以(u, v)为结尾的那些序列中，取得$P(x_1,x_2,...x_k,y_1,...y_k)$的最大值  

显然，$π(n, u, v)$就是我们要求的东西，它可以用动态规划的方式求得。

一个序列`*    *   x1   x2   x3   ......    xn`,  首先对dp数组做初始化。$π(0, *, *) = 1$, 其他都为0. 

![img](https://picx.zhimg.com/80/v2-6f034c767fbb9b9241a61a186f61f7f0_720w.png)

![RNHTlxO-aqNvguCPosqS0pkGoS1M1gA12iKy](https://cdn-media-1.freecodecamp.org/images/RNHTlxO-aqNvguCPosqS0pkGoS1M1gA12iKy)

![img](https://pic2.zhimg.com/80/v2-7c2426e733753239d94745bd9f00bdcd_720w.png)

所以复杂度为 O(n|K|³)。如果只考虑二元组，那么复杂度就是O(n|K|²)。



#### 3. CRF（条件随机场）

先来看一句话：“Bob drank coffee at Starbucks”，注明每个单词的词性后是这样的：“Bob (名词) drank(动词) coffee(名词) at(介词) Starbucks(名词)”。下面，就用CRF来解决这个问题。
以上面的话为例，有5个单词，我们将：**(名词，动词，名词，介词，名词)** 作为一个标注序列，称为 $I$ ，可选的标注序列有很多种，比如还可以是这样：**（名词，动词，动词，介词，名词）**，我们要在这么多的可选标注序列中，挑选出一个**最靠谱**的作为我们对这句话的标注。
**怎么判断一个标注序列靠谱不靠谱呢？**
就我们上面展示的两个标注序列来说，第二个显然不如第一个靠谱，因为它把第二、第三个单词都标注成了动词，动词后面接动词，这在一个句子中通常是说不通的。
假如我们给每一个标注序列**打分**，打分越高代表这个标注序列越靠谱，我们至少可以说，凡是标注中出现了**动词后面还是动词**的标注序列，要给它**负分！！**
上面所说的动词后面还是动词就是一个**特征函数**，我们可以定义一个**特征函数集合**，用这个特征函数集合来为一个标注序列打分，并据此选出最靠谱的标注序列。把所有特征函数对同一个标注序列的评分综合起来，就是这个标注序列最终的评分值。

###### 特征函数

现在，我们正式地定义一下什么是CRF中的特征函数，所谓特征函数，就是这样的函数，它接受四个参数：

- 句子 $s$（就是我们要标注词性的句子）
- $i$ ，用来表示句子s中第i个单词
- $l_i$，表示要评分的标注序列给第i个单词标注的**词性**
- $l_{i-1}$，表示要评分的标注序列给第i-1个单词标注的词性

它的输出值是0或者1,0表示要评分的标注序列不符合这个特征，1表示要评分的标注序列符合这个特征。

**Note:**这里，我们的特征函数仅仅依靠**当前单词的标签和它前面的单词的标签**对标注序列进行评判，这样建立的CRF也叫作线性链CRF，这是CRF中的一种简单情况。为简单起见，本文中我们仅考虑线性链CRF(也就是符合**一阶Markov**假设)。

######  从特征函数到概率

定义好一组特征函数后，我们要给每个特征函数$f_j$赋予一个权重$λ_j$。现在，只要有一个句子$s$，有一个标注序列 $l$，我们就可以利用前面定义的特征函数集来对$l$评分。

![img](https://pic4.zhimg.com/80/v2-a9ea6ecb86be324fcc0b0df5e4963a77_720w.jpg)

上式中有两个求和，外面的求和用来求**每一个特征函数**$f_j$评分值的和，里面的求和用来求句子中**每个位置**的单词的的特征值的和。

对这个分数进行**指数化和标准化**，我们就可以得到这个标注序列 $l$ 的**概率值**$p(l|s)$，如下所示：

![img](https://pic3.zhimg.com/80/v2-1cf129da6abc1d75af0488b21a82149e_720w.jpg)

可见，分母就是所有可能的标注序列$l'$的得分之和。

###### 几个特征函数的例子

下面我们再看几个具体的特征函数设计的例子：

- 当第$i$个单词以"-ly"结尾，并且$l_i$标记为“副词”，特征函数$f_1 = 1$，其他情况$f_1$为0。不难想到，$f_1$特征函数的权重$λ_1$应当是正的，因为我们倾向于把以"-ly"结尾的单词标注为“副词”。

- 如果$l_1$="动词"，并且句子$s$是以“？”结尾时，特征函数$f_2=1$，其他情况$f_2=0$。同样，$λ_2$应当是正的，因为我们越倾向把问句的第一个单词标注为“动词”。

- 如果$l_i$和$l_{i-1}$都是介词，那么特征函数$f_3$等于1，其他情况$f_3=0$。这里，我们应当可以想到$λ_3$是负的，因为我们不认可介词后面还是介词的标注序列。

总结一下：
为了建一个条件随机场，我们首先要定义一个**特征函数集**，然后为每一个特征函数赋予一个**权重**，然后针对每一个标注序列$I$，对所有的特征函数加权求和；之后，把这个得到的Score转化成概率值。

##### CRF与逻辑回归相似性

观察公式：

![img](https://pic3.zhimg.com/80/v2-90900527c9206df1b45081c31d5053fe_720w.jpg)

是不是有点逻辑回归的味道？

事实上，**条件随机场是逻辑回归的序列化版本。逻辑回归是用于分类的对数线性模型，条件随机场是用于序列化标注的对数线性模型。**

##### CRF与HMM的比较

对于词性标注问题，HMM模型也可以解决。HMM的思路是用生成办法，就是说，在已知要标注的句子$s$的情况下，去判断生成标注序列$l$的概率，如下所示：

![img](https://pic1.zhimg.com/80/v2-7a05244a7e166b70e0c1404a4feda34c_720w.jpg)
其中，

- $p(l_i|l_i-1)$ 是转移概率，比如$l_{i-1}$是介词而$l_i$是名词的概率。

- $p(w_i|l_i)$表示发射概率（emission probability)，比如$l_i$是"名词"，$w_i$是单词“ball”，此时的发射概率则表示：在是名词的条件下，单词为“ball”的概率。

那么，HMM和CRF有什么区别和联系呢？
答案是：CRF比HMM要强大，因为它可以解决所有HMM能够解决的问题，并且还可以解决许多HMM解决不了的问题。事实上，对上面的HMM模型取对数，就变成下面这样：

![img](https://pic4.zhimg.com/80/v2-cda191ab097dca1fbd91f8c4baf498bb_720w.jpg)

我们把这个式子与CRF的式子进行比较：

![img](https://pic1.zhimg.com/80/v2-804a62f01cae5dd45e9448cc67203174_720w.jpg)

不难发现，我们可以构造一个CRF，使它与HMM的对数形式相同。怎么构造呢？
对于HMM中的每一个转移概率 $p(l_i=y|l_{i-1}=x)$ ,我们就可以定义这样的一个特征函数：

![img](https://pic4.zhimg.com/80/v2-efb0b3ef55448e99f5413c4c4ba50317_720w.jpg)

该特征函数仅当$l_{i-1}=x, l_i = y$时才等于1。这个特征函数的权重就是：

![img](https://pic1.zhimg.com/80/v2-60da284a9bc8f934e2bbfbdb553a2008_720w.jpg)

同样的，对于HMM中的每一个emission probability，我们也都可以定义相应的特征函数，并让该特征函数的权重等于HMM中的emission probability的log值。
因此，**每一个HMM模型都等价于某个CRF**
但是，CRF要比HMM更加强大，原因主要有两点：

- **CRF可以定义种类更丰富的特征函数**。一阶HMM模型有个重要假设，即当前的标签只依赖于前一个标签。这样的假设限制了HMM只能定义很有局限的特征函数。但是CRF却可以着眼于**整个句子s**定义更具有全局性的特征函数，如"如果$l_1$=动词，并且句子s是以"?"结尾时，$f_2 = 1$，其他情况$f_2=0$。

- **CRF可以使用任意的权重。**将对数HMM模型看做CRF时，特征函数的权重就是转移概率/emission probability的log形式；但在CRF中，每个特征函数的权重可以是任意值，没有限制。

