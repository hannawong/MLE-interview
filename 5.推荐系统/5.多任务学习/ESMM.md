# ESMM

本文介绍 阿里妈妈团队 发表在 SIGIR’2018 的论文《[Entire Space Multi-Task Model: An Eﬀective Approach for Estimating Post-Click Conversion Rate](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/1804.07931)》。文章基于 Multi-Task Learning 的思路，提出一种新的CVR预估模型——ESMM，有效解决了真实场景中CVR预估面临的**数据稀疏**以及**样本选择偏差**这两个关键问题。 实践出真知，论文一点也不花里胡哨，只有4页，据传被 SIGIR’2018 高分录用。

## 一、Motivation

不同于CTR预估问题，CVR预估面临两个关键问题：

1. **Sample Selection Bias (SSB)** 。转化是在点击之后才“有可能”发生的动作，传统CVR模型通常**以点击数据为训练集，其中点击未转化为负例，点击并转化为正例**。但是训练好的模型实际使用时，则是对**整个空间的样本**进行预估，而非只对点击样本进行预估。即是说，训练数据与实际要预测的数据**来自不同分布**，这个偏差对模型的泛化能力构成了很大挑战。
2. **Data Sparsity (DS)** 作为CVR训练数据的点击样本远小于CTR预估训练使用的曝光样本。

一些策略可以缓解这两个问题，例如**从impression中对unclicked样本抽样做负例缓解SSB**，对转化样本过采样缓解DS等。但无论哪种方法，都没有很elegant地从实质上解决上面任一个问题。

可以看到：点击—>转化，本身是两个强相关的连续行为，作者希望在模型结构中显示考虑这种“行为链关系”，从而可以在整个空间上进行训练及预测。这涉及到CTR与CVR两个任务，因此使用多任务学习（MTL）是一个自然的选择，论文的关键亮点正在于“如何搭建”这个MTL。

## 二、Model

介绍ESMM之前，我们还是先来思考一个问题——“**CVR预估到底要预估什么**”，论文虽未明确提及，但理解这个问题才能真正理解CVR预估困境的本质。想象一个场景，一个item，由于某些原因，例如在feeds中的展示头图很丑，它被某个user点击的概率很低，但这个item内容本身完美符合这个user的偏好，若user点击进去，那么此item被user转化的概率极高。CVR预估模型，预估的正是这个转化概率，**它与CTR没有绝对的关系，很多人有一个先入为主的认知，即若user对某item的点击概率很低，则user对这个item的转化概率也肯定低，这是不成立的。**更准确的说，**CVR预估模型的本质，不是预测“item被点击，然后被转化”的概率**（CTCVR）**，而是“假设item被点击，那么它被转化”的概率**（CVR）。这就是不能直接使用全部样本训练CVR模型的原因，因为咱们压根不知道这个信息：那些unclicked的item，"假设"他们被user点击了，它们是否会被转化。如果直接使用0作为它们的label，会很大程度上误导CVR模型的学习。

认识到点击（CTR）、转化（CVR）、点击然后转化（CTCVR）是三个不同的任务后，我们再来看三者的关联：

![[公式]](https://www.zhihu.com/equation?tex=%5Cunderbrace%7B+p%28z%5C%26y%3D1+%7C+%5Cbm%7Bx%7D%29+%7D_%7BpCTCVR%7D+%3D+%5Cunderbrace%7B+p%28z%3D1+%7Cy%3D1%2C+%5Cbm%7Bx%7D%29++%7D_%7BpCVR%7D+~+%5Cunderbrace%7B+p%28y%3D1+%7C+%5Cbm%7Bx%7D%29++%7D_%7BpCTR%7D%2C++~~~~~~~~~~~~~~~~~~~~~~~%281%29)

其中 ![[公式]](https://www.zhihu.com/equation?tex=z%2Cy) 分别表示conversion和click。注意到，在全部样本空间中，CTR对应的label为click，而CTCVR对应的label为click & conversion，**这两个任务是可以使用全部样本的**。**那为啥不绕个弯，通过这学习两个任务，再根据上式隐式地学习CVR任务呢？**ESMM正是这么做的，具体结构如下：

![img](https://pic2.zhimg.com/80/v2-d999a47e9ebfcc3fe1b61559b421e2c9_1440w.jpg)图1. ESMM网络结构

仔细观察上图，留意以下几点：1）**共享Embedding** CVR-task和CTR-task使用相同的特征和特征embedding，即两者从Concatenate之后才学习各自部分独享的参数；2）**隐式学习pCVR** 啥意思呢？这里pCVR（粉色节点）仅是网络中的一个**variable，没有显示的监督信号。**

具体地，反映在目标函数中：

![[公式]](https://www.zhihu.com/equation?tex=L%28%5Ctheta_%7Bcvr%7D%2C+%5Ctheta_%7Bctr%7D%29+%3D+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+l+%28+y_i%2C+f%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bctr%7D%29+%29+%2B+%5Csum_%7Bi%3D1%7D%5E%7BN%7D+l+%28+y_i%5C%26z_i%2C+f%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bctr%7D%29%2Af%28%5Cbm%7Bx%7D_i%3B%5Ctheta_%7Bcvr%7D+%29%29+%EF%BC%8C)

即利用**CTCVR和CTR**的监督信息来训练网络，【隐式】地学习CVR，这正是ESMM的精华所在，至于这么做的必要性以及合理性，本节开头已经充分论述了。

再思考下，ESMM的结构是基于“乘”的关系设计——pCTCVR=pCVR*pCTR，是不是也可以通过“除”的关系得到pCVR，即 pCVR = pCTCVR / pCTR ？例如分别训练一个CTCVR和CTR模型，然后相除得到pCVR，其实也是可以的，但这有个明显的缺点：真实场景预测出来的pCTR、pCTCVR值都比较小，“除”的方式容易造成数值上的不稳定。作者在实验中对比了这种方法。

## 三、Experiment

**数据集** 目前还没用同时包含点击、转化信息的公开数据集，作者从淘宝日志中抽取整理了一个数据集Product，并开源了从Product中随机抽样1%构造的数据集[Public](https://link.zhihu.com/?target=https%3A//tianchi.aliyun.com/dataset/dataDetail%3FdataId%3D408%26userId%3D1)（约38G）。

![img](https://pic3.zhimg.com/80/v2-dceac8cecc3ea415121917e0a7b9000a_1440w.png)表1. Public与Product数据集

**实验设置**

1. 对比方法：
   - BASE——图1左部所示的CVR结构，训练集为点击集；
   - AMAN——从unclicked样本中**随机抽样作为负例**加入点击集合；
   - OVERSAMPLING——对点击集中的正例（转化样本）过采样；
   - DIVISION——分别训练CTR和CVCTR，相除得到pCVR；
   - ESMM-NS——ESMM结构中CVR与CTR部分不share embedding。
2. 上述方法/策略都使用NN结构，relu激活函数，嵌入维度为18，MLP结构为360\*200\*80\*2，adam优化器 with ![[公式]](https://www.zhihu.com/equation?tex=%5Cbeta_1%3D0.9%2C+%5Cbeta_2%3D0.999%2C+%5Cepsilon%3D10%5E%7B-8%7D) 。
3. 按时间分割，1/2数据训练，其余测试

**衡量指标** 在点击样本上，计算CVR任务的AUC；同时，单独训练一个和BASE一样结构的CTR模型，除了ESMM类模型，其他对比方法均以pCTR*pCVR计算pCTCVR，在全部样本上计算CTCVR任务的AUC。

**实验结果**

如表1所示，ESMM显示了最优的效果。这里有趣的一点可以提下，ESMM是使用全部样本训练的，而CVR任务只在点击样本上测试性能，因此这个指标对ESMM来说是在biased samples上计算的，但ESMM性能还是很牛啊，说明其有很好的泛化能力。

![img](https://pic1.zhimg.com/80/v2-d8dcadaf947d25dbce48fca415d4c368_1440w.jpg)表2. 在Public上的实验结果，AUC以%为单位

在Product数据集上，各模型在不同抽样率上的AUC曲线如图2所示，ESMM显示的稳定的优越性，曲线走势也说明了Data Sparsity的影响还是挺大的。

![img](https://pic1.zhimg.com/80/v2-7c79179865aad9288b1a8ba6139e0844_1440w.jpg)图2. 在Product上，各模型在不同抽样率上的AUC曲线

## 四、Discussion

\1. ESMM 根据用户行为序列，显示引入CTR和CTCVR作为辅助任务，“迂回” 学习CVR，从而在完整样本空间下进行模型的训练和预测，解决了CVR预估中的2个难题。

\2. 可以把 ESMM 看成一个**新颖的 MTL 框架**，其中子任务的网络结构是可替换的，当中有很大的想象空间。至于这个框架的意义，这里引用论文作者之一[@朱小强的描述](https://zhuanlan.zhihu.com/p/54822778)：

> 据我所知这个工作在这个领域是最早的一批，但不唯一。今天很多团队都吸收了MTL的思路来进行建模优化，不过大部分都集中在传统的MTL体系，如研究怎么对参数进行共享、多个Loss之间怎么加权或者自动学习、哪些Task可以用来联合学习等等。ESMM模型的特别之处在于我们额外**关注了任务的Label域信息，**通过展现>点击>购买所构成的行为链，巧妙地构建了multi-target概率连乘通路**。**传统MTL中多个task大都是隐式地共享信息、任务本身独立建模，ESMM细腻地捕捉了契合领域问题的任务间显式关系，**从feature到label全面利用起来**。这个角度对互联网行为建模是一个较有效的模式，后续我们还会有进一步工作。