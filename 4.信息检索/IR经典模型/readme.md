

# 经典IR模型





## Representation-based IR Models(2017年之前)

基于表示的信息检索模型采用典型的双塔结构（这也是在召回中使用的结构）。query用query vector表示，document用document vector表示，最后计算相关度。

![img](https://pic3.zhimg.com/v2-2ac668d17db282ae246d7c52b3640496_b.png)





### 1. [Convolutional Neural Network Architectures for Matching Natural Language Sentences. (2015](https://arxiv.org/pdf/1503.03244.pdf)) 中的**ARC-I**

这篇文章出自华为诺亚方舟实验室，采用 **CNN** 模型来解决语义匹配问题。首先，文中提出了一种基于CNN的句子建模网络，如下图：

![img](https://pic4.zhimg.com/v2-e3580ba7aa3bdec67b2adceb3a7daccb_b.png)

图中灰色的部分表示对于长度较短的句子，其后面不足的部分填充的全是0值(Zero Padding)。图中的卷积计算和传统的CNN卷积计算无异，而池化则是使用Max-Pooling。

下面是基于之前的句子模型，建立的两种用于两个句子的匹配模型。

ARC-1：

![img](https://pic1.zhimg.com/v2-e40b13f6bd05f711a2b94eadcab1c0b0_b.png)

这个模型比较简单，但是有一个较大的缺点：两个句子在建模过程中是完全独立的，**没有任何交互行为**，一直到最后生成抽象的向量表示后才有交互，这样做使得句子在抽象建模的过程中会丧失很多语义细节，同时过早地失去了句子间语义交互计算的机会。（这也是双塔模型被诟病的原因）



#### 2. [Deep Semantic Similarity Model(DSSM, 2013)](https://posenhuang.github.io/papers/cikm2013_DSSM_fullversion.pdf)

这篇文章出自微软2013年的文章，算是双塔最早的文章之一。



![img](https://pic2.zhimg.com/v2-c909f2e53530ee888ceff90f7cdfbb8d_b.png)

输入层为经过word-hashing之后的30k维结果，输出为计算的query和文档之间的cosine similarity。

文中提到的word hashing方法是为了解决token过多的问题（对于one-hot编码），同时解决OOV，但是会在一定程度上带来一些哈希冲突。文中提到的character-trigram在今天还经常使用，作为BPE、word embedding、n-gram word embedding的必要补充。

![img](https://pic3.zhimg.com/v2-608a5df2edcdf6e818a94d5ccae13b32_b.png)

![img](https://pic3.zhimg.com/v2-6a796d041a8ab1dd9b2cae4bd4021fee_b.png)



#### 3. [Convolutional Latent Semantic Model (CLSM,2014)](https://www.iro.umontreal.ca/~lisa/pointeurs/ir0895-he-2.pdf)



![img](https://pic4.zhimg.com/v2-86d45f725d6c73ff3191ef28351955c7_b.png)



CLSM(convolutional latent semantic model) 主要的思想是使用CNN模型来提取语料的语义信息，卷积层的使用保留了词语的上下文信息，池化层的使用提取了对各个隐含语义贡献最大的词汇。

首先用一个滑动窗口得到word-n-gram, 然后通过word-hashing得到每个窗口的向量表示。max-pooling层把不同的滑窗的最大信息提取出来，构成一个fix-length vector。本文和DSSM最大的区别就是不是词袋模型了，而是考虑了滑窗的位置信息。

![img](https://pic1.zhimg.com/v2-bc710de19b1ef9a7b296733529b00b20_b.png)

## 基于交互的模型



![img](https://pic2.zhimg.com/80/v2-1febb520a383bfe330e086ff64d1dde9_1440w.jpg)

representation-based IR model（双塔模型）和Interaction-based IR model（**金字塔模型**）的区别可以形象地表示为下图：

![img](https://pic1.zhimg.com/80/v2-c5e6fe48437d5b5989d964db0c30e9d8_1440w.jpg)

代表论文：

1）[Convolutional Neural Network Architectures for Matching Natural Language Sentences. (201](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1503.03244.pdf)5) 中的ARC-II：

![img](https://pic1.zhimg.com/80/v2-be1db2a8f2a727314779e67ebfca7a14_1440w.jpg)

上图所示的 ARC-II 在第 1 层卷积后就把文本 X 和 Y 做了融合，具体的融合方式是，首先从Sentence x中任取一个向量 ![[公式]](https://www.zhihu.com/equation?tex=x_a) ，再从Sentence y中将每一个向量和![[公式]](https://www.zhihu.com/equation?tex=x_a)进行一维卷积操作，这样就构造出一个 2D 的 feature map，然后对其做 2D MAX POOLING，多次 2D 卷积和池化操作后，输出固定维度的向量，接着输入 MLP 层，最终得到文本相似度分数。

2）MatchPyramid

[https://arxiv.org/pdf/1606.04648.pdfarxiv.org/pdf/1606.04648.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1606.04648.pdf)

![img](https://pic1.zhimg.com/80/v2-c1676b2a260533280265be1bb37755dc_1440w.jpg)

其中， ![[公式]](https://www.zhihu.com/equation?tex=%5Cotimes) 是计算相似度的符号。文中提出四种计算相似度的方法：

![img](https://pic4.zhimg.com/80/v2-5c85ea1f1c4b1f5e54a764e7a701c197_1440w.jpg)

**3）Deep Relevance Matching Model (DRMM)**

[https://arxiv.org/pdf/1711.08611.pdfarxiv.org/pdf/1711.08611.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1711.08611.pdf)

![img](https://pic1.zhimg.com/80/v2-e54b6de0b1b4bdd5e34322d546d074b4_1440w.jpg)对于query的每个token，都去计算它和document的每个token的相似度，然后分桶，作为若干层MLP的输入

![img](https://pic2.zhimg.com/80/v2-f493ce00d69205b1ae12c246dddb0f7d_1440w.jpg)

![[公式]](https://www.zhihu.com/equation?tex=%5Cotimes) 表示计算相似度，h表示分桶函数，g是门控。

![img](https://pic2.zhimg.com/80/v2-0b5a1cdd00676ebed32a277e0dd5bc7d_1440w.jpg)

**4) Kernel-based Neural Ranking Model (K-NRM, 2017)**

[https://arxiv.org/pdf/1706.06613.pdfarxiv.org/pdf/1706.06613.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1706.06613.pdf)

本文和 MatchPyramid 的核心的不同之处在于 RBF Kernel：

![img](https://pic4.zhimg.com/80/v2-74ddd7f9bd02dcc1775f0a6e7531a687_1440w.jpg)

先把 term 映射为 Word Embedding,再计算两两相似度矩阵M，然后通过 RBF Kernel：



![img](https://pic4.zhimg.com/80/v2-a88782a21346c300d74471a94852ec6f_1440w.jpg)



translation矩阵的每一行经过kernel之后都变成一个数，总共有K个kernel。

最后把所有的soft-TF输出取log相加，再过若干层MLP输出结果。采用 pairwise learning to rank loss 进行训练：

![img](https://pic4.zhimg.com/80/v2-2002a2c87a76c80ad0d2c5e7041d0513_1440w.png)

**5）Conv-KNRM (2018)**

[http://www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdfwww.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf](https://link.zhihu.com/?target=http%3A//www.cs.cmu.edu/~zhuyund/papers/WSDM_2018_Dai.pdf)

Conv-knrm相比k-nrm，最大的改变就是它添加了n-gram的卷积，增加了原先模型的层次，它能够捕捉更加细微的语义实体，交叉的粒度也更加细。

![img](https://pic3.zhimg.com/80/v2-00b748512714b6417e3581f890e23716_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-310eae5a13c4793c99b6aa4c28214594_1440w.jpg)

**6) ⭐BERT (2019)**

![img](https://pic4.zhimg.com/80/v2-7f71a868565f1648da368702e076c503_1440w.jpg)

![img](https://pic1.zhimg.com/80/v2-c65d65a9335a830e6880ae11a828da2c_1440w.jpg)

![img](https://pic2.zhimg.com/80/v2-fa959513522eea091c597549dcbb46ed_1440w.jpg)

### 4. Further Combination

**Duet model (2017)**

[https://arxiv.org/pdf/1610.08136.pdfarxiv.org/pdf/1610.08136.pdf](https://link.zhihu.com/?target=https%3A//arxiv.org/pdf/1610.08136.pdf)

- local model: if each term is represented by a unique identifiers (local representation) then the query-document relevance is a function of the pattern of occurrences of the **exact** query terms in the document.
- distributed model: if the query and the document text is first projected into a continuous latent space, then it is their distributed representations that are compared.

Local model 与 distributed model 各有所长。Local model具有记忆能力，distributed model具有泛化能力。例如，distributed model会非常了解"Barack Obama" (因为训练语料很多)，而不了解"Bhaskar Mitra",所以在后者上会表现较差(甚至接近于随机初始化！)。Local model对"Barack Obama"和"Bhaskar Mitra"都没有了解，在local model看来，它们不过是一些token。但是在"Bhaskar Mitra"上，local model会表现得比distributed model出色。所以，为什么不把二者结合起来呢？

![img](https://pic1.zhimg.com/80/v2-2239de7e24c41c64fb42a0be3452cc3c_1440w.jpg)Local Model全部靠精确匹配

![img](https://pic1.zhimg.com/80/v2-52d72cf8c1896e5802a2bd9c22fad3e8_1440w.jpg)local model是exact-match

我的理解是，exact-match和semantic similarity都很重要。现在工业界的搜索召回一般都是多路召回，其中用关键词去elasticsearch等搜索框架中去搜依然是很重要的一路召回。但是还要加上语义向量检索等多路召回以辅助，这样才能把用户可能感兴趣的item找出来。如果光用exact match，会导致很多和query相关的item搜不出来，比如搜"sneaker"不会出来"running shoes"；如果只用semantic search,会导致一些不相关的结果出现，比如搜"adidas shoes"会出现"nike sneakers".