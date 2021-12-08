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