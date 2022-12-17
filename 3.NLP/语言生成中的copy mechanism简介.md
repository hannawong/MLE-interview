# 语言生成中的copy mechanism简介

## 1. 背景

在NLP领域中，由于词表大小的限制，很多低频词无法被纳入词表，这些词即为**OOV**（Out Of Vocabulary）。它们会统一地被表示为[UNK]，其语义信息也被丢弃。由于Zipf's定律，低频词语会在词表中占据很大的空间。另外，一些人名、地名等专有名词，作为特定的**命名实体**，很多无法被纳入词表。

解决OOV的方法有：

（1）扩大词表：扩大词表后，可以将部分低频词纳入了词表，但是这些低频词由于缺乏足够数量的语料，训练出来的词向量往往效果不佳。

（2）Copy Mechanism：这正是本文将要介绍的方法。

（3）subword：使用BPE等子词分割的方法。

另外，在文本摘要中，一个很自然的方法就是把原文中一些词和短语原封不动地“抄过来”，这也正是copy mechanism的思想。

## 2.1 Pointing the Unknown Words (2016)

在decoder的每个时间步都用两个softmax，一个用来在一个有限的词表中获得每个词的概率分布、另一个用来获得输入句子中每个词的概率分布。另外，用一个Switching Network来判断要选择哪个softmax来进行预测。

**Shortlist Softmax**：将decoder hidden state经过全连接，输出**词表**里的所有词的条件概率。选中词 wt 的概率为 p(wt|zt=1,(y,z)<t,x) ,其中 zt=1 表示这一步模型选择从词表的概率。

![img](https://pic4.zhimg.com/80/v2-eca612d7b89cd0e7a4376d372b5a0baf_1440w.webp)

选择词语w_t的概率 * 决定从词表中选的概率

**Location Softmax：**直接利用了Attention机制中分配给每个输入词的权重，这个权重的向量长度是源序列的长度。选择位置 lt 的概率为： p(lt|zt=0,(y,z)<t,x) . 其中 zt=0 表示这一步模型选择从输入中copy的概率。

![img](https://pic3.zhimg.com/80/v2-2a20f478b019748775dfc57e12896276_1440w.webp)

选择位置l_t的概率 * 决定从输入中copy的概率

**Switching Network：**负责选择是采纳Shortlist Softmax输出的预测词、还是采纳Location Softmax输出的预测词。Switching Network由一个多层感知机MLP实现，并在最后接一个sigmoid激活函数：

![img](https://pic1.zhimg.com/80/v2-9f2d14f625915da4ae3a1bf8129c2ca8_1440w.webp)

## 2.2 Get To The Point: Summarization with Pointer-Generator Networks(2017)

本论文模型被应用于文本摘要任务，通过同时得到词表和输入序列中词的概率分布，能够实现摘要时抽取式(abstractive)和生成式(generative)的平衡。

一个经典的seq2seq架构中，我们对于每个decoder位置都得到词表的概率分布 pvocab ; 对于输入序列计算出的权重向量为 at , 计算出输入的context vector ht∗=∑iaithi .

在decoder 的 step t，从词表中选词的概率 pgen 是根据context vector ht∗ 、decoder hidden state st 、decoder 输入 xt 计算出来的：

![img](https://pic2.zhimg.com/80/v2-cc0a475962a684fbbceaaaddb5052279_1440w.webp)

那么，生成一个词语 w 的概率就是从vocab中生成出来的概率+从输入copy过来的概率：

![img](https://pic1.zhimg.com/80/v2-468639b926f146d1293b8945e19d99bc_1440w.webp)

模型结构如下：

![img](https://pic3.zhimg.com/80/v2-ab7b898d2e3fc31f62489f28e1b30be2_1440w.webp)

此外，文中还提出了Coverage Mechanism，这是用来惩罚词语重复使用的。coverage vector ct 是先前所有step的attention权重累积和，它反映了在之前所有步中对于输入序列中每个词的关注程度：

![img](https://pic2.zhimg.com/80/v2-9d2ee9966b684b13606610d9a80f7fd1_1440w.webp)

在计算attention的时候，把 ct 纳入其中进行考量，即可以避免过多的关注同一个位置：

![img](https://pic2.zhimg.com/80/v2-d893494e6ac50820c692b0da7dd94b69_1440w.webp)



发布于 2022-11-30 21:35・IP 属地美国