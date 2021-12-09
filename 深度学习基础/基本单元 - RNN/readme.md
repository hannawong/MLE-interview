# 基本单元 - RNN

RNN是处理**序列数据**的算法，RNN使用内部存储器(internal memory)来记住其输入，这使其非常适合涉及序列数据的问题。

本文介绍引入RNN的问题--Language Model，并介绍RNN的公式。

### 1. Language Model

在介绍RNN之前，我们先介绍最初引入RNN的问题---**Language Modeling**。

**定义：**Language Modeling就是预测下一个出现的词的概率的任务。(Language Modeling is the task of predicting what word comes next.)

![img](https://pic3.zhimg.com/80/v2-0aac9354c677b27fd7602711041464aa_1440w.jpg)

即：

![[公式]](https://www.zhihu.com/equation?tex=P%28x%5E%7Bt%2B1%7D%7Cx%5Et%2Cx%5E%7Bt-1%7D...x%5E1%29)

**1.1 统计学方法：n-gram language model**

简化：一个词出现的概率只和它前面的n-1个词有关系，这就是"n-gram"的含义。因此有:

![img](https://pic3.zhimg.com/80/v2-b8891240727e5a7878dbe7afff983612_1440w.jpg)



n-gram model 是不使用深度学习的方法，直接利用**条件概率**来预测下一个单词是什么。但这个模型有几个问题：

- 由于丢弃了比较远的单词，它不能够把握全局信息。例如，“as the proctor started the clock” 暗示这应该是一场考试，所以应该是students opened their **exam**. 但如果只考虑4-gram，的确是book出现的概率更大。
- sparsity problem. 有些短语根本没有在语料中出现过，比如"student opened their petri-dishes". 所以，petri-dishes的概率为0. 但是这的确是一个合理的情况。解决这个问题的办法是做拉普拉斯平滑，对每个词都给一个小权重。
- sparsity problem的一个更加糟糕的情况是，如果我们甚至没有见过"student open their",那么分母直接就是0了。对于这种情况，可以回退到二元组，比如"student open".这叫做backoff
- 存储空间也需要很大。

**1.2 neural language model**

想要求"the students opened their"的下一个词出现的概率，首先将这四个词分别embedding，之后过两层全连接，再过一层softmax，得到词汇表中每个词的概率分布。我们只需要取概率最大的那个词语作为下一个词即可。

![img](https://pic1.zhimg.com/80/v2-8ef229889a0a4186c8084e9c1a26ae90_1440w.jpg)



**优点：**

- 解决了sparsity problem, 词汇表中的每一个词语经过softmax都有相应的概率。
- 解决了存储空间的问题，不用存储所有的n-gram，只需存储每个词语对应的word embedding即可。

**缺点：**

- 窗口的大小还是不能无限大，不能涵盖之前的所有信息。更何况，增加了窗口大小，就要相应的增加**权重矩阵W**的大小。
- 每个词语的word embedding只和权重矩阵W对应的列相乘，而这些列是完全分开的。所以这几个不同的块都要学习相同的pattern，造成了浪费。

![img](https://pic4.zhimg.com/80/v2-9436bc04b088d009433e6d290fcd2dfb_1440w.jpg)



### 2. RNN

正因为上面所说的缺点，需要引入RNN。

**2.1 RNN模型介绍**

![img](https://pic1.zhimg.com/80/v2-9a1239d746dbe03fe3c20d9a7babc43c_1440w.jpg)



**RNN的结构：**

- 首先，将输入序列的每个词语都做embedding，之后再和矩阵![[公式]](https://www.zhihu.com/equation?tex=W_e) 做点乘，作为hidden state的输入。
- 中间的hidden state层: 初始hidden state ![[公式]](https://www.zhihu.com/equation?tex=+h%5E%7B%280%29%7D) 是一个随机初始化的值，之后每个hidden state的输出值都由前一个hidden state的输出和当前的输入决定。
- 最后的输出，即词汇表V的概率密度函数是由最后一个hidden state决定的

**RNN的优势：**

- 可以处理任意长的输入序列
- 前面很远的信息也不会丢失(这样我们就可以看到前面的"as the proctor start the clock",从而确定应该是"student opened their exam"而不是"student opened their books").
- 模型的大小不会随着输入序列变长而变大。因为我们只需要 ![[公式]](https://www.zhihu.com/equation?tex=W_e) 和 ![[公式]](https://www.zhihu.com/equation?tex=W_h) 这两个参数
- ![[公式]](https://www.zhihu.com/equation?tex=W_e%2CW_h%2Cb) 对于每一步都是一样的(**共享权重**)，每一步都能学习 ![[公式]](https://www.zhihu.com/equation?tex=W_e%2CW_h%2Cb) ,更加efficient

**RNN的坏处：**

- 慢。因为只能串行不能并行
- 实际上，不太能够利用到很久以前的信息，因为**梯度消失**。



**2.2 RNN模型的训练**

- 首先拿到一个非常大的文本序列 ![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%281%29%7D%2C...x%5E%7B%28T%29%7D) 输入给RNN language model
- 对于每一步 t ，都计算此时的输出概率分布 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%28t%29%7D) 。(i.e. predict probability distribution of every word, given the words so far)
- 对于每一步 t，损失函数![[公式]](https://www.zhihu.com/equation?tex=J%5E%7B%28t%29%7D%28%5Ctheta%29) 就是我们预测的概率分布 ![[公式]](https://www.zhihu.com/equation?tex=%5Chat+y%5E%7B%28t%29%7D) 和真实的下一个词语![[公式]](https://www.zhihu.com/equation?tex=y%5E%7B%28t%29%7D) (one-hot编码)的交叉熵损失)。
- 对每一步求平均得到总体的loss：

![[公式]](https://www.zhihu.com/equation?tex=J%28%5Ctheta%29+%3D+%5Cfrac%7B1%7D%7BT%7D+%5Csum_%7Bt%3D1%7D%5ET+J%5E%7B%28t%29%7D%28%5Ctheta%29)

![img](https://pic2.zhimg.com/80/v2-1c4aaf9989e86f2fe77416e9002c06cd_1440w.jpg)



**2.3 Language Model的重要概念--困惑度(perplexity)**

我们已知一个真实的词语序列![[公式]](https://www.zhihu.com/equation?tex=x%5E%7B%281%29%7D...x%5E%7B%28T%29%7D) ,

![img](https://pic3.zhimg.com/80/v2-dcc239736e5f2dc0b5bbe5d67c2ef5ce_1440w.jpg)

即，困惑度和交叉熵loss的指数相等。



**2.4 基础RNN的应用**

（1）生成句子序列

![img](https://pic3.zhimg.com/80/v2-44e6078e179561de2bf369f2099a6a3e_1440w.jpg)

每一步最可能的输出作为下一个的输入词，这个过程可以一直持续下去，生成任意长的序列。



（2）词性标注

![img](https://pic2.zhimg.com/80/v2-18a358cf5d653ea2af623c9e89e76785_1440w.jpg)

每个隐藏层都会输出



（3）文本分类

其实RNN在这个问题上就是为了将一长串文本找到一个合适的embedding。当使用最后一个隐藏状态作为embedding时：

![img](https://pic1.zhimg.com/80/v2-a7b73f39b85b3579cffb75f5005bacf8_1440w.jpg)



当使用所有隐藏状态输出的平均值作为embedding时：

![img](https://pic3.zhimg.com/80/v2-e4bf0b0687422eccd9f16023f7388b36_1440w.jpg)



参考资料：

[http://web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture06-rnnlm.pdfweb.stanford.edu/class/cs224n/slides/cs224n-2019-lecture06-rnnlm.pdf](https://link.zhihu.com/?target=http%3A//web.stanford.edu/class/cs224n/slides/cs224n-2019-lecture06-rnnlm.pdf)