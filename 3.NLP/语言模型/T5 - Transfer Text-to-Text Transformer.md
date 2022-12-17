# T5 - **Transfer Text-to-Text Transformer** 

Transfer 来自 Transfer Learning，**预训练模型**大体在这范畴，Transformer 也不必多说，那么 Text-to-Text 是什么呢。那就是作者在这提出的一个统一框架，靠着大力出奇迹，**将所有 NLP 任务都转化成 Text-to-Text （文本到文本）任务**。

![img](https://pic2.zhimg.com/80/v2-82deada7be746017fe4d3808b6657af9_1440w.webp)

比如英德翻译，只需将训练数据集的输入部分前加上“translate English to German（” 就行。假设需要翻译"That is good"，那么先转换成 "translate English to German：That is good." 输入模型，之后就可以直接输出德语翻译 “Das ist gut.”

再比如情感分类任务，输入"sentiment：This movie is terrible!"，前面直接加上 “sentiment：”，然后就能输出结果“negative”。

最神奇的是，对于需要输出连续值的 STS-B（文本语义相似度任务），居然也是直接输出文本，而不是加个连续值输出头。以每 0.2 为间隔，从 1 到 5 分之间分成 21 个值作为输出分类任务。比如上图中，输出 3.8 其实不是数值，而是一串文本，之所以能进行这样的操作，应该完全赖于 T5 模型**强大的容量**。

通过这样的方式就能将 NLP 任务都转换成 Text-to-Text 形式，也就可以**用同样的模型，同样的损失函数，同样的训练过程，同样的解码过程来完成所有 NLP 任务**。

### **Data：C4 **

作者从 Common Crawl（一个公开的网页存档数据集，每个月大概抓取 20TB 文本数据） 里清出了 750 GB 的训练数据，然后取名为 ” Colossal Clean Crawled Corpus ）“.

### **Architecture：The Best One**

首先作者们先对预训练模型中的多种模型架构（Transformer）进行了比对，最主要的模型架构可以分成下面三种。

![img](https://pic2.zhimg.com/80/v2-b1a8d9af6110e6d1b6a7615fc300a229_1440w.webp)

第一种，**Encoder-Decoder 型**，即 Seq2Seq 常用模型。

第二种，自回归language model

第三种，**Prefix LM（Language Model） 型**，可看作是上面 Encoder 和 Decoder 的融合体，一部分如 Encoder 一样能看到全体信息，一部分如 Decoder 一样只能看到过去信息。最近开源的 UniLM 便是此结构。

上面这些模型架构都是 Transformer 构成，之所以有这些变换，主要是**对其中注意力机制的 Mask 操作**。

![img](https://pic1.zhimg.com/80/v2-b06b504f19febe0f1582f8b162cfbb9c_1440w.webp)

通过实验作者们发现，在提出的这个 Text-to-Text 架构中，Encoder-Decoder 模型效果最好。于是乎，就把它定为 T5 模型，因此**所谓的 T5 模型其实就是个 Transformer 的 Encoder-Decoder 模型**。

### **预训练目标**

之后是对预训练目标的大范围探索，具体做了哪些实验，下面这张图就能一目了然。

![img](https://pic3.zhimg.com/80/v2-247e53593f78282caf557d84c1d2c1fa_1440w.webp)

第一个方面，**高层次方法（自监督的预训练方法）对比**，总共三种方式。

1. **语言模型式**，就是 GPT-2 那种方式，从左到右预测；
2. **BERT-style 式**，就是像 BERT 一样将一部分给破坏掉，然后还原出来；
3. Deshuffling （顺序还原）式，就是将文本打乱，然后还原出来。

![img](https://pic3.zhimg.com/80/v2-4188a5cef8a88085705b0e7cc9991ff2_1440w.webp)

其中发现 Bert-style 最好，进入下一轮。

第二方面，对文本一部分进行**破坏时的策略**，也分三种方法。

1. **Mask 法**，如现在大多模型的做法，将被破坏 token 换成特殊符如 [M]；
2. **replace span（小段替换）法**，可以把它当作是把上面 Mask 法中相邻 [M] 都合成了一个特殊符，每一小段替换一个特殊符，提高计算效率；
3. **Drop 法**，没有替换操作，直接随机丢弃一些字符。

![img](https://pic4.zhimg.com/80/v2-f5b13a845911a7f57dec821cfe57713f_1440w.webp)

此轮获胜的是 **Replace Span 法**，类似做法如 SpanBERT 也证明了有效性。

当当当，进入下一轮。

第三方面，到底该**对文本百分之多少进行破坏**呢，挑了 4 个值，10%，15%，25%，50%，最后发现 BERT 的 **15%** 就很 ok了。这时不得不感叹 BERT 作者 Devlin 这个技术老司机直觉的厉害。

接着进入更细节，第四方面，因为 Replace Span 需要决定**对大概多长的小段进行破坏**，于是对不同长度进行探索，2，3，5，10 这四个值，最后发现 **3** 结果最好。

----

其他区别：T5用的是相对位置编码。



