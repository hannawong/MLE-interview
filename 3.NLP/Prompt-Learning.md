# Prompt-Learning

在很长的一段时间内，NLP的任务采用的都是 **Pretrain + Fine-tuning** 的方案，但是这种方案，需要对于每个任务都重新 fine-tune 一个新的模型，不能共用，十分不高效。是否存在一种方式，可以将预训练语言模型作为电源，不同的任务当作电器，仅需要根据不同的任务，选择不同的插座。对于模型来说，即插入不同的任务特定的参数，就可以使得模型适配该下游任务。Prompt Learning 就是这个适配器，它能高效得进行预训练语言模型的使用。下图中，

- 左边是传统的 Model Tuning 的范式：对于不同的任务，都需要将整个预训练语言模型进行精调，每个任务都有自己的一整套参数。
- 右边是Prompt Tuning，对于不同的任务，仅需要插入不同的prompt 参数，**每个任务都单独训练Prompt 参数**，不训练预训练语言模型，这样子可以大大缩短训练时间，也极大的提升了模型的使用率。

![img](https://pic2.zhimg.com/80/v2-ffa9e652a07961216d1ed260dfdea95d_1440w.webp)



Prompt的定义如下：

> Prompt is the technique of **making better use of the knowledge** from the pre-trained model by **adding additional texts to the input**.
>
> Prompt 是一种**为了更好的使用预训练语言模型的知识**，采用在输入段**添加额外的文本**的技术。

- 目的：更好挖掘预训练语言模型的能力
- 手段：在输入端添加文本，即重新定义任务（task reformulation）

## Prompt 的工作流

Prompt 的工作流包含以下4部分：

1. Prompt 模版（Template）的构造
2. Prompt 答案空间映射（Verbalizer）的构造
3. 文本代入template，并且使用预训练语言模型进行预测
4. 将预测的结果映射回label。

具体的步骤如下图，我们将一步步进行拆解分析。

![img](https://pic4.zhimg.com/80/v2-65b25d4895d4d7b81d747282cdb4c7f3_1440w.webp)

### Step 1: prompt construction【Template】

首先我们需要构建一个模版Template，模版的作用是将输入和输出进行重新构造，变成一个新的带有mask slots的文本，具体如下：

- 定义一个模版，包含了2处代填入的slots：[x] 和 [z]
- 将[x] 用输入文本代入

例如：

- 输入：x = 我喜欢这个电影。
- 模版：[x]总而言之，它是一个[z]电影。
- 代入（prompting）：我喜欢这个电影。总而言之，它是一个[z]电影。

![img](https://pic2.zhimg.com/80/v2-e6c4edefdb2498229dfa37f9fc883f15_1440w.webp)

### Step 2: answer construction【Verbalizer】

对于我们构造的prompt，我们需要知道我们的预测词和我们的label 之间的关系，并且我们也不可能运行z是任意词，这边我们就需要一个映射函数（mapping function）将输出的词与label进行映射。例如我们的这个例子，输出的label 有两个，一个是 ，一个是 ，我们可以限定，如果预测词是`fantastic` 则对应 ，如果是 `boring` 则对应 .

![img](https://pic3.zhimg.com/80/v2-6c3ab4435a08d559c69d2b46b18a5d1e_1440w.webp)

![img](https://pic4.zhimg.com/80/v2-4708199326266b548a6b3e0361b1bb47_1440w.webp)

### Step 3: answer prediction【Prediction】

到了这边我们就只需要选择[合适的预训练语言模型](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/transformers/model_summary)，然后进行mask slots [z] 的预测。例如下图，得到了结果 `fantastic`, 我们需要将其代入[z] 中。

![img](https://pic2.zhimg.com/80/v2-c12486224648f205c3c8199101a06b75_1440w.webp)

### Step 4: answer-label mapping【Mapping】

第四步骤，对于得到的 `answer`，我们需要使用 `Verbalizer` 将其映射回原本的label。

例如：fantastic 映射回 label：

![img](https://pic2.zhimg.com/80/v2-b9a60cb83f1c6772490053801d13885d_1440w.webp)

### 总结

![img](https://pic1.zhimg.com/80/v2-45666504e1714ef274be6ed35a86d388_1440w.webp)

## Prompt-based 方法的工程选择问题

在知乎中有个提问：

> 现代的deep learning 就是为了规避 feature engineering，可是prompt 这边选择了template和answer不还是 feature engineering吗？

从这个问题中我们可以发现，确实如果使用BERT的 fine-tuning 范式（下图左），我们是不需要使用任何的人工特征构造，而使用prompt-based的方法的话，需要人工参与的部分包含了以下部分：

- template 构造
- answer 构造
- 预训练模型选择
- prompt 的组合问题选择
- 以及训练策略的选择等

![img](https://pic3.zhimg.com/80/v2-011fccce5f1d7367c243def5d3da4e32_1440w.webp)

下面我们会先进行每个需要人工engineering 的部分进行详细讲解，然后再分析为什么我们还需要prompt 这种范式。

### Prompt Template Engineering（Prompt模版工程）

如何构造合适的Prompt 模版？对于同一个任务，不同的人可能构造不同的Template。

![img](https://pic3.zhimg.com/80/v2-6b1eef9478acd01687934bcaa07a3d0a_1440w.webp)

且每个模版都具有合理性。Tempalte的选择，对于Prompt任务起到了很重大的作用，就算一个word的区别，也坑导致10几个点的效果差别，论文[GPT Understands, Too](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2103.10385) 给出了如下的结果：

![img](https://pic1.zhimg.com/80/v2-b98e6f18abfad252f96b04a549cc9898_1440w.webp)

对于不同的template，可以从以下两种角度进行区分：

1. 根据slot 的形状/位置区分

- 1.1 完形填空（Cloze）的模式，即未知的slot在template的中间等不定的位置
- 1.2 前缀模式（Prefix），未知的slot在template的开头

1. 根据是否是由人指定的来区分

- 2.1 人工指定 template

- 2.2 自动搜索 template

- - 2.3 Discrete 离散Template，即搜索的空间是离散的，为预训练语言模型的字典里的字符。
  - 2.4 Continuous 连续Template，即搜索的空间是连续的，因为所有新增的这些prompt的参数主要是为了让机器更好地服务于任务，所以其参数的取值空间不需要限定在特定的取值范围内，可以是连续的空间。



具体的思维导图如下：

![img](https://pic3.zhimg.com/80/v2-c96d89a06ca2ec58e31a7cd2b4f30e7e_1440w.webp)

### Answer Engineering（答案工程）

在给定一个任务或者Prompt，如何对 label 空间 和 answer 空间进行映射？

![img](https://pic2.zhimg.com/80/v2-3df9747e3a96385804fce424e5c7b619_1440w.webp)

在上图，我们的label 空间 Y 是: `Positive, Negative`, 答案空间 Z 可以是表示positive或者negative 的词，例如 `Interesting/Fantastic/Happy/Boring/1-Star/Bad`，具体的答案空间 Z的选择范围可以由我们指定。我们可以指定一个 y 对应1-N个字符/词。

![img](https://pic4.zhimg.com/80/v2-acf52745920aae345e6c5ac9ee20e92b_1440w.webp)

具体的答案空间的选择可以有以下三个分类标注：

1. 根据形状

- 1.1 Token 类型
- 1.2 Span 类型
- 1.3 Sentence 类型

1. 是否有界

- 2.1 有界
- 2.2 无界

1. 是否人工选择

- 3.1 人工选择

- 3.2 自动搜素

- - 3.2.1 离散空间
  - 3.2.2 连续空间



具体的思维导图如下：

![img](https://pic4.zhimg.com/80/v2-9e58677317b0576537d58fa669c6fc5b_1440w.webp)

**Pre-trained Model Choice（预训练模型选择）**

在定义完模版以及答案空间后，我们需要选择合适的预训练语言模型对 prompt 进行预测，如何选择一个合适的预训练语言模型也是需要人工经验判别的。

具体的预训练语言模型分类可以分为如下5类，具体参考：[Huggingface Summary of the models](https://link.zhihu.com/?target=https%3A//huggingface.co/docs/transformers/model_summary)

- autoregressive-models: 自回归模型，主要代表有 GPT，主要用于生成任务
- autoencoding-models: 自编码模型，主要代表有 BERT，主要用于NLU任务
- seq-to-seq-models：序列到序列任务，包含了an encoder 和 a decoder，主要代表有 BART，主要用于基于条件的生成任务，例如翻译，summary等
- multimodal-models：多模态模型
- retrieval-based-models：基于召回的模型，主要用于开放域问答

基于此，例如下图想要做summary 任务，我们可以选择更合适的 BART 模型。

![img](https://pic3.zhimg.com/80/v2-f7cf71aaa125547394fc53faadc0e7e6_1440w.webp)

其他分类标准也可参考：

![img](https://pic3.zhimg.com/80/v2-654bc36f0f684b17b97f1966fe5904aa_1440w.webp)



![img](https://pic2.zhimg.com/80/v2-f8e5c095bf30cd97176098cd725974e1_1440w.webp)

### Expanding the Paradigm（范式拓展）

如何对已有的 Prompt 进行任务增强以及拓展，具体可以从以下几个方面进行探讨：

- Prompt Ensemble：Prompt 集成，采用多种方式询问同一个问题

![img](https://pic2.zhimg.com/80/v2-0143f03d078e4c78231b987dd041752d_1440w.webp)

- Prompt Augmentation：Prompt 增强，采用类似的 prompt 提示进行增强

![img](https://pic1.zhimg.com/80/v2-04b7abb8bac6d642808f95c3eaeef088_1440w.webp)

----

可能看到前面对continuous prompt不是很理解，下面用一篇论文来说明一下：

###### **WARP: Word-level Adversarial ReProgramming**

本文最大的贡献在于，不同于Discrete Prompt需要手工寻找或者学习**离散的token作为prompt**，本文直接优化embedding作为prompt，这给了我们的模型**更多的自由度**，并最终在下游任务中有更好的表现。

文章的思路很简单，我们需要优化的参数就是两组embedding $Θ={Θ_P,Θ_V}$，P代表prompt，V是对每一类的分类参数，类似上文所说的Verbalizer。



![img](https://pic4.zhimg.com/80/v2-7898abbdbc2b20dece8c98129c08d043_1440w.webp)

如上图所示，具体来说，我们把prompt tokens $P_1,...,P_N$ 插入到输入序列中，再经过encoder和一个MLM head，然后通过$Θ_V$，那么我们分类的概率可以通过如下公式计算:

![img](https://pic4.zhimg.com/80/v2-3df621c58bfb3857c8fcfc0338f484d3_1440w.webp)

公式中的$T_{Θ^P}(x)$是插入了prompt $Θ^P$的序列，C是所有类别，f(x)是预训练语言模型的的输出。

训练过程也很简单，就是在下游任务的数据集中通过梯度优化寻找使得cross-entropy loss最小的参数。

实验过程中所有的prompt tokens都被初始化为 [MASK]的embedding。

在最常用的benchmark GLUE上，WARP取得了非常不错的效果，并且参数量少了好多个数量级。因为每个不同的task只需要增加很少量的参数就行，而不需要把所有参数都finetune一遍。

在可解释性方面，作者通过寻找距离embedding最近的token来进行离散化，我们摘取两个任务对应的prompt来看一看，可以看到基本上没有啥真实的语义信息。相比之下ΘV就比较好解释了，对MNLI的“contradiction”类的embedding接近于token “unless”。SST-2任务中“negative”和“positive”类的embedding分别接近于“defective”和“important”。

![img](https://pic1.zhimg.com/80/v2-6b038ae41062a7241fc62e4f1ff97aa0_1440w.webp)