# GALAXY：基于半监督学习的任务型对话预训练模型

目前的对话预训练方法主要侧重于增强Dialog Understanding和Dialog Generation，而忽视了Dialog Policy的利用。于是，作者提出了一种新的预训练对话模型GALAXY，其通过**半监督学习**，从有限的有标记对话和大规模无标记对话语料库中显式学习Dialog Policy。

## 背景介绍

目前Task-Oriented Dialog预训练领域的研究主要集中在：如何解决任务型对话数据量不足的问题；以及怎样设计更适用于对话系统的**预训练任务**来捕捉对话中的任务相关的信息。现有的任务型对话预训练的相关研究并没有在预训练阶段丰富有关Dialog Policy的知识，作者假设在预训练阶段直接进行对话策略的学习（DA prediction）可以使模型学习到更好地表示，并进一步提高端到端地性能。因此，本文主要关注于怎样在**预训阶段**来对对话策略进行更好地建模。

一个简单的方式是将有监督Dialog Action分类损失和预训练的无监督 MLM 损失一起进行多任务训练，但大量数据都是无标记的，该怎么办呢？

## 模型框架

![img](https://pic1.zhimg.com/80/v2-611e188b1198926df99068b70f9ac4d3_1440w.png)

- UniLM 为 backbone，它包含一个用于理解的双向编码器和一个用于生成的单向解码器，编码器和解码器是权重共享的
- 输入表示包括四个部分：**位置编码、轮次编码、角色编码、token 编码**

### 预训练任务

- Response Selection：构造正负样例进行二分类，负例是进行in-batch采样得到的：

![img](https://pic2.zhimg.com/80/v2-5a12d1bb33a46197eddfe39ff9921ba9_1440w.webp)

- Response Generation：根据context和response已经生成的部分来生成剩下的部分

![img](https://pic3.zhimg.com/80/v2-d00e529dddc6ab88442485f06ffa338a_1440w.webp)

- Dialog Action Prediction：由于可以有多个Action，所以是多分类任务。仅对有标注数据有用

![img](https://pic2.zhimg.com/80/v2-a1f02a85cbc0415abdf8e5b80ca794cd_1440w.webp)

- **一致性正则化(Consistent Regularization)**：将一段对话context输入编码器，由于 dropout 扰动会得到两个不同的分布，采用 **KL loss 来最小化**这两个分布之间的距离，如下图所示。

![img](https://pic4.zhimg.com/80/v2-3cf7b9a86e3f242ea3329728a6ac446b_1440w.webp)

![img](https://pic2.zhimg.com/80/v2-a52a06a5100c55b3bfdaf6cf3355fb01_1440w.webp)

### 半监督预训练范式

- **有标注数据**的损失函数，就是上面说的四种Loss之和：

![img](https://pic1.zhimg.com/80/v2-da8b9f075dc056eecdfca835d0ecc0dc_1440w.webp)

- **无标注数据**的损失函数：

![img](https://pic4.zhimg.com/80/v2-78334cab5dfdfc8fe4965b5b41fe661b_1440w.webp)

这个g表示一个selection gate，因为有些无标注数据太noisy，不适合用来计算loss。

- 总的损失函数（有标和无标数据混合训练）

![img](https://pic1.zhimg.com/80/v2-f5f70249640f3aff731bb7cbd5dcc1f8_1440w.webp)

### Finetuning and Inference

对于有标注信息的对话数据，损失函数为：

![img](https://pic2.zhimg.com/80/v2-7539c9f0f9fe48d1b7eca8c00d4f879d_1440w.webp)

微调的损失函数

GALAXY 的实验结果表明大规模任务型对话数据进行预训练可以带了很好的收益，且有监督对话动作预测任务对学习对话策略是有效的。