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
  ![img](https://picx.zhimg.com/80/v2-afb3c32fec8b9bece5da94727fe7e972_1440w.png)

- Response Generation：根据context和response已经生成的部分来生成剩下的部分
  ![img](https://pica.zhimg.com/80/v2-37450339bdd90d2e3b24cc141a81f39d_1440w.png)

- Dialog Action Prediction：由于可以有多个Action，所以是多分类任务。仅对有标注数据有用
  ![img](https://picx.zhimg.com/80/v2-37ccceba419d1f4eff51c3b87e3396e2_1440w.png)



- **一致性正则化(Consistent Regularization)**：将一段对话context输入编码器，由于 dropout 扰动会得到两个不同的分布，采用 **KL loss 来最小化**这两个分布之间的距离，如下图所示。


  ![img](https://picx.zhimg.com/80/v2-b5ef14b1480e49a92808ec34f9f16a2c_1440w.png)

  ![img](https://picx.zhimg.com/80/v2-4100c791f2613bf67cb2ec2bba9b5daf_1440w.png)

### 半监督预训练范式

- **有标注数据**的损失函数，就是上面说的四种Loss之和：

![img](https://picx.zhimg.com/80/v2-70aca685747b619cb1a53c059eba301e_1440w.png)

- **无标注数据**的损失函数：

![img](https://pica.zhimg.com/80/v2-6d2a01b1eb9a5733628f278ec298e55c_1440w.png)

这个g表示一个selection gate，因为有些无标注数据太noisy，不适合用来计算loss。

- 总的损失函数（有标和无标数据混合训练）

![img](https://picx.zhimg.com/80/v2-62ee7300df68104bdb4557f57cabd4eb_1440w.png)

### Finetuning and Inference

对于有标注信息的对话数据，损失函数为：

![img](https://pic2.zhimg.com/80/v2-7539c9f0f9fe48d1b7eca8c00d4f879d_1440w.webp)

GALAXY 的实验结果表明大规模任务型对话数据进行预训练可以带了很好的收益，且有监督对话动作预测任务对学习对话策略是有效的。