# Vision Transformer

本文重点介绍ViT原理，同时简单介绍三篇相关论文，这四篇论文的源码见 [https://github.com/google-research/vision_transformer](https://link.zhihu.com/?target=https%3A//github.com/google-research/vision_transformer)

[arXiv:2010.11929](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2010.11929)：An image is worth 16x16 words: Transformers for image recognition at scale（ViT大法，一般人没钱做的工作）

[arXiv:2105.01601](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2105.01601)：MLP-Mixer: An all-MLP Architecture for Vision （用MLPs替代self-attention可以得到和ViT同样好的结果）

[arXiv:2106.01548](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.01548)：When Vision Transformers Outperform ResNets without Pre-training or Strong Data Augmentations （不使用大规模预训练和强数据增强ViT是否依然可以表现优秀）

[arXiv:2106.10270](https://link.zhihu.com/?target=https%3A//arxiv.org/abs/2106.10270)：How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers （通过大量实验，总共训练了超过5w个ViT，教你如何训练自己的ViT模型，以及数据增广和模型正则化什么时候有用）

有关transformer结构和原理，大家可以参考：[Transformer解析](https://zhuanlan.zhihu.com/p/410258597)

## arXiv:2010.11929 (ViT)

![img](https://pic2.zhimg.com/80/v2-692d349ee289e1cea9d7d100b6919771_1440w.jpg)

### **简介**

ViT是2020年Google团队提出的将Transformer应用在**图像分类**的模型，虽然不是第一篇将transformer应用在视觉任务的论文，但是因为其模型“简单”且效果好，可扩展性强（scalable，**模型越大效果越好**），成为了transformer在CV领域应用的里程碑著作，也引爆了后续相关研究

把最重要的说在最前面，ViT原论文中最核心的结论是，当拥有**足够多**的数据进行预训练的时候，ViT的表现就会**超过CNN**，突破transformer缺少inductive bias的限制，可以在下游任务中获得较好的迁移效果。但是， 当训练数据集不够大的时候，ViT的表现通常比同等大小的ResNet要差一些，因为Transformer和CNN相比缺少inductive bias，即一种**先验知识**，提前做好的假设。CNN具有两种inductive bias，一种是**局部性**（locality），即图片上相邻的区域具有相似的特征；一种是平移不变形（translation equivariance）， f(g(x))=g(f(x)) ，其中g代表卷积操作，f代表平移操作。当CNN具有以上两种归纳偏置，就有了很多先验信息，需要相对少的数据就可以学习一个比较好的模型。

### ViT的结构

ViT将输入图片分为多个patch（16x16），再将每个patch**投影**为固定长度的向量送入Transformer，后续encoder的操作和原始Transformer中完全相同。但是因为对图片分类，因此在输入序列中加入一个特殊的token，该token对应的输出即为最后的类别预测（类似NLP中的[CLS]）

![img](https://pic4.zhimg.com/80/v2-5afd38bd10b279f3a572b13cda399233_1440w.jpg)ViT只使用了Transformer的encoder

按照上面的流程图，一个ViT block可以分为以下几个步骤

(1) patch embedding：例如输入图片大小为224x224，将图片分为固定大小的patch，patch大小为16x16，则每张图像会生成224x224/16x16=196个patch，即输入序列长度为**196**，每个patch维度16x16x3=**768**（这里的3是通道数），线性投射层的维度为768xN (这里取N=768)，因此输入通过线性投射层之后的维度依然为196x768。这里还需要加上一个特殊字符cls，因此最终的输出是**197x768**。

(2) positional encoding（standard learnable **1D** position embeddings）：注意位置编码的操作是sum，而不是concat。加入位置编码信息之后，维度依然是**197x768**

(3) LN+multi-head attention：LN输出维度依然是197x768。多头自注意力时，先将输入映射到q，k，v，如果只有一个头，qkv的维度都是197x768，如果有12个头（768/12=64），则qkv的维度是197x64，一共有12组qkv，最后再将12组qkv的输出拼接起来，输出维度是197x768，然后在过一层LN，维度依然是**197x768**

(4) MLP：将维度放大再缩小回去，197x768放大为197x3072，再缩小变为**197x768**

一个block之后维度依然和输入相同，都是197x768，因此可以堆叠多个block。最后会将特殊字符cls对应的输出 $z_L^0$  作为encoder的最终输出 ，代表最终的image presentation（另一种做法是不加cls字符，对所有的tokens的输出做一个平均），如下图公式(4)，后面接一个MLP进行图片分类.

下图中，MSA为multi-head self-attention, LN 为 layer-norm:

![img](https://pic2.zhimg.com/80/v2-ebf697b1994598019a6a59855dc0dbed_1440w.jpg)

其中输入image $x∈R^{H×W×C}$ ，把它按照patch切分之后得到的image小块为：$x_p∈R^{N×(P^2⋅C)}$ ， C 是通道数， P 是patch大小，一共有 N 个patches， $N=HW/P^2$.

**关于image presentation**

是否可以直接使用average pooling得到最终的image presentation，而不加特殊字符cls，通过实验表明，同样可以使用average pooling，原文ViT是为了尽可能是模型结构接近原始的Transformer，所以采用了类似于BERT的做法，加入特殊字符

![img](https://pic1.zhimg.com/80/v2-4a8b39b1d2dd43d1e9b16edbc38b1660_1440w.jpg)学习率的影响较大，注意调参

**关于positional encoding**

1-D 位置编码：例如3x3共9个patch，patch编码为1到9

2-D 位置编码：patch编码为11,12,13,21,22,23,31,32,33，即同时考虑X和Y轴的信息，每个轴的编码维度是D/2

实际实验结果表明，不管使用哪种位置编码方式，模型的精度都很接近，甚至不适用位置编码，模型的性能损失也没有特别大。原因可能是ViT是作用在image patch上的，而不是image pixel，对网络来说这些patch之间的相对位置信息很容易理解，所以使用什么方式的位置编码影像都不大

![img](https://pic1.zhimg.com/80/v2-e152c9ad22f6984912fb0652cf294018_1440w.jpg)

**关于CNN+Transformer**

既然CNN具有inductive bias的特性，Transformer又具有很强全局归纳建模能力，使用CNN+Transformer的混合模型是不是可以得到更好的效果呢？将224x224图片送入CNN得到16x16的特征图，拉成一个向量，长度为196，后续操作和ViT相同

**关于输入图片大小**

通常在一个很大的数据集上预训练ViT，然后在下游任务相对小的数据集上微调，已有研究表明在分辨率更高的图片上微调比在在分辨率更低的图片上预训练效果更好（It is often beneficial to fine-tune at higher resolution than pre-training）

当输入图片分辨率发生变化，输入序列的长度也发生变化，虽然ViT可以处理任意长度的序列，但是预训练好的**位置编码**无法再使用，一种做法是使用插值算法，扩大位置编码表。但是如果序列长度变化过大，插值操作会损失模型性能，这是ViT在微调时的一种局限性

### 实验部分

**数据集**

为了探究模型的可扩展性（to explore model scalability），预训练阶段使用了ImageNet-1K（1.3million）、ImageNet-21K（14million），JFT-18K（303million）三个数据集。同时参考BiT，删除预训练数据集中和下游任务测试集中重复的数据（de-duplicate the pre-training datasets w.r.t. the test sets of the downstream）

下游数据集包括：ImageNet（on the original validation labels），ImageNet （on the cleaned-up ReaL labels ），CIFAR-10/100，Oxford-IIIT Pets，Oxford Flowers-102，VTAB (19 tasks)

**模型及变体**

（1）ViT：参考BERT，共设置了三种模型变体（增加了Huge变体）如下图所示。例如ViT-**L**/16，代表Large变体，输入patch size为16x16。（2）CNN：baseline CNNs选择ResNet，同时用Group Normalization替代Batch Normalization，使用standardized convolutions，以提升模型迁移性能。（3）Hybrid：混合模型就是使用ResNet50输出的特征图，不同stage会得到不同大小的特征图，即生成不同长度序列

![img](https://pic2.zhimg.com/80/v2-54f717f71079becca62a0247660a171d_1440w.jpg)

所有模型的训练均使用Adam（ β1=0.9 , β2=0.999 ），batch_size设为4096，权重衰减（apply a high weight decay of 0.1），同时使用了学习率warmup策略（use a linear learning rate warmup and decay）；微调阶段，使用SGD with momentum，batch_size设为512

**实验结果**

![img](https://pic2.zhimg.com/80/v2-d5e21c0fadf2591220271021f570299d_1440w.jpg)ViT和其它SOTA模型性能对比，展示了准确率accuraces的均值和标准差，所有结果都是取三轮微调均值的结果（averaged over three fine-tunning runs）。有关ImageNet的实验，在更高分辨率图片上微调(512 for ViT-L/16 and 518 for ViT-H/14)，同时使用了Polyak averaging(0.9999)

可以看到在JFT数据集上预训练的ViT模型，迁移到下游任务后，表现要好于基于ResNet的BiT和基于EfficientNet的Noisy Student，且需要更少的预训练时间

![img](https://pic2.zhimg.com/80/v2-ff04d612cd03446278622d368ec1c6c9_1440w.jpg)各类模型在VTAB上的表现，ViT同样性能更好

上面的实验显示，当在很大的数据集上预训练时，ViT性能超越CNN，后面探究不同大小预训练数据集对模型性能的影响（不能只看超大数据集）

![img](https://pic2.zhimg.com/80/v2-65138c61d2f9c57448b4ba23dad4af55_1440w.jpg)transfor to ImageNet

这里当在更小的数据集上预训练时（ImageNet），优化三个超参数以提升模型性能，分别是weight decay, dropout 和 label smoothing。可以看到当在小数据集上预训练时（ImageNet-1k，1.3million），ViT微调后的效果远远比不上ResNet；在中等数据集上预训练时（ImageNet-21K，14million），两者效果相当；当在很大的数据集上（JFT-300M, 300million）预训练时，ViT的效果要更好。所以当我们只有较小的数据集时，更适合使用ResNet（并不是所有数据集都适合硬套transformer）

![img](https://pic1.zhimg.com/80/v2-719f695ec8fb9c33e19fd8515c1ff230_1440w.jpg)Linear few-shot evaluation on ImageNet versus pre-training size

如上图，在同一个数据集（JFT），分别抽取不同数量的数据（10M，30M，100M，300M），避免不同数据集之间的gap，同时不适用额外的regularization，超参数保证相同。linear evaluation是指直接把预训练模型当做特征提取器，不fine-tune，拿提取到的特征直接做logistic regression。few-shot是指在evaluation的时候，每一类只sample五张图片。

可以看到当数据集很小时，CNN预训练模型表现更好，证明了CNN归纳偏置的有效性，但是当数据集足够大时，归纳偏置和Transformer比较就失去了优势，甚至没有归纳偏置，直接从数据learn patterns会更有效。同时细心观察会发现即使预训练的数据集很大，最后ViT的性能提升也不是很明显，因此如何使用ViT来做这种小样本学习任务，是一个有待继续研究的方向

![img](https://pic4.zhimg.com/80/v2-53da3593bafc05bad3b72099583d909b_1440w.jpg)Performance versus cost for different architectures: Vision Transformers, ResNets, andhybrids.

上图实验证明了ViT的预训练比ResNet要更便宜，即在相同的预训练计算复杂度下，ViT的效果要比ResNet更好。可以看到，当模型较小时，混合模型的表现要更好，但是随着模型的增大，ViT的表现超过了混合模型（为什么混合模型这个时候不如ViT，直觉上混合模型吸收了双方的优点，应该表现更好）。

### 模型可视化

![img](https://pic2.zhimg.com/80/v2-fb19c6d629b419b02f26b4db31598a51_1440w.jpg)ViT block第一层（linear projection）的前28个主成分

![img](https://pic2.zhimg.com/80/v2-99f02198921e7aed8162cd7af8a29805_1440w.jpg)位置编码得相似性分析(cos)，位置越接接近，patches之间的相似度越高；相同行/列的patches有相似的embeddings；

为了理解self-attention是如何聚合信息的（To understand how ViT uses self-attention to integrate information across the image），基于attention weight计算不同layer不同head的average attention distance

![img](https://pic4.zhimg.com/80/v2-8539fe277eae097183add2d6d2f559e3_1440w.jpg)每一个layer的每一个head的average attention distance，类似于CNN感受野的概念，可以发现一些head在第一个layer就attent到了几乎整张图片的范围

average attention distance，是基于attention weight计算，具体做法是用attention weight乘以query pixel和所有其它pixels的距离，再求平均。原文中是这么表述的——Attention distance was computed for 128 example images by averaging the distance between the query pixel and all other pixels, weighted by the attention weight. Each dot shows the mean attention distance across images for one of 16 heads at one layer. Image width is 224 pixels.

![img](https://pic4.zhimg.com/80/v2-dabb6afe03b02498e0a8081d00c0b437_1440w.jpg)Representative examples of attention from the output token to the input space.

## arXiv:2105.01601

![img](https://pic3.zhimg.com/80/v2-30735a1641e51d73675e69a99dfe0476_1440w.jpg)

### 简介

ViT作者团队出品，在CNN和Transformer大火的背景下，舍弃了卷积和注意力机制，提出了MLP-Mixer，一个完全基于MLPs的结构，其MLPs有两种类型，分别是**channel-mixing MLPs**和**token-mixing MLPs**，前者独立作用于image patches（融合通道信息），后者跨image patches作用（融合空间信息）。实验结果表明该结构和SOTA方法同样出色，证明了convolution和attention不是必要操作，如果将其替换为简单的MLP，模型依然可以完美work

> MLP-Mixer contains two types of layers: one with MLPs applied independently to image patches (i.e. “mixing” the per-location features), and one with MLPs applied across patches (i.e. “mixing” spatial information) / MLP-Mixer's architecture is based entirely on multi-layer perceptrons (MLPs) that are repeatedly applied across either spatial locations or feature channels / Mixer makes use of two types of MLP layers: channel-mixing MLPs and token-mixing MLPs

### Mixer结构

![img](https://pic3.zhimg.com/80/v2-ed96c7a5add85b9151c7f10fbda4943a_1440w.jpg)类似于Transformer，每一个layer输入和输出的维度相同，可以堆叠多个layer

类似于ViT，首先进行patch embedding操作，一个Mixer Layer中包含了channel-mixing MLPs和token-mixing MLPs，但是Mixer不适用positional encoding，因为token-mixing MLPs对输入tokens的顺序非常敏感

![img](https://pic2.zhimg.com/80/v2-58df6c737fdea377071868f106b7c1c1_1440w.jpg)

token-mixing MLPs：允许信息在空间维度交互，独立作用于每一个channel，作用于列，融合不同token的特征

channel-mixing MLPs：允许信息在通道交互，独立作用于每一个token，作用于行，融合不同channel的特征

输入image的分辨率为 H×W ，patch的分辨率为 P×P，则patch的数量 S=HW/P2 ，所有的patch拉直后线性投影到维度 C ，则得到Mixer Layer的输入 X∈RS×C 。token-mixing MLPs作用于 X 的列，特征维度不发生变化（ RS→RS ），channel-mixing MLPs作用于 X 的行，特征维度同样不发生变化（ RC→RC ）。每一个MLP包含两个全连接层和一个非线性激活（GELU），一个Mixer layers的公式如下，计算复杂度和输入patches的数量成线性关系（ViT是平方关系）

![img](https://pic3.zhimg.com/80/v2-a0abfc39a823b43e061ee5acd14da0e2_1440w.jpg)

需要注意token-mixing MLPs共享参数，channel-mixing MLPs同样共享参数，因此避免了当输入特征维度增加（ C 变大）或者输入序列长度增加（ S 变大）时，模型参数量急剧增加的情况，极大减少了内存消耗

模型最后接global average pooling+a linear classifier

### 实验结果

当在大规模数据集上预训练（100million images），Mixer可以接近CNNs和Transformers的SOTA表现，在ImageNet上达到87.94%的top-1 accuracy；当在更小规模数据集上预训练时（10million），结合一些regularization techniques，Mixer可以接近ViT的性能，但是稍逊于CNN

## arXiv:2106.01548

![img](https://pic2.zhimg.com/80/v2-a8110308e4ae9bc117f689db11af50dd_1440w.jpg)

### 简介

ViTs和MLPs的相关研究目前大多都非常依赖海量数据，在大规模数据集上的预训练以及强数据增广都是基本操作，但是在模型实际优化过程中依然存在诸多困难，比如对初始化和学习率非常敏感。因此作者从损失几何（loss geometry/loss landscape geometry）的角度探究ViTs和MLP-Mixers（从损失几何的角度说白了就是用SAM方法对loss做平滑），**旨在让模型摆脱对大规模预训练以及强数据增广的依赖**，提升模型在训练阶段对数据的利用效率，以及推理阶段的泛华能力（intending to improve the models' data efficiency at training and generalization at inference）

通过可视化和 Hessian 发现了收敛模型极其尖锐的局部最小值（sharp local minima of the converged models），因此使用最近提出的锐度感知优化器（sharpness-aware optimizer/sharpness-aware minimizer，**SAM**）提高平滑度（promote smoothness），得到更加平滑的损失函数（much flatter loss landspace/ smoothed loss landspace），大大提升了ViTs和MLPs在多个任务上的准确度和鲁棒性，包括监督、对抗、对比、迁移学习等（使用简单的 Inception 式预处理，ViT-B/16 和 Mixer-B/16 在 ImageNet 上的top-1准确率分别提升了5.3% 和11.0%）

改进的平滑度归因于前几层中较稀疏的激活神经元（the improved smoothness attributes to sparser active/activated neurons in the first few layers）。在没有大规模预训练或强数据增强的情况下，在 ImageNet 上从头开始训练时，所得 ViT 的性能优于类似大小和吞吐量（throughput）的 ResNet。还拥有更敏锐的注意力图（more perceptive attention maps）

## arXiv:2106.10270

![img](https://pic4.zhimg.com/80/v2-d9282ace32876d718911c1f21a5911ff_1440w.jpg)

和ViT和Mixer一样，还是熟悉的Google团队，还包括timm作者Ross Wightman

### 简介

ViT在很多视觉任务上都展现了相当优秀的性能，但是和CNN相比，缺少归纳偏置让ViT应用于小数据集时非常依赖模型正则化（model regularization）和数据增广（data augmentation）（把模型正则化和数据增广合起来简称AugReg）

作者使用系统性的实证研究方法（systematic empirical study），探究训练数据量、AugReg、模型大小、计算成本的interpaly（相互影响/影响），说白了就是做了大量实验，用实验结果说明问题，总共训练了超过50000个ViT模型，结果发布在了 [https://github.com/rwightman/pytorch-image-models](https://link.zhihu.com/?target=https%3A//github.com/rwightman/pytorch-image-models) 和 [https://github.com/google-research/vision_ transformer](https://link.zhihu.com/?target=https%3A//github.com/google-research/vision_)

实验表明，当增加计算成本（Improved compute/Increased compute budget），即让模型训练更长时间已达到一定的性能，同时使用AugReg会带来意想不到的效果：在ImageNet-21k（14million）上训练的ViT模型，和在JFT-300M上训练的ViT模型相比拥有更好的性能。同时大量实验也揭示了各类techniques的对模型性能的影响，以及什么时候AugReg对模型性能有益/什么时候无益

作者还对ViT迁移学习进行了深入分析。结论是**即使下游数据似乎与预训练数据只有微弱的关联，迁移学习仍然是最佳选择。**作者分析还表明，对于迁移学习来说，训练数据更多的模型和数据增强更多的模型相比较（among similarly performing pre-trained models），前者可能是更好的选择，下游任务性能表现更好。该研究的意义在哪，当我们的计算成本有限时，可以通过本研究的结论选择一种方式，更高效的优化ViT模型