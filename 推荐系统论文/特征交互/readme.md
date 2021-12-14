

## TFNet: Multi-Semantic Feature Interaction for CTR prediction

以前的一些CTR预测模型例如FM、Wide&Deep、DeepFM等都大多使用特征对之间的向量乘积 (vector-product of each pair of features），这样是不好的，因为不同的两两特征可能在不同的语义空间，如果直接做向量乘积，就相当于默认它们在同一个语义空间了。例如：<user, ad>特征对和<banner-position，ad>特征对明显在不同的语义空间下：前者学习user对广告的喜好， 而后者代表广告主在这个广告上所支付的费用的影响。

> The general architecture of these methods is simply concatenating the first-order features and interactive second-order features, inputting them into the Multilayer Perceptron (MLP) to learn higher-order feature interactions and finally predicting the click-through rate.

文章的意思就是，把每个两两特征对都在m个空间上做点乘，这样每个特征对就得到m维向量，总共$n(n-1)/2$个特征对。其中，加入了两个attention机制，一个是对$n(n-1)/2$个特征对的注意力权重；另一个是m个特征空间的注意力权重。

![img](https://pic2.zhimg.com/80/v2-92a597a443f8501b1bb3c68712ce90bd_1440w.jpg)

模型使用类似Wide&Deep 这样的双路方法，一路是普通的embedding+concat+MLP,另外一路就是做完特征交互之后的结果+DNN。



