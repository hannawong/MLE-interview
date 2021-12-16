# PNN[2016]

如果说FNN是在DNN时代的一个初步尝试的话，那么PNN就是真正意义上把FM融入到了DNN之中。

相比Embedding+MLP的传统结构，PNN在embedding层后设计了**Product Layer**，以显示捕捉基于Field的二阶特征（其实，这个product layer就是一个FM）。这是因为NN层之间的"**add** operation"不足以捕获不同的Field特征间的相关性, 而 "**product**" 相比 "add" 能更好得捕捉特征间的dependence，因此作者希望在NN中显示地引入"product"操作，从而更好地学习不同Field特征间的相关性。

> The 'add' operations of the perceptron layer might not be useful to explore the interactions of categorical data in multiple fields. 

![img](https://pic4.zhimg.com/v2-e369b34fe90f9b5cdba0ed40bc56ec4f_b.png)

product layer分成了两个部分：

- 一个部分称为z，是linear部分(一阶项) , 实际就是原始的embedding concat到一起。为了保持格式上的统一性，不妨把它当成原始embedding和1的内积。
- 另一部分称为p，就是真正的product产生的部分。product部分其实就是FM，计算所有特征的两两product，得到$n(n-1)/2$个数。计算product的时候可以选择是使用“内积”还是“外积”，也就是所谓的IPNN和OPNN。内积就和图上画的一样，直接对两个 ![[公式]](https://www.zhihu.com/equation?tex=f) 做**内积**得到 ![[公式]](https://www.zhihu.com/equation?tex=p) 就好了。但是外积怎么做呢？按照外积的定义： ![[公式]](https://www.zhihu.com/equation?tex=g%28f_i%2Cf_j%29%3Df_if_j%5ET) 这样得到的结果是一个$emb\_size \times emb\_size$的矩阵，要想把它映射成一个数，就还需要一个3维的矩阵来做转换。如果每次都先算完两两的外积然后再转换成一个数，这样的复杂度是有点爆炸的。这篇文章在这里做了一个近似：先计算所有 ![[公式]](https://www.zhihu.com/equation?tex=f) 的和，再做外积，即 ![[公式]](https://www.zhihu.com/equation?tex=p%3Df_%5Csum+f_%5Csum%5ET) 。这样先计算求和的复杂度不高，最后把它转化为$L_1$层即可。



PNN这个模型结构相对来说是比较舒服的，每一步操作都挺自然，也不需要两阶段。