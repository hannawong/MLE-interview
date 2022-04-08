## 梯度优化裁剪(PCGrad)

###### Project Conflicting Gradients (PCGrad) [斯坦福，NIPS2020]

出自论文 Gradient Surgery for Multi-Task Learning，这个名字十分形象："Gradient Surgery".

在多任务训练期间，如果能知道具体的梯度就可以利用梯度来动态更新 ![[公式]](https://www.zhihu.com/equation?tex=w) 。 如果两个任务的梯度存在**冲突**（即余弦相似度为负），将任务A 的梯度投影到任务B 梯度的法线上。**即是消除任务梯度的冲突部分**，减少任务间冲突。同时，**对模长进行了归一化操作**，防止梯度被某个特别大的主导了。

![img](https://pic3.zhimg.com/v2-ffaccdc4ffe5fec9e85f000475153ca2_b.jpg)



步骤：

1. 首先通过计算 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 与 ![[公式]](https://www.zhihu.com/equation?tex=g_j) 之间的余弦相似度来判断 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 是否与 ![[公式]](https://www.zhihu.com/equation?tex=g_j) 冲突；其中负值表示梯度冲突。
2. 如果余弦相似度是负数，我们用它在 ![[公式]](https://www.zhihu.com/equation?tex=+g+) 的法线平面上的投影替换。如果梯度不冲突，即余弦相似度为非负，原始梯度 ![[公式]](https://www.zhihu.com/equation?tex=g_i) 保持不变。

![[公式]](https://www.zhihu.com/equation?tex=g_j%3Ag_i+%3D+g_i+-+%5Cfrac%7Bg_i+g_j%7D%7B%7C%7Cg_j%7C%7C%5E2%7D%5C%5C)

其实，这篇文章是很有问题的。梯度冲突不一定是个坏事，而可以带来正则化的好处，如果只是把冲突梯度完全抹去恐有不妥。而且，文章并没有对梯度裁剪进行消融实验。万一是梯度裁剪、而不是Project conflict 导致的性能提升呢？不过，这个方法在我的测试上也是表现比原来的好的。