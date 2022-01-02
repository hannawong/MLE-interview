# DeepFM [2017]

我们知道FM只能够去显式地捕捉二阶交叉信息，而对于高阶的特征组合却无能为力。DeepFM就是在FM模型的基础上，增加DNN部分，进而提高模型对于高阶组合特征的信息提取。DeepFM包含了FM和NN两部分，这两部分共享了Embedding层：

![img](https://pic4.zhimg.com/v2-dc51b2f7aac27fd712e4b95037de3813_b.png)

左侧就是一个FM: +号那里，表示的是FM中的一阶项，就是把各个 ![[公式]](https://www.zhihu.com/equation?tex=w) 加起来。乘号就是内积，最后把所有结果加起来，就是FM这部分的输出。

右侧是Embedding+MLP，也得到logit输出。最后把两边的输出加起来即可。

![img](https://pic2.zhimg.com/v2-af9547e2c18bfe8a6f7cbf2dfaebdb45_b.png)

**优点：**

- 模型具备同时学习低阶与高阶特征的能力
- 共享embedding层，共享了特征的信息表达

**不足：**

- DNN部分对于高阶特征的学习仍然是隐式的，为了解决此问题，可以使用显式有限阶交叉网络DCN、xDeepFM.