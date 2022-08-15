# 如何筛选特征

#### 1. 前向特征选择方法

从所有特征中选择特征的重要性最大的top k个特征加入模型。

##### 1.1 直接利用feature importance变量来判断特征重要性
我们在构建树类模型（XGBoost、LightGBM等）时，如果想要知道哪些变量比较重要的话，可以通过模型的feature\_importances\_ 方法来获取特征重要性。例如LightGBM的feature\_importances\_可以通过特征的分裂次数或利用该特征分裂后的信息增益来衡量。---这便是树模型的优势，其天生具有非常好的可解释性（不像神经网络的可解释性很差）。

##### 1.2 卡方独立性检验
卡方检验是一种**假设检验**方法，属于**非参数检验**。根本思想在于比较理论频数和实际频数的吻合程度。（可以用在A/B测试中，来判断当前的改变是否是有效的）
举个例子：

喝牛奶组和不喝牛奶组的感冒率为30.94%和25.00%，两者的差别可能是抽样误差导致，也可能是牛奶对感冒率真的有影响。那么假设喝牛奶对感冒发病率没有影响，即喝牛奶与感冒无关。之后就要检验这个假设是否成立，或者推翻这个假设。

**在理论上**，如果喝牛奶和不喝牛奶对感冒并没有任何影响的话，分布应该是均匀的，如下表：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210221162719575.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

但**实际的结果**却是：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20210221162835203.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



卡方检验的计算公式：
​                                                                           ![\chi^2 = \sum \frac{(A-T)^2}{T}](https://www.zhihu.com/equation?tex=%5Cchi%5E2%20%3D%20%5Csum%20%5Cfrac%7B(A-T)%5E2%7D%7BT%7D)  



其中，A 是每一项的理论值，T 是每一项的实际值。 ![\chi^2](https://www.zhihu.com/equation?tex=%5Cchi%5E2)  值的意义:衡量理论与实际的差异程度。在本例中， ![\chi^2=1.077](https://www.zhihu.com/equation?tex=%5Cchi%5E2%3D1.077)  
下面就需要检验这个值是否在拒绝域中。我们需要查询卡方分布的临界值，将计算的值与临界值比较。查询临界值就需要知道自由度：       

​                                                                         **自由度V=(行数-1)*(列数-1)**             
对于该问题V=1，查询可得临界值为3.84.
如果  $\chi^2$<临界值 则假设成立。

可以用卡方检验来提取特征，起到特征降维的目的。

##### 1.3 信息增益
使用决策树中的信息增益计算方法，即计算按此特征分类之后熵的减少程度，以此来衡量特征的重要程度。
另外还可以用GBDT来构造特征。（见经典推荐模型/GBDT+LR）

##### 1.4 互信息
和信息增益类似。
https://github.com/hannawong/MLE-interview/tree/master/%E7%BB%9F%E8%AE%A1%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0/%E5%90%84%E7%A7%8D%E7%86%B5%E3%80%81KL%E6%95%A3%E5%BA%A6%E3%80%81%E4%BA%92%E4%BF%A1%E6%81%AF

#### 2. 后向特征选择
首先使用所有特征进行训练，然后从中选择特征的重要性最小的特征移除然后重新训练。

##### 2.1 permutation importance
常规思路：首先让全部特征参与训练然后预测得出score1（mse,rmse等），然后依次去掉一个特征去训练模型（有多少个特征就会训练多少个模型），分别预测会得到对应的缺失特征的得分score2，score2-score1就代表一个特征的预测能力。然而，有100个特征岂不是要训练100个模型？这样做太麻烦了，所以Permutation importance 的算法是这样的：

- 用全部特征，训练一个模型。
- 验证集预测得到得分。
- **验证集**的一个特征列的值进行随机打乱，预测得到得分。将上述得分做差即可得到特征x1对预测的影响。
- 依次将每一列特征按上述方法做，得到每个特征对预测的影响。



