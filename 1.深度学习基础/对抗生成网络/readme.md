# 对抗生成网络

GAN的主要结构包括

- **生成器**G（Generator）：输入random noise，输出假的图片；
- **判别器**D（Discriminator）：判断是真的图片，还是生成器生成的假图片。

训练流程如下：

- 初始化判别器D的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_%7Bd%7D) 和生成器G的参数 ![[公式]](https://www.zhihu.com/equation?tex=%5Ctheta_g) 。
- 从真实样本中采样 ![[公式]](https://www.zhihu.com/equation?tex=m) 个样本 { ![[公式]](https://www.zhihu.com/equation?tex=x%5E1%2C+x%5E2%2C+...+x%5Em+) } ，从先验分布噪声中采样 ![[公式]](https://www.zhihu.com/equation?tex=m) 个噪声样本 { ![[公式]](https://www.zhihu.com/equation?tex=z%5E1%2C+z%5E2%2C+...%2Cz%5Em+) } 并通过生成器获取 ![[公式]](https://www.zhihu.com/equation?tex=m+) 个生成样本 { ![[公式]](https://www.zhihu.com/equation?tex=%5Cwidetilde+x%5E1%2C+%5Cwidetilde+x%5E2%2C+...%2C+%5Cwidetilde+x%5Em) } 。固定生成器G，训练判别器D尽可能好地准确判别真实样本和生成样本，尽可能大地区分正确样本和生成的样本。
- **循环k次更新判别器之后，使用较小的学习率来更新一次生成器的参数**，训练生成器使其尽可能能够减小生成样本与真实样本之间的差距，也相当于尽量使得判别器判别错误。
- 多次更新迭代之后，最终理想情况是使得判别器判别不出样本来自于生成器的输出还是真实的输出。亦即最终**样本判别概率均为0.5**。

> Tips: 之所以要训练k次判别器，再训练生成器，是因为要先拥有一个**好的判别器**，使得能够教好地区分出真实样本和生成样本之后，才好更为准确地对生成器进行更新。



论文中包含min，max的公式如下图形式：

![img](https://pic2.zhimg.com/80/v2-9aaed21e79bebcc6638742fb126de225_1440w.jpg)

其中，pdata是真实的分布。首先，固定生成器，来训练一个好的判别器。第一项的意思就是，希望判别器给真实的样本以接近1的预测值。z是噪声，G(z)就是生成器生成的“假样本”。第二项的意思就是，希望判别器给假样本以接近0的预测值。

训练了k步判别器之后，要训练生成器了。生成器的loss函数是要最小化第二项，意思就是要让判别器对于假样本的输出更加接近于1.

论文中的算法描述：

![img](https://pic2.zhimg.com/80/v2-30351197d426b7610b479d26869df8e9_1440w.jpg)



# Multi-scale GAN

MSGAN被用于根据几张连续的图片，去预测未来若干时间点的图片。使用GAN比起MSE loss更不容易模糊，也更加接近真实的情况。在气象局的项目中，我们用MSGAN来预测降水。这是因为降水不同于温度湿度可以直接用生成模型，它更需要保证真实性。

一种最简单的想法是用**卷积**来根据历史图片获得输出，然后直接用Lp loss（这就是我们预测温度湿度的方法）,例如：
![img](https://pic3.zhimg.com/80/v2-7ea26df2d0d6ec2e4d6dc787ccea94c2_1440w.jpeg)

但是，这样做有两个问题：

- 卷积核只能看到很小的范围，不能关注全局。（所以才需要用Multi-scale，或者UNet）
- l2 loss会导致模糊。这是因为，假如v1和v2都是比较可能的值，虽然(v1+v2)/2能够降低l2 loss，但是(v1+v2)/2并不是一个可能值。

为此，我们引入了Multi-scale GAN。用了不同尺度的图片来做GAN，这样能够把握住不同尺度的信息（局部+全局）

公式：

![img](https://pic3.zhimg.com/80/v2-1c894ed08ef7a87757345a21be564d5e_1440w.jpeg)


![img](https://pic1.zhimg.com/80/v2-36af2b07e8591d309e93f6d6f67a1f05_1440w.jpeg)



训练过程：

- 训练判别器：

![img](https://pica.zhimg.com/80/v2-e299ec182acb3d23e9b3d0031b9f221f_1440w.png)

- 训练生成器：

​       生成器的损失函数由三部分组成：

 - - adversarial loss：让判别器对生成的假样本输出为1.


   ![img](https://pica.zhimg.com/80/v2-051b36559194b95ff24ccdf0e54fc5bf_1440w.png)



   - Lp loss: 生成接近于真实样本的输出。

   - Gradient Difference Loss：
     ![img](https://pic1.zhimg.com/80/v2-73919b47d4c10c843c06a5bac4ccc651_1440w.jpeg)

     让生成的样本更sharp（不模糊）
