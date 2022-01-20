# EM算法

全称 Expectation Maximization Algorithm。期望最大算法是一种迭代算法，用于含有**隐变量**的参数模型的最大似然估计或极大后验概率估计。



### 0x01. 预备知识

##### 1.1 极大似然估计

我们需要调查我们学校的男生和女生的身高分布。 假设你在校园里随便找了100个男生和100个女生。他们共200个人。将他们按照性别划分为两组，然后先统计抽样得到的100个**男生** 的身高。假设他们的身高是服从正态分布的。但是这个分布的均值  ![\mu](https://www.zhihu.com/equation?tex=%5Cmu)  和方差  ![\sigma^2](https://www.zhihu.com/equation?tex=%5Csigma%5E2)   我们不知道，这两个参数就是我们要估计的。记作 ![θ = [ μ , σ ]](https://www.zhihu.com/equation?tex=%CE%B8%20%3D%20%5B%20%CE%BC%20%2C%20%CF%83%20%5D) 。 
  
问题数学化. 设样本集  ![X = x_1 , x_2 , … , x_N](https://www.zhihu.com/equation?tex=X%20%3D%20x_1%20%2C%20x_2%20%2C%20%E2%80%A6%20%2C%20x_N)  ，其中 N = 100， ![p ( x_i ∣ θ ) ](https://www.zhihu.com/equation?tex=p%20(%20x_i%20%E2%88%A3%20%CE%B8%20)%20)  表示在一定参数下抽到男生身高为  ![x_i](https://www.zhihu.com/equation?tex=x_i)  的概率。由于100个样本之间独立同分布，所以我同时抽到这100个男生的概率就是他们各自概率的乘积：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220161538785.png)

这个概率反映了在参数是  ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)   时，得到 X 这组样本的概率。 我们需要找到一个参数  ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)   ，使得抽到 X 这组样本的概率最大，即 $L(\theta)$最大。这个  ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)  的最大似然估计量，记为 θ′ = argmax L(θ)

求最大似然函数估计值的一般步骤：

- 首先，写出似然函数
- 然后，对似然函数取对数：
- 接着，对上式按 θ 求导，令导数为0，得到似然方程；
- 最后，求解似然方程，得到的参数 θ 即为所求。



##### 1.2 琴生不等式

> 设 f 是定义域为实数的函数，如果f(x)的二次导数恒大于等于0，那么f是凸函数。

Jensen不等式表述如下：如果 f  是凸函数，X 是随机变量，那么： ![E [ f ( X ) ] ≥ f ( E [ X ] ) ](https://www.zhihu.com/equation?tex=E%20%5B%20f%20(%20X%20)%20%5D%20%E2%89%A5%20f%20(%20E%20%5B%20X%20%5D%20)%20)  。当且仅当 X 是常量时，上式取等号。

例如，图2中，实线 f  是凸函数， X  是随机变量，有0.5的概率是a，有0.5的概率是 b。X的期望值就是a和b的中值了，图中可以看到 ![ E [ f ( X ) ] ≥ f ( E [ X ] )](https://www.zhihu.com/equation?tex=%20E%20%5B%20f%20(%20X%20)%20%5D%20%E2%89%A5%20f%20(%20E%20%5B%20X%20%5D%20)) 成立。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220190159228.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)



## 0x02. EM算法详解

##### 2.1 问题描述

先看一个例子：

我们目前有100个男生和100个女生的身高，共200个数据，但是我们不知道这200个数据中哪个是男生的身高，哪个是女生的身高。假设男生、女生的身高分别服从正态分布。这个时候，对于每一个样本，就有两个方面需要猜测或者估计： **这个身高数据是来自于男生还是来自于女生**？男生、女生身高的正态分布的参数分别是多少？EM算法要解决的问题正是这两个问题。
又如样本点属于若干个正态分布，我们不知道每个正态分布的参数，**也不知道这些样本点都是属于哪个分布**的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220142101196.png)

身高来自男生还是女生、样本点属于哪个分布，这些就是”隐变量“。

##### 2.2 推导

给定数据集，假设样本间相互独立，我们想要拟合模型 ![[公式]](https://www.zhihu.com/equation?tex=p%28x%3B%5Ctheta%29) 到数据的参数。根据分布我们可以得到如下似然函数：

![[公式]](https://www.zhihu.com/equation?tex=%5Cbegin%7Baligned%7D+L%28%5Ctheta%29+%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dlog+p%28x_i%3B%5Ctheta%29++%5C%5C+%26%3D+%5Csum_%7Bi%3D1%7D%5E%7Bn%7Dlog+%5Csum_%7Bz%7Dp%28x_i%2C+z%3B%5Ctheta%29+%5Cend%7Baligned%7D+%5C%5C)

第一步是对极大似然函数取对数，第二步是对每个样本的每个可能的类别 z 求联合概率分布之和。

对于每个样本 i，我们用 ![[公式]](https://www.zhihu.com/equation?tex=Q_i+%28z%29) 表示样本 i 隐含变量 z 的某种分布，且 ![[公式]](https://www.zhihu.com/equation?tex=Q_i+%28z%29) 满足条件（ ![[公式]](https://www.zhihu.com/equation?tex=%5Csum_%7Bz%7D%5EZQ_%7Bi%7D%28z%29%3D1%2C+%5Cquad+Q_%7Bi%7D%28z%29+%5Cgeq+0) ）。

我们将上面的式子做以下变化：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220191901841.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80MTMzMjAwOQ==,size_16,color_FFFFFF,t_70)

(3)式是由(2)式根据Jensen不等式得到。这里简单介绍一下(2)式到(3)式的转换过程：

![img](https://pic2.zhimg.com/80/v2-04e012b6ea03a151108bbde349a5c4b9_1440w.png)



，而 log(x)为凹函数，根据Jensen不等式,当 f 是凹函数时，E[f(X)]≤f(E[X]) 成立，故可由(2)式得到(3)式。

上述过程可以看作是对  ![logL(θ)](https://www.zhihu.com/equation?tex=logL(%CE%B8))   求了下界。对于 ![Q_i(z^{(i)})](https://www.zhihu.com/equation?tex=Q_i(z%5E%7B(i)%7D))  --即隐变量--的选择有多种可能，那么哪种更好呢？假设 ![\theta](https://www.zhihu.com/equation?tex=%5Ctheta)  已经给定，那么![logL(θ)](https://www.zhihu.com/equation?tex=logL(%CE%B8))的值就取决于 ![Q_i(z^{(i)})](https://www.zhihu.com/equation?tex=Q_i(z%5E%7B(i)%7D))和 ![p(x^{(i)},z^{(i)})](https://www.zhihu.com/equation?tex=p(x%5E%7B(i)%7D%2Cz%5E%7B(i)%7D))  了,这两个值都是只和隐变量有关的。我们可以通过调整这两个概率使下界不断上升，以逼近 ![ logL(θ)](https://www.zhihu.com/equation?tex=%20logL(%CE%B8))  的真实值，那么什么时候算是调整好了呢？当不等式变成等式时，说明我们调整后的概率能够等价于 ![ logL(θ)](https://www.zhihu.com/equation?tex=%20logL(%CE%B8)) 了。等式成立的条件是什么呢？根据Jensen不等式，要想让等式成立，需要让随机变量变成常数值：

![img](https://pic1.zhimg.com/80/v2-3619442456a7dca630444102ad03e3ae_1440w.png)

对此式做进一步推导：由于

![img](https://pic3.zhimg.com/80/v2-45fc0e4c0e92006fc16e1806861ee9d8_1440w.png)

因此得到下式：

![z是隐型变量(男\女)，x是观测的值。](https://img-blog.csdnimg.cn/2020122019243499.png)

z是隐型变量（男\女），x是观测的值。

至此，我们推出了: 在固定参数θ后，![Q_i(z^{(i)})](https://www.zhihu.com/equation?tex=Q_i(z%5E%7B(i)%7D))的计算公式就是后验概率，即：我们假定了参数就是 θ , 那么每个样本 ![x^{(i)}](https://www.zhihu.com/equation?tex=x%5E%7B(i)%7D)  属于各个类的概率会是多少。这一步就是E步，来让 ![logL(θ)](https://www.zhihu.com/equation?tex=logL(%CE%B8))  的下界逼近 ![logL(θ)](https://www.zhihu.com/equation?tex=logL(%CE%B8)) 的真实值。

接下来的M步，就是在给定隐变量 ![Q_i(z^{(i)})](https://www.zhihu.com/equation?tex=Q_i(z%5E%7B(i)%7D))  后，调整 θ ，去极大化 logL(θ) 的下界。

##### 2.3 步骤

1、初始化分布参数 θ； 重复E、M步骤直到收敛：
2、E步骤：根据现在的参数θ，来计算出隐性变量(如类别)的后验概率（即隐性变量的期望），作为隐性变量的现估计值：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220193127511.png)

3、M步骤：将似然函数最大化以获得新的参数值θ ,即更新θ 



**举例**

两种硬币，但是两个硬币的材质不同导致其出现正反面的概率不一样。目前我们只有一组观测数据，要求出每一种硬币投掷时正面向上的概率。总共投了五轮，每轮投掷五次。
1、现在先考虑一种简单的情况，假设我们知道这每一轮用的是哪一个硬币去投掷的：

![在这里插入图片描述](https://img-blog.csdnimg.cn/2020122019344623.png)

那么我们拿着这样的一组数据，就可以很轻松的估计出A硬币和B硬币出现正面的概率，如下：
PA = ( 3 + 1 + 2 ) / 15 = 0.4 

PB = ( 2 + 3 ) / 10 = 0.5

2、现在把问题变得复杂一点，假设我们不知道每一次投掷用的是哪一种硬币，等于是现在的问题加上了一个**隐变量**，就是每一次选取的硬币的种类。

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220193653795.png)

那么现在可以想一想，假设我们把每一次硬币的种类设为z,则这五次实验生成了一个5维的向量`(z1,z2,z3,z4,z5)`.

那么这个时候EM算法的作用就体现出来了! EM算法的基本思想是：先初始化一个PA,PB(就是上文的θ)，然后我们拿着这个初始化的PA,PB用最大似然概率估计出z，接下来有了z之后，就用z去计算出在当前z的情况下的PA,PB是多少，然后不断地重复这两个步骤直到收敛。

有了这个思想之后现在用这个思想来做一下这个例子，假设初始状态下PA=0.2, PB=0.8，然后我们根据这个概率去估计出z：

![在这里插入图片描述](https://img-blog.csdnimg.cn/20201220194029393.png)

标粗体的是按最大似然估计，最有可能的硬币种类。
按照最大似然估计，z=(B,A,A,B,A)，有了z之后我们反过来重新估计一下PA,PB：
PA = （2+1+2）/15 = 0.33
PB =（3+3）/10 = 0.6

可以看到PA,PB的值已经更新了，假设PA,PB的真实值0.4和0.5，那么你在不断地重复这两步你就会发现PA,PB在不断地靠近这两个真实值。




EM算法在GMM、K-means中有应用。