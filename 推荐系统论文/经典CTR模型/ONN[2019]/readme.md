# ONN：Operation-aware Neural Network [2019] -- FFM与NN的结合体

如果说之前的PNN是FM和DNN的结合体，那么ONN就是FFM和DNN的结合体。PNN给每个feature都给予一个embedding，然后做向量的点积。但是，ONN认为针对不同的**交叉操作**(内积or外积)、以及**不同field**的交叉，都应该用不同的Embedding。

如果用同样的Embedding，从好处来讲，就是不同的交叉操作会对对方都有一个正则化的效果，尤其是当数据量比较少的时候，可以缓解过拟合。但是，对于CTR/CVR任务，数据量从来就不是问题！那么，还只用一个embedding的话就只能是限制模型的capacity了。

其实，ONN的思路在本质上和FFM、AFM都有异曲同工之妙，这三个模型都是通过引入了额外的信息来区分不同特征交叉具备的信息表达。总结下来：

- FFM：引入**Field-aware**，对于field a来说，与field b交叉和field c交叉应该用不同的embedding
- AFM：引入Attention机制，a与b的交叉特征重要度与a与c的交叉重要度不同
- ONN：引入Operation-aware，field a与field b进行内积所用的embedding，不同于field a与field b进行外积用的embedding；field a与field b做内积用的embedding，亦不同于field a与field c做内积用的embedding。

![img](https://pic3.zhimg.com/v2-e6110f85fc89c4c6dc26b03c0d4cfd62_b.png)

上图中，一个feature有多个embedding。在图中以红色虚线为分割，第一列的embedding是feature本身的embedding，之后直接用来拼接过DNN的；从第二列开始往后是当前特征与**第n个特征**（field）以**方式m**（Operation：内积/外积）交叉所使用的embedding。

