## BM25

首先，让我们来看一下BM25的公式，随后我将会细细分解这个公式：

![img](https://pic4.zhimg.com/v2-f5804e5c248660833e862e53a595b7b3_b.png)

- ![q_i](https://www.zhihu.com/equation?tex=q_i) 表示第 ![i](https://www.zhihu.com/equation?tex=i)个query term。

比如搜索"Hogwarts School"，ElasticSearch会按照white space将其划分，于是得到两个token："Hogwarts", "School". 由公式可知，BM25值就是将所有token的得分计算加和。

- ![IDF(qi)](https://www.zhihu.com/equation?tex=IDF(qi))IDF(qi) 是第 ![i](https://www.zhihu.com/equation?tex=i)i 个query term的逆文档频率(inverse document frequency)。

这里的IDF和TF-IDF中的IDF类似，都是用来惩罚那些出现在很多document中的词语，只是有一些小小的不同。Lucene/BM25的IDF计算公式如下：

![img](https://pic3.zhimg.com/v2-8f334449440cddf735246c5be476eea6_b.png)

其中， ![docCount](https://www.zhihu.com/equation?tex=docCount)docCount 是在ElasticSearch的一个shard (或者多个shards) 中的document个数； ![f(q_i)](https://www.zhihu.com/equation?tex=f(q_i))f(q_i) 是含有 ![q_i](https://www.zhihu.com/equation?tex=q_i)q_i 的document的个数。

举个例子，假如总共有4个document，"school"出现在2个document中，那么IDF("school")为：

![img](https://pic2.zhimg.com/v2-fd3f0efd0d6d9c1b56f5c58d4a3eb785_b.png)

也就是说，我们要给罕见的term分配较高的权重。

- fieldLen/avgFieldLen

在分母中的 ![fieldLen/avgFieldLen](https://www.zhihu.com/equation?tex=fieldLen%2FavgFieldLen)其实是**【给那些长document以惩罚】**（这里的length是用term个数衡量的）。这也是符合我们的直觉的：假如一篇300页的文章提过一次query中的词，那肯定不如一个短短的句子里面提过query更相关。

- $b$ : 这是一个决定 ![fieldLen/avgFieldLen](https://www.zhihu.com/equation?tex=fieldLen%2FavgFieldLen)fieldLen/avgFieldLen 影响大小的超参数。b越大，document长度的惩罚就越大。在ElasticSearch中，b的default值取0.75.

- ![f(q_i,D) ](https://www.zhihu.com/equation?tex=f(q_i%2CD)%20)f(q_i,D)  

第 ![i](https://www.zhihu.com/equation?tex=i)i 个 query term在document D中出现的次数。当然越多越好。

- $k_1$: 用来决定[term frequency saturation](https://www.elastic.co/guide/en/elasticsearch/guide/current/pluggable-similarites.html#bm25-saturation)。即，限制了一个query term最多能够对最后的score有多大的影响。例如，一个文章中出现了20次query term和出现1000次query term的效果应该是差不多的。如果不做此限制，那么那些高频的词的tf值就会过大，导致整个query的得分都被那些高频词所主导。ElasticSearch中，default k1 = 1.4。BM25和TFIDF的对比如下图所示：

![img](https://pic1.zhimg.com/v2-60c54715687bb960954aea9162bc590c_b.png)


  