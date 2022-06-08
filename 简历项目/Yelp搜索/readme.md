# ColXLM

### 1. 背景

原来Yelp搜索引擎召回模块是基于NrtSearch(ElasticSearch)的升级版本，并没有使用深度模型，而是基于倒排索引的。这就非常有问题了。

- 不能召回近义词（搜"sneaker",不能召回"shoes"）
- 不支持多语言检索。

实际上，yelp将输入的query先经过一些处理，比如query扩增(women->woman), query纠错（iphonne->iphone）,query翻译(“饺子”->dumpling). 但是，这还是会带来一些问题。例如，搜“饺子”的时候，甚至会把“饺子”切词切成“饺”&“子”，然后去召回了一些含有“子”的店铺。

不可否认，基于exact-match的召回是非常重要的一路，但是仅仅靠exact-match还是远远不够的。

### 2. 预训练

![img](https://github.com/hannawong/ColXLM/raw/main/fig/ColBERT-Framework-MaxSim-W370px.png)

####  2.1 预训练task

mBERT和XLM都是非常成功的语言模型，但是他们并不是针对检索任务的。在这个项目中，我**专门针对检索任务**设计的task：

- Relevance Ranking Task (RR)：就是直接的检索任务。训练数据是MSMARCO给的数据集（已经有了正负样本），还有wiki数据集。wiki数据集的采样方法就是用inverse cloze task（ICT），每个文章选出一句作为query，选出一段作为相关的document。这样获得<query, doc+,doc->对，模型直接预测哪一个是正样本。用ColBERT输出的这个score做softmax，然后使用cross-entropy loss。这个是句子层面的检索任务。
- Query Language Modeling Task (QLM)：类似于BERT中的MLM，是单词层面的检索任务。mask掉一些query token，然后让document来预测之。
- Representative wOrds Prediction：用sampling的方法找出document的likelihood最高的query，即“Representative Word Sets Sampling”。



为了支持多语言，用翻译模型把MSMARCO数据集翻译成15种语言；wiki数据集选用multi-wiki。



#### 2.2 预训练细节

在mBERT 上继续训练。所以，已经隐式地用MLM任务预训练过了。对每种语言，都依次训三种预训练任务，每个step都做梯度回传。每种语言训20万次个batch。



### 3. index

对document，把每个document的每个token embedding存入faiss索引。900万document可以在3小时内建好索引。



----

【一些思考】

  其实在来美国之前，我一直奇怪一件事情：为什么美国的公司加班那么少，但是效率那么高，创造出那么多利润；为什么我们的员工每天那么辛苦，但是公司创造的价值却比不上美国？

半年前我来到美国读书，在这期间我与Yelp和 JP Morgan都有过合作。我惊讶的发现，他们在工程上实际使用的模型都非常之简单。Yelp的搜索引擎是Nrtsearch，就是一个类似elasticsearch的精确查找引擎，甚至都没有用深度模型—这在国内的公司里是不可想象的。JPMorgan用来做语音意图识别的模型就是cnn+lstm+self-attention。简单到让人怀疑人生，但是真的能用。而且这些简单的模型可能已经用了很多年，所以后面的人就是在这个框架上修修补补。

而我们对于改革，有一种盲目的崇拜。似乎只有不断的变革、推翻，才能体现出我的价值。我本科的时候在中国的互联网公司实习，后来过了一年，我又问他们现在在用什么模型，然后发现和我在那的时候用的模型已经完全不一样了。那之前的工作是不是很多都白干了？上线全新的模型，又要消耗多少人力物力财力？

所以我试图回答一下开头提出的问题，这个问题的产生有很多原因，但其中一个原因大概是：美国的公司一般有一套非常成熟的体系和模型，且不会频繁的在工程上变动；而在我们的公司里，模型更新换代过于频繁，导致了资源的浪费。所以，不是我们不够聪明、不够优秀，而是我们的努力很多都白白耗散了。

**“利不百，不变法；功不十，不易器”。**

改革都是需要成本的。只有当改革带来的好处减去成本大于一个threshold的时候，我们才需要去改变现有的方法。

这当然不意味着不用紧跟时代了—最新顶会的趋势和热点都要不断关注，但要带着批判的眼光来看，不是越新的模型越好、不是越复杂的模型越好、不是论文里report出的结果越高越好。

要消除对改革的盲目崇拜。在再次出发之前，不妨回头看看，我们已经走过了怎样的路。  







Pytorch实验中每次结果不一致，复现困难，同样的模型数据和参数，跑出效果好的模型变成小概率事件。

原因分析：
尝试固定住电脑的随机数，排除随机数的干扰。
解决方案：
np.random.seed(seed)
torch.manual_seed(seed) #CPU随机种子确定
torch.cuda.manual_seed(seed) #GPU随机种子确定
torch.cuda.manual_seed_all(seed) #所有的GPU设置种子