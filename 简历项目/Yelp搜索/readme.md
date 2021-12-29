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



