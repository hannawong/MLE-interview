ERNIE(Enhanced Representation through kNowledge IntEgration)是对于BERT的一种改进，这篇文章我们介绍三个ERNIE的版本：百度ERNIE1.0 & 2.0, 和清华ERNIE。

## ERNIE1.0 

论文题目：ERNIE: Enhanced Representation through Knowledge Integration

ERNIE1.0 并没有对BERT模型进行任何修改，而是在**mask的方式**上下功夫，使得其分层次地进行mask -- 有basic-level mask, phrase-level mask 和 entity-level mask。ERNIE1.0 在自然语言推理、语义相似度、命名实体识别、情感分析和问答等任务中均取得了更好的结果。

为什么要做不同层次的mask呢？原始BERT的mask方式有什么问题呢？

我们来看下面这个例子：

“Harry Potter is a series of fantasy novels written by J.K.Rowling”

Harry Potter 和 J.K. Rowling 是两个**entity**，如果我们只用BERT的方式随机mask，假如mask掉"K.", 那么我们根据"J."和"Rowling"也可以预测出"K.", 这样模型就根本没有学习到"Harry Potter"和"J.K. Rowling"这两个实体之间的**关系**（即：author关系）。 如果我们把"J.K.Rowling"这个entity整个mask掉，那么就可以“逼着”模型去学两个实体之间的关系。注意这种学习两个实体之间的关系的方法是隐式的，我们并没有显式的加入knowledge embedding, 而是完全靠mask的方法学到的。这一点和下文介绍的清华ERNIE有所不同。

![img](https://pic2.zhimg.com/80/v2-a70cb4737bd103f513c2a21b4130ed9c_1440w.png?source=d16d100b)

不同粒度的masking

ERNIE1.0是用多阶段(multi-stage)的方法来融合不同粒度的表征的。

- 第一阶段：和BERT类似的basic-level masking. 对英文来说是**word(注意不是subword)**、对中文来说是character。这个阶段我们只能学习到一个词的表示，而更高级的relationship信息没有学到。
- 第二阶段：Phrase-Level Masking。对于英语，使用**词法分析工具来获取句子中短语的边界**；对于中文，使用分词工具来获取短语信息。在这个阶段，短语信息被编码到embedding中。
- 第三阶段：Entity-Level Masking。entity包括人名、地点、组织、产品等。在实体mask阶段，我们首先分析一个句子中的命名实体，然后mask实体中的所有词。

在训练过程当中，模型没有进行任何改变。基于如上提出的mask策略，本文是一级一级训练的，先基于原Bert的MLM策略进行训练，然后再基于phrase-level masking进行训练，最后再基于Entity-level masking进行训练。粒度由粗到细，知识融合也越来越精细。 



## ERNIE 2.0 

论文题目：ERNIE 2.0: A Continual Pre-Training Framework for Language Understanding

我理解ERNIE2.0 比 1.0的改进就是它使用了**更丰富的预训练任务**，来获取**词汇(lexical)、句法(syntactic)和语义(semantic)**信息。ERNIE2.0使用连续预训练框架，**增量**地构建预训练任务，然后在这些构建的任务上通过连续多任务学习（continual multi-task learning）学习预训练模型。

什么是连续学习(**continual learning)**呢？

持续学习是用几个任务顺序地训练模型，在学习新任务时需要**记住**之前学习过的任务。通过不断的学习，模型能够很好地完成新的任务，同时这也要归功于之前的训练中获得的知识。

由于我们**随时可能会添加新的任务**，所以为了使得这个过程可控，通常我们需要用持续学习（Continual Learning）进行多任务的管理，如下图右一所示。然而，深度模型训练过程中存在**遗忘效应**，经常后面训练的任务会**覆盖**之前学习到的知识；这个时候就可以考虑采用多任务学习（Multi-task learning）的方式进行学习，如下图左2所示，**通过联合多个任务同时训练**，可以保证模型同时能对多任务的信息进行感知。将两者结合起来，就有了持续多任务学习框架了，如下图左一所示。

![img](https://pic3.zhimg.com/80/v2-3d884a021bf96235ec5d79d398c908fa_1440w.png?source=d16d100b)



ERNIE框架支持不断引入各种定制任务，这就是通过不断的多任务学习来实现的。连续多任务学习法在给定一个或多个新任务时，将新引入的任务与原来的任务同时进行有效的训练，不会忘记之前学习过的知识。

ERNIE添加了粒度从粗到细的一系列task：

- word-aware pretraining task
  - Knowledge masking: 就是ERNIE 1.0里面提出的**entity-level和phrase-level masking策略**。
  - Capitalization Prediction Task：与句子中的其他词语相比，**大写**的词语通常具有特定的语义信息。这种知识在**命名实体识别等任务中有一些优势，因此可以添加一个任务来预测单词是否大写**。
  - Token-Document Relation:该任务预测某个片段中的token是否出现在文本的其他段中。根据经验，文本中多次出现的词通常是常用的词或与该文本的主题有关。因此，通过识别片段中出现的文本中经常出现的单词，该任务可以使模型在一定程度上捕获文本的关键字。
- Structure-aware pre-training task
  - Sentences Reordering: 将给定的段落随机划分为1到m段，然后将其打乱，预测段的正确顺序
  - Sentences Distance：这个是一个分类任务，如果给定两个句子是领接的，那么判断为0；如果给定句子是同一个文章的，那么判断为1；如果给定句子是来自不同文档的，判断为2。这个可以用来以句子的角度去判断文章主题。
- Semantic-aware pretraining task
  - Discourse Relation：通过Sileo [et.al](http://et.al/) 的工作产生数据集，判断给定句子之间的修辞，语义承接关系。
  - IR Relevance: 是一个分类任务，采用了百度自己内部搜索结果的数据，对于某个Query而言，产生的doc的Title，如果这两个是匹配的（展示了，而且用户点击了），判断为0；如果在搜索引擎上展示了但是用户没点击，那么是1；如果完全不展示，那么是2。

通过这些预训练任务，ERNIE2.0能够获得更多粒度的信息，把握实体之间的关系、句子结构、句子之间关联、语义相似性等信息，而原始的BERT只有MLM和NSP任务，所以只能把握住token-level 和 sentence-level的信息。

