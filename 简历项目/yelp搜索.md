# ColXLM

### 1. 背景

原来Yelp搜索引擎召回模块是基于NrtSearch(ElasticSearch)的升级版本，并没有使用深度模型，而是基于倒排索引的。这就非常有问题了。

- 不能召回近义词（搜"sneaker",不能召回"shoes"；搜"ribeye"不能召回"steak"）

实际上，yelp将输入的query先经过一些处理，比如query扩增(women->woman), query纠错（iphonne->iphone）,query翻译(“饺子”->dumpling). 但是，这还是会带来一些问题。例如，搜“饺子”的时候，甚至会把“饺子”切词切成“饺”&“子”，然后去召回了一些含有“子”的店铺。

不可否认，基于exact-match的召回是非常重要的一路，但是仅仅靠exact-match还是远远不够的。

### 2. 预训练

![img](https://github.com/hannawong/ColXLM/raw/main/fig/ColBERT-Framework-MaxSim-W370px.png)

####  2.1 预训练task

mBERT和XLM都是非常成功的语言模型，但是他们并不是针对检索任务的。在这个项目中，我**专门针对检索任务**设计的task：

- Relevance Ranking Task (RR)：就是直接的检索任务。训练数据是7天的用户搜索-点击数据，1亿条，只考虑query文本和点击商铺的简介（包括菜谱和商家介绍、还有提取出来的keyword）。然后通过in-batch随机负采样得到负样本。这样获得<query, doc+,doc->对，模型直接预测哪一个是正样本。用ColBERT输出的这个score做softmax，然后使用cross-entropy loss。这个是句子层面的检索任务。
- Query Language Modeling Task (QLM)：类似于BERT中的MLM，是单词层面的检索任务。mask掉一些query token，然后通过document来预测之。“逼着document”让它恢复出query，这是一个很创新的想法！QLM任务和Facebook search中用document去预测query类别的想法异曲同工，都是强迫document去理解query意图。（在实现的时候，只是把query和document拼接起来，然后套用BERT做MLM的方式，但现在想想，似乎应该让query只关注document）



为了支持多语言，把训练的数据集翻译成15种不同的语言，这样能够支持multilingual search，但是不能支持cross-lingual search。其实这也是经过我深思熟虑之后的结果。一开始，我打算训一个cross-lingual model，比如搜索"饺子", 可以返回"dumpling". 但是，后来发现这样做召回准确率很低，在进行文献调研的时候我本来就发现cross-lingual information retrieval的文章很少，其实就是因为这个任务过于困难，且实际意义不大 -- 毕竟，我们可以先把query翻译成英语，然后再做召回；而且，如果我用中文搜"饺子"，可能就说明我想要找中文商铺，如果召回英文结果，可能不是用户想要的。



#### 2.2 预训练细节

在mBERT 上继续训练。所以，已经隐式地用MLM任务预训练过了。对每种语言，都依次训两种预训练任务，每个step都做梯度回传。每种语言训20万次个batch。



### 3. index

对document，把每个document的每个token embedding存入faiss索引。900万document可以在3小时内建好索引。



#### 4. 查询

首先，先用距离找出用户所在位置附近 100 mile之内的所有商铺，这个大概是10万量级的（这个数据量就已经很小了）。之后，我们要在这10万多个商铺中，召回和query相关的100余个商铺，作为后序排序工作的candidate。可以达到500 QPS。



#### 5. 结果

离线结果：在MSMARCO数据集上的Recall @ 1000 达到95.7 %； 在搜索log上，top10 precision90%以上。

线上结果：

增加了向量召回这一路之后（当然还做了很多其他的改进，比如组内其他人做的多模态召回、增加用户个性化进行排序之类的优化），线上的CTR上升了。但是带来的一个问题就是召回的相关度下降了。->所以为了避免更多的bad case，我们可能需要一个相关性控制模块，来筛选掉那些exact-match很少的查询结果。

#### 6. 其他可优化点

- 随机负采样只能得到简单负样本，为了得到梯度更大的难负例，我们应该用一些难负例采样方法，如ACNE、BM25采样等。
- 这里所有token都具有同样的权重，然而我们知道，不同的query token应该具有不一样的权重。



----





```python
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertTokenizerFast
from colXLM.parameters import DEVICE

class ColBERT(BertPreTrainedModel):
    def __init__(self, config, query_maxlen, doc_maxlen, mask_punctuation, dim=128, similarity_metric='cosine'): ##query_maxlen = 32, doc_maxlen = 512

        super(ColBERT, self).__init__(config)
        self.config = config
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.similarity_metric = similarity_metric
        self.dim = dim

        self.mask_punctuation = mask_punctuation ##True, 是否把document中的停用词都去掉
        self.skiplist = {}

        if self.mask_punctuation:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
            self.skiplist = {w: True
                             for symbol in string.punctuation
                             for w in [symbol, self.tokenizer.encode(symbol, add_special_tokens=False)[0]]}

        self.bert = BertModel(config)
        self.linear = nn.Linear(config.hidden_size, dim, bias=False) ##最后把每个token都映射到128维，完成降维

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = nn.ReLU()
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) #【注1】eps是为了防止分母除以0
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias


        self.init_weights()

    def MLMhead(self,hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

    def forward(self, Q, D,mode,label = None):
        if mode == "rr": ##"*"表示解压缩，Q其实为[input_ids,attention_mask], D也为[input_ids,attention_mask]
            return self.score(self.query(*Q), self.doc(*D))
        
        if mode == "qlm": 
            return self.qlm_score(Q,D,label)


    def query(self, input_ids, attention_mask):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        Q = self.bert(input_ids, attention_mask=attention_mask)[0]
        Q = self.linear(Q)

        return torch.nn.functional.normalize(Q, p=2, dim=2)

    def doc(self, input_ids, attention_mask, keep_dims=True):
        input_ids, attention_mask = input_ids.to(DEVICE), attention_mask.to(DEVICE)
        D = self.bert(input_ids, attention_mask=attention_mask)[0] ##过bert
        D = self.linear(D)##降维

        mask = torch.tensor(self.mask(input_ids), device=DEVICE).unsqueeze(2).float()
        D = D * mask##去掉后面不足512位的

        D = torch.nn.functional.normalize(D, p=2, dim=2) 

        return D

    def score(self, Q, D): 

        if self.similarity_metric == 'cosine': ##maxsim
            return (Q @ D.permute(0, 2, 1)).max(2).values.sum(1)


    def mask(self, input_ids):
        mask = [[(x not in self.skiplist) and (x != 0) for x in d] for d in input_ids.cpu().tolist()]
        return mask

    def qlm_score(self,Q,D,label):
        QD = torch.cat([Q[0], D[0]], axis = 1).cuda() ##Query和Docuemnt的input id拼接起来
        QD_mask = torch.cat([Q[1], D[1]], axis = 1).cuda() ##query和document的mask拼接起来
        output = self.bert(QD, attention_mask = QD_mask) ##得到bert输出
        sequence_output = output[0]
        prediction_scores = self.MLMhead(sequence_output) ##得到mlm输出
        loss_fct = torch.nn.CrossEntropyLoss()  # -100 index = padding token
        label = label.cuda()
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), label.view(-1))
        return masked_lm_loss
```

```python
def get_mask(queries,passages,args,reader):
    queries_qlm = (queries[0][:args.bsize],queries[1][:args.bsize]) 
    passage_qlm = (passages[0][:args.bsize],passages[1][:args.bsize])

    input_ids = queries_qlm[0] ##query input ids

    probability_matrix = torch.full(input_ids.shape, args.mlm_probability) ##每个位置都是0.15
    query_mask = (queries_qlm[0] == 101) | (queries_qlm[0] == 100)|(queries_qlm[0] == 103)
    probability_matrix.masked_fill_(query_mask, value=0.0) ##特殊部位一定不能mask掉
            
    masked_indices = torch.bernoulli(probability_matrix).bool() ##生成mask

    labels_qlm = input_ids.clone()
    labels_qlm[~masked_indices] = -100 # We only compute loss on masked tokens
    probability_matrix_doc = torch.full(passage_qlm[0].shape, -100)
    labels_qlm = torch.cat([labels_qlm,probability_matrix_doc],axis = 1)

    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices ##其中80%被mask掉
    input_ids[indices_replaced] = reader.query_tokenizer.mask_token_id
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced ##其他的随机替换掉
    random_words = torch.randint(len(reader.query_tokenizer), input_ids.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    queries_qlm = (input_ids,queries_qlm[1]) ##现在，query已经被mask了
    
    return queries_qlm,passage_qlm,labels_qlm

```

【注】torch.full:

```
torch.full((2, 3), 3.141592)
tensor([[ 3.1416,  3.1416,  3.1416],
        [ 3.1416,  3.1416,  3.1416]])
```

torch.bernoulli:

```python
>>> a
tensor([[ 0.1737,  0.0950,  0.3609],
        [ 0.7148,  0.0289,  0.2676],
        [ 0.9456,  0.8937,  0.7202]])
>>> torch.bernoulli(a)
tensor([[ 1.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 1.,  1.,  1.]])
```



```python
from transformers import BertTokenizerFast
from colXLM.modeling.tokenization.utils import _split_into_batches

class QueryTokenizer():
    def __init__(self, query_maxlen):
        self.tok = BertTokenizerFast.from_pretrained('bert-base-multilingual-uncased')
        self.query_maxlen = query_maxlen ##32
        self.Q_marker_token, self.Q_marker_token_id = '[Q]', self.tok.convert_tokens_to_ids('[unused0]')
        self.cls_token, self.cls_token_id = self.tok.cls_token, self.tok.cls_token_id
        self.sep_token, self.sep_token_id = self.tok.sep_token, self.tok.sep_token_id
        self.mask_token, self.mask_token_id = self.tok.mask_token, self.tok.mask_token_id

    def __len__(self):
        return len(self.tok)

    def tokenize(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        tokens = [self.tok.tokenize(x, add_special_tokens=False) for x in batch_text]

        if not add_special_tokens:
            return tokens

        prefix, suffix = [self.cls_token, self.Q_marker_token], [self.sep_token]
        tokens = [prefix + lst + suffix + [self.mask_token] * (self.query_maxlen - (len(lst)+3)) for lst in tokens] ###补全

        return tokens

    def encode(self, batch_text, add_special_tokens=False):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        ids = self.tok(batch_text, add_special_tokens=False)['input_ids']

        if not add_special_tokens:
            return ids

        prefix, suffix = [self.cls_token_id, self.Q_marker_token_id], [self.sep_token_id]
        ids = [prefix + lst + suffix + [self.mask_token_id] * (self.query_maxlen - (len(lst)+3)) for lst in ids]

        return ids

    def tensorize(self, batch_text, bsize=None):
        assert type(batch_text) in [list, tuple], (type(batch_text))

        # add placehold for the [Q] marker
        batch_text = ['. ' + x for x in batch_text]

        obj = self.tok(batch_text, padding='max_length', truncation=True,
                       return_tensors='pt', max_length=self.query_maxlen)

        ids, mask = obj['input_ids'], obj['attention_mask']

        # postprocess for the [Q] marker and the [MASK] augmentation
        ids[:, 1] = self.Q_marker_token_id
        ids[ids == 0] = self.mask_token_id

        if bsize:
            batches = _split_into_batches(ids, mask, bsize)
            return batches

        return ids, mask
```





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