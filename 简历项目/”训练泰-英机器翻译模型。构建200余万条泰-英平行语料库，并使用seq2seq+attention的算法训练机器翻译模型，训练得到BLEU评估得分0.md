”训练泰-英机器翻译模型。构建200余万条泰-英平行语料库，并使用**seq2seq+attention**的算法训练机器翻译模型，训练得到**BLEU**评估得分0.59的离线翻译模型。目前数据库中的“商品英文名称”字段使用此翻译模型做泰-英翻译，大大降低了后续NLP算法的难度。 “



为什么不用transformer？-> 当时(2020年)还没有BERT的泰文预训练模型





- 我们将需要每个单词的唯一索引，以便稍后用作网络的输入和目标。为了跟踪所有这些，我们将使用一个名为`Lang`的辅助类，它具有 word→index（`word2index`）和index→word（`index2word`）的字典，以及用于稍后替换稀有单词的每个单词`word2count`的计数。

```python
SOS_token = 0
EOS_token = 1


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
```



这些文件都是Unicode格式，为了简化我们将Unicode字符转换为ASCII，使所有内容都小写，并去掉大多数标点符号。

```python
# 将Unicode字符串转换为纯ASCII, 感谢https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# 小写，修剪和删除非字母字符

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s
```



**编码器：**

输入句子中每个单词的ID，先经过embedding层把它们都变成向量，然后经过GRU得到输出。

```python
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

**解码器：**

在最简单的seq2seq解码器中，我们仅使用encoder的**最后一个输出**。 最后一个输出有时称为上下文向量，因为它编码整个序列的上下文。 该上下文向量用作解码器的初始隐藏状态。

在解码的每个步骤中，给予解码器输入token和隐藏状态。初始输入token是**开始字符串**`<SOS>`标记，**第一个隐藏状态是编码器的最后隐藏状态）**。

```
class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size) ##output语言的vocab大小为output_size
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

**带attention的decoder：**

```python
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
```

问题：

- 没有使用subword方法，而是对整个词进行编码，这会导致出现OOV问题 -- 如果在测试的时候出现一个训练的时候未出现的词，那么就导致出现OOV。->只能原封不动的写下来，或者随机初始化。不过好在，泰国站的商品名称变化不是很大






#### BLEU评估方法：

用谷歌翻译的结果当成标准答案，我们的翻译作为candidate。对于candidate的每个n-gram（n = 1，2，3，4），去统计这个n-gram在标答中有没有**不重复的**出现，如果真的出现了就+1.最后计算所有n-gram出现的概率。

问题：

- prefer短的翻译 -> 增加一个对长度的惩罚
- 对于那些词不同意同的词语，不能判断出来 -> 增加多个candidate