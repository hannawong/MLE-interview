# Decoding 方法

人类选择的词并不是像机器选择的那样总是那些**条件概率最大的词**。这些概率大的词会发生正反馈，产生**循环**。从生成的结果也可以看出，机器生成的结果有大量**重复**。于是，人们尝试了各种办法对Beam Search进行改进。

### Sampling

Sample的概率依据就是解码器输出的词典中每个词的概率分布。相比于按概率“掐尖”，这样会增大所选词的范围，引入更多的随机性。有论文表示，这种随机采样的方法远好于Beam Search。但这其实也是有条件的，随机采样容易产生**前后不一致**的问题。而在开放闲聊领域，生成文本的**长度都比较短**，这种问题就被自然的淡化了。

采样的时候有一个可以控制的超参数，称为**温度**(temperature, T)。解码器的输出层后面通常会跟一个softmax函数来将输出概率归一化，通过改变T可以控制概率的形貌。softmax的公式如下，当T大的时候，概率分布趋向平均，随机性增大；当T小的时候，概率密度趋向于集中，即强者愈强，随机性降低，会更多地采样出“放之四海而皆准”的词汇。

#### **top-k采样**

这个方法就是在采样前将输出的**概率分布截断**，取出概率最大的k个词构成一个集合，然后将这个子集词的概率**再归一化**，最后从新的概率分布中采样词汇。这个办法据说可以获得比Beam Search好很多的效果，但也有一个问题，就是这个k不太好选。

为啥呢？因为这个概率分布变化比较大，有时候可能很均匀(flat)，有的时候比较集中(peaked)。对于集中的情况还好说，当分布均匀时，一个较小的k容易丢掉很多优质候选词。但如果k定的太大，这个方法又会退化回普通采样。

#### **核采样（Nucleus sampling)** -- top p采样

这种方法选择一个**最小**候选集，使得

​                                                                        $∑_{x∈V}P(x)>p$

选出来这个集合之后也和top-k采样一样，重新归一化集合内词的概率，并把集合外词的概率设为0。这种方式也称为top-p采样。

### **惩罚重复**

为了解决重复问题，还可以通过**惩罚因子**将出现过词的概率变小或者**强制不使用重复词**来解决。惩罚因子来自于 CTRL: A Conditional Transformer Language Model for Controllable Generation



代码：

```python
def top_k_top_p_filtering(logits, top_k=0, top_p=1.0, filter_value=-float("Inf")):
    """ 
     logits: logits distribution shape (batch size, vocabulary size)
     if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
     if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
    """
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits
```

