# SimpleTOD:Task-Oriented Dialog 端到端生成

一、问题

传统的任务型对话包含三部分：基于自然语言理解（NLU）的Dialog State Tracking、基于对话管理（DM）的Dialog Policy Prediction、基于自然语言生成方案（NLG）的答案生成。针对这三个组成部分，分别独立的进行有监督训练。例如，许多传统的对话系统在每次对话过程，都不会考虑整个对话过程，只是依赖于自然语言理解模块（NLU）的结果可靠地**传给后面的模块中**，这会导致error propagation。

二、简介

论文将面向任务型对话的三个部分，重铸为一个简单单向的language model任务。Simple TOD充分利用大型语言模型（如GPT-2）通过端到端的方式优化所有任务，同时建模子任务之间的内在依赖关系。

![img](https://pic1.zhimg.com/80/v2-b2132a58df3df8fbc0f4d01f0940b98e_1440w.png)

在第 t 轮对话中用户的输入记作$U_t$，对话系统的输出记作$S_t$。SimpleTOD模型将前t-1轮对话的用户输入+机器输出作为输入：$Ct=[U_0，S_0，U_1，S_1，...，U_t]$。将Ct作为输入，生成Belief State $B_t$。

$B_t = SimpleTOD(C_t)$

$B_t$是一个三元组：（domain，slot_name，value），将这个三元组转换为SQL查询语句，通过查询语句到数据库中检索，得到满足条件的结果。SimpleTOD将数据库的查询结果表示为Dt。Dt包含返回结果数量等信息。最后，SimpleTOD将Ct，Bt，Dt作为输入，得到Action At（即上图中粉红色文本框）

​                                                                 $A_t = SimpleTOD([C_t, B_t, D_t])$

Simple将所有的先验信息拼接成一个序列，生成一个Delexicalized Response St，

​                                                         $S_t = SimpleTOD([C_t, B_t, D_t, A_t])$

用数据库搜索结果的信息对$S_t$做Lexicalize，输出人类可阅读的文本。

#### 训练

在训练期间，就是把$C_t,B_t,D_t,A_t,S_t$拼接起来，一股脑输入；再预测的时候，就需要一一生成了。

使用的数据为Multi-domain Wizard-of-Oz（MultiWOZ），该数据包含10438个多回合对话，平均13.68个回合，跨越7个领域（餐厅，火车，景点，酒店，出租车，医院，警察）。由于警察和医院领域没有有效的测试分割，因此，未使用这两个领域。

![img](https://pic1.zhimg.com/80/v2-edcf512de8d80d03c00bd3ec2973e265_1440w.png)

![img](https://picx.zhimg.com/80/v2-50f3df95d48953901ebbf4b1701620db_1440w.png)

SimpleTOD除了端到端生成，还能够做Dialog State Tracking，并取得了State-of-the-art效果。