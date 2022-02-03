## 1. 简介

pytorch是与Numpy类似的张量（Tensor） 操作库，可以实现强大的GPU加速，同时还支持动态神经网络，这一点是现在很多主流框架如TensorFlow都不支持的。 PyTorch提供了两个高级功能： 

- 具有强大的GPU加速的张量计算（如Numpy） 
- 包含自动求导系统的深度神经网络

TensorFlow和Caffe都是命令式的编程语言，而且是静态的，首先必须构建一个神经网络，然后一次又一次使用相同的结构，如果想要改 变网络的结构，就必须从头开始。但是对于PyTorch，通过反向求导技术，可以让你零延迟地任意改变神经网络的行为，而且其实现速度快。正是这一灵活性是PyTorch对比TensorFlow的最大优势。



## 2. 自动微分

**autograd** 包是 PyTorch 中所有神经网络的核心，为 Tensors 上的所有操作提供自动微分。它是一个**由运行定义的框架**，这意味着以代码运行方式定义你的后向传播，并且每次迭代都可以不同。

1. 如果将torch.tensor的属性 **requires_grad** 设置为 True，则会开始跟踪针对 tensor 的所有操作：

```python
x = torch.ones(2, 2, requires_grad=True)
```

要停止跟踪历史记录（和使用内存），可以将代码块使用 with torch.no_grad(): 包装起来。例如，但在评估阶段我们不需要梯度，就可以这样做。

2. 对torch.tensor做一系列的计算，完成计算后，调用 .**backward()** 来自动**计算所有梯度**。该张量的梯度**将累积到 .grad 属性**中。

例如：

```python
y = x + 2
print(y)
```

输出：

```python
tensor([[3., 3.],
        [3., 3.]], grad_fn=<AddBackward0>)
```

这里的grad_fn指的是Function，它保存整个完整的计算过程的历史信息，是一个有向无环图。每个张量都有一个 .grad_fn 属性保存着**创建了张量的 Function 的引用**，（如果用户自己创建张量，则grad_fn 是 None ）。

针对 y 做更多的操作：

```python
z = y * y * 3
out = z.mean()
print(z, out)
```

输出：

```python
tensor([[27., 27.],
        [27., 27.]], grad_fn=<MulBackward0>) 
tensor(27., grad_fn=<MeanBackward0>)
```

3. 如果你想计算导数，你可以调用 Tensor.**backward()**。如果输出只是标量（即它包含一个元素数据），则不需要指定任何参数backward()，但是如果它有更多元素，则需要指定一个gradient 参数来指定张量的形状。

例如：

```
out.backward()
print(x.grad)
```

输出：

```python
tensor([[4.5000, 4.5000],
        [4.5000, 4.5000]])
```

我们现在后向传播，因为输出只是**一个标量**，out.backward() 等同于out.backward(torch.tensor(1.))。

要停止 tensor 历史记录的跟踪，您可以调用 .detach()，它将其与计算历史记录分离，并防止将来的计算被跟踪。



## 3. 神经网络

#### 3.1 定义神经网络

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
```

forward函数（前馈函数）需要自己定义，然后**反向传播函数**被自动通过 autograd 定义了。

可训练的参数可以通过调用 net.parameters() 返回。

#### 3.2 反向传播

首先，需要计算损失函数loss。这样，当我们调用 loss.backward()，**整个图都会微分**，而且所有的在图中的requires_grad=True 的张量将会让他们的 grad 张量累计梯度。需要先清空现存的梯度，要不然梯度将会和现存的梯度累计到一起。

```python
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)
```

输出：

```python
conv1.bias.grad before backward
tensor([0., 0., 0., 0., 0., 0.])
conv1.bias.grad after backward
tensor([-0.0054,  0.0011,  0.0012,  0.0148, -0.0186,  0.0087])
```

#### 3.3 更新网络参数

```
weight = weight - learning_rate * gradient
```

使用不同的更新规则，类似于 SGD, Nesterov-SGD, Adam, RMSProp 等。为了让这可行，我们建立了一个小包：torch.optim 实现了所有的方法。

```python
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
```



### 4. 常见操作

##### 4.1 损失函数

`nn.crossEntropyLoss`就是直接去拿logits和ground truth label进行比较，并不用提前做softmax！

```python
import torch
import torch.nn as nn
import math

criterion = nn.CrossEntropyLoss()
input = torch.Tensor([[-0.7715, -0.6205,-0.2562]]) ##输出的未经归一化的logits
target = torch.tensor([0])
loss = criterion(input, target)
print(loss)  ##1.3447
```

实际上，`nn.crossEntropyLoss`相当于nn.Softmax+log+nn.NLLLoss

```python
import numpy as np
m = nn.Softmax()
input=m(input)  ##做softmax归一，输出为[0.2606, 0.3031, 0.4363]
print(-np.log(0.2606)) ##这就是loss，为1.3447
```

![img](https://pic3.zhimg.com/80/v2-423f83896fca1af3179f203c062fdf55_1440w.png)

