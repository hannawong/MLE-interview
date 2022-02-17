# 手撕 logistic regression

```python
import torch
from torch.autograd import Variable

torch.manual_seed(2)
x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0], [4.0]]))
y_data = Variable(torch.Tensor([[0.0], [0.0], [1.0], [1.0]]))

class Model(torch.nn.Module):
	def __init__(self):
		super(Model, self).__init__()  
		self.linear = torch.nn.Linear(1, 1)  #输入维度和输出维度都为1

	def forward(self, x):
		y_pred = self.linear(x)
		return y_pred

model = Model() 

criterion = torch.nn.BCEWithLogitsLoss()  ##sigmoid + 
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)   #进行优化梯度下降
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001)
#Pytorch类方法正则化方法，添加一个weight_decay参数进行正则化

#befor training 
hour_var = Variable(torch.Tensor([[2.5]]))
y_pred = model(hour_var)
print("predict (before training)given", 4, 'is', float(model(hour_var).data[0][0]>0.5))

epochs = 40
for epoch in range(epochs):
	#计算grads和cost
	y_pred = model(x_data)   #x_data输入数据进入模型中
	loss = criterion(y_pred, y_data)
	print('epoch = ', epoch+1, loss)
	optimizer.zero_grad() #梯度清零
	loss.backward() #反向传播
	optimizer.step()  #优化迭代

#After training 
hour_var = Variable(torch.Tensor([[4.0]]))
y_pred = model(hour_var)
print("predict (after training)given", 4, 'is', float(model(hour_var).data[0][0]>0.5))
```

