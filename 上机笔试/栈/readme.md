# 栈

### 1. 中缀表达式求值

中缀表达式指的就是我们平时使用的那种表达式，例如(1+2)*3 - 1。

首先定义一个矩阵，表示栈顶运算符和当前运算符的优先等级。


![img](https://pica.zhimg.com/80/v2-9ed6c0aa54da649343e66254626ecabc_1440w.png)


  

```python
def operation(string):
    oper_dic = {"+":0,"-":1,"*":2,"/":3,"(":4,")":5,'$':6}
    prior = [['>','>','<','<','<','>','>'],['>','>','<','<','<','>','>'],['>','>','>','>','<','>','>'],['>','>','>','>','<','>','>'],['<','<','<','<','<','=',' '],[' ',' ',' ',' ',' ',' ',' '],['<','<','<','<','<',' ','=']]
    def compare(stack_top, new):
        stack_top_idx = oper_dic[stack_top]
        new_idx =  oper_dic[new]
        return prior[stack_top_idx][new_idx]


    number = '1234567890'
    op_stack = ['$'] ##哨兵
    num_stack = []
    i = 0
    while(len(op_stack)!=0):
        if string[i] in number:
            num_stack.append(int(string[i]))
            i += 1
            print(num_stack)
        else:
            ### '<':栈顶运算符优先级更低，静待时机
            if compare(op_stack[-1],string[i]) == '<':
                op_stack.append(string[i])
                i += 1
            ### '=':右括号，或者已经到了结尾，终须了断
            elif compare(op_stack[-1],string[i]) == "=":
                op_stack.pop()
                i += 1
            ### '>': 栈顶运算符优先级更高，时机成熟，实施相应计算
            else:
                op = op_stack.pop()
                num1 = num_stack.pop()
                num2 = num_stack.pop()
                if op == "+":
                    num_stack.append(num1+num2)
                if op == "-":
                    num_stack.append(num2-num1)
                if op == "*":
                    num_stack.append(num1*num2)
                if op == "/":
                    num_stack.append(num2/num1)
        print(num_stack)
        print(op_stack)


operation("(3*(1+2)-5)+2$")
```





#### 逆波兰表达式

逆波兰表达式就是专门为了栈求值而生的，用栈来计算十分简单，就是遇数字入栈、遇运算符则进行一元/二元运算，然后入栈。

![b40c27177059b3a67c1b889fcd14d4f](C:\Users\zh-wa\AppData\Local\Temp\WeChat Files\b40c27177059b3a67c1b889fcd14d4f.jpg)