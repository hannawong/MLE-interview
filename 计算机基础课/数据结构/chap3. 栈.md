# 栈

#### 1.1 调用栈

![img](https://pic1.zhimg.com/80/v2-6a5db014b28e6537583e9f8a8315f64b_1440w.jpeg)

有关栈帧的知识之后补充。

**消除递归**：

- 动机：递归函数的空间复杂度主要取决于最大递归深度（调用栈），而非递归实例总数
- 为隐式地维护调用栈，需要花费额外的时间、空间
- 所以，可以显式地维护调用栈，将递归版本改写为迭代版本。

**尾递归：**

- 线性递归中，若递归调用系“最后一步”，则称为尾递归。这是最简单的递归模式。
- 一旦抵达递归基，便会引发一连串的return，且返回地址相同；调用栈相应的连续pop().
- 故不难改写为迭代形式。越来越多的编译器可以自动识别并代为改写。
- 时间复杂度有常系数改进，空间复杂度或有渐进改进。

尾递归的调用栈：

![img](https://pic2.zhimg.com/80/v2-bdb990c8d29f020cd28a6f60e0881e7f_1440w.jpeg)

![img](https://pica.zhimg.com/80/v2-bd1d925f130a9f4e02b8c478cb54f277_1440w.jpeg)



#### 1.2 栈混洗

![img](https://pic2.zhimg.com/80/v2-59708a0b560d2a60a64255289e4dff1c_1440w.jpeg)

只允许：

- 将A的顶元素弹出并压入S，或
- 将S的顶元素弹出并压入B

经过一系列以上操作之后，A种元素全部转入B种，则称为A的一个栈混洗。

**问题一：**长度为n的序列，**可能的混洗总数SP(n)** = ?

- 设栈S在第k次pop()之后再度变空
- 则此时的栈混洗数量为SP(k-1)SP(n-k).
- 总共k可以取1~n，故栈混洗数目为$SP(n) = \sum_{k=1}^n SP(k-1)SP(n-k) = catalan(n) = \frac{(2n)!}{(n+1)!n!}$

卡特兰数就是栈混洗的数目。

**问题二：**如何甄别一个序列是否为栈混洗？

解法：直接模拟，O(n).

```python
##leetcode 946, 验证栈序列
class Solution:
    def validateStackSequences(self, pushed, popped): ##pushed:A, popped:B
        stack = [] ##中间的模拟栈
        ptr = 0
        cnt = 0
        n = len(pushed)
        while(True):
            cnt += 1
            if cnt > 2*n or ptr > n:
                break
            expect = popped[ptr]
            if len(stack): ##可以从中间栈中拿
                top = stack[-1]
                if expect == top:
                    stack.pop()
                    ptr += 1
                    continue
            ## 栈为空 or 栈顶不匹配
            if len(pushed) == 0: #再也没有元素了，一定为非法
                return False
            front = pushed[0]
            pushed = pushed[1:]
            stack.append(front)
        return len(pushed) == 0
```



#### 1.3 表达式求值

**中缀表达式求值**：

中缀表达式指的就是我们平时使用的那种表达式，例如(1+2)*3 - 1。

首先定义一个矩阵，表示栈顶运算符和当前运算符的优先等级：

[![img](https://camo.githubusercontent.com/af0c0fb5d4cab39161367028b3f3b36ca1598774e9f5c68a6868906af828d29a/68747470733a2f2f706963612e7a68696d672e636f6d2f38302f76322d39656436633061613534646136343933343365363632353436323665636162635f31343430772e706e67)](https://camo.githubusercontent.com/af0c0fb5d4cab39161367028b3f3b36ca1598774e9f5c68a6868906af828d29a/68747470733a2f2f706963612e7a68696d672e636f6d2f38302f76322d39656436633061613534646136343933343365363632353436323665636162635f31343430772e706e67)

使用一个操作数栈、一个操作符栈：

```python
class Solution:
    def calculate(self, s: str) -> int:
        s = s.replace(" ","")
        s+="$" ##末尾加入哨兵

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
            num_cnt = 0
            while s[i] in number:  ##读入尽可能多的数字
                num_cnt = num_cnt*10+int(s[i])
                i += 1
                if s[i] not in number:
                    num_stack.append(num_cnt)
            else:
            ### '<':栈顶运算符优先级更低，静待时机
                if compare(op_stack[-1],s[i]) == '<':
                    op_stack.append(s[i])
                    i += 1
            ### '=':右括号，或者已经到了结尾，终须了断
                elif compare(op_stack[-1],s[i]) == "=":
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
                        num_stack.append(num2//num1)
        print(num_stack) ##操作数栈必只有一个值，即答案。
        print(op_stack) ##操作符栈必为空
        return num_stack[0]
```

**逆波兰表达式求值**

直接用栈即可。

