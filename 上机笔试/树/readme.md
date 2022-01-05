### 1. 层次遍历

```python
queue = []
max_list = []
queue.append(root)
while not len(queue) == 0:
	ans = []
    for i in range(len(queue)):
    	front = queue[0]
        ans.append(front.val)
        if front.left:
        	queue.append(front.left)
        if front.right:
            queue.append(front.right)
        queue = queue[1:]
```



### 2. 找到从根节点到叶子节点的所有路径

```python
self.num = []
self.num_list = []

def find_path(self,root):
        if root and not root.left and not root.right:
            self.num_list.append(self.num+[str(root.val)])
            return
        if not root:
            self.num_list.append(self.num[:])
            print(self.num)
            return
        self.num.append(str(root.val))
        self.find_path(root.left)
        self.num.pop()

        self.num.append(str(root.val))
        self.find_path(root.right)
        self.num.pop()
```



### 3. 中序后继

```python
def inorderSuccessor(self, root: 'TreeNode', p: 'TreeNode') -> 'TreeNode': ##O(logn)
        cur = root
        result = None
        while (cur) :
            if (cur.val > p.val):
                result = cur
                cur = cur.left
            else:
                cur = cur.right
            
        return result

```



## 4. 前序、中序、后序遍历

#### 4.1 前序遍历

根->左->右

```c++
void preorder_traverse(Node* root){ //树的先序遍历,递归解法
    if(root == NULL)
        return;
    cout<<root->data<<"$";
    preorder_traverse(root->left_child);
    preorder_traverse(root->right_child);
}
```

迭代解法：先令根节点入栈。在栈空之前，每次取top节点，使右孩子入栈、左孩子入栈。

```c++
stack<Node*> Stack;
    Stack.push(root);
    while(!Stack.empty()){
        Node* top = Stack.top();
        cout<<top->data<<"&";
        Stack.pop();
        if(top->right_child)
            Stack.push(top->right_child);
        if(top->left_child)
            Stack.push(top->left_child);
```



#### 4.2 中序遍历

左->中->右

